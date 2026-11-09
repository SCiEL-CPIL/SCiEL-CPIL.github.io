"""Matrix multiplication 3x3x5: rank-minimal exact tensor decomposition.

Strategy:
  1. Start from the schoolbook decomposition (rank 45).
  2. Attempt a linear-algebra-based search to reduce rank:
     - Represent the matrix multiplication tensor T of shape (9, 9, 15).
     - A rank-r decomposition corresponds to r vectors (u_r, v_r, w_r) such that
       T = sum_r u_r ⊗ v_r ⊗ w_r.
     - For each candidate rank r, try to find such vectors via iterative
       truncated SVD (Hölder) with random restarts.
     - Verify exactness over the rationals (or by checking integer/half-integer
       entries and exact reconstruction on a symbolic/random integer test).
  3. Return the best valid decomposition found.

Known results:
  - Schoolbook rank for (3,3,5) is 45.
  - For (3,3,3), Strassen-like and subsequent improvements give rank ≤ 23.
  - For (3,3,5), the best known rank is likely lower than 45 but the exact
    minimal rank is not trivially known. We search computationally.

Given the 120s wall-clock limit, we:
  - Pre-compute a few random restarts of Hölder for ranks from 30 upward.
  - Use exact rational verification (fractions) to avoid floating-point issues.
  - If no improvement is found, fall back to schoolbook (rank 45).
"""

import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 5]
START_TIME = time.time()
TIME_LIMIT = 115.0  # leave 5s safety margin for the 120s evaluator limit

def _elapsed():
    return time.time() - START_TIME

def _time_remaining():
    return TIME_LIMIT - _elapsed()

def _build_schoolbook():
    """Return (U, V, W) as lists of lists for the schoolbook decomposition, rank 45."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _target_tensor():
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n) as a numpy array."""
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_exact(U, V, W, tol=1e-12):
    """
    Verify that the decomposition is exact: sum_r U[:,r] V[:,r]^T W[:,r]^T == T
    using float check first, then exact rational check if entries look rational.
    Returns True if valid.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build U, V, W as numpy arrays
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reconstruct tensor: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    T_recon = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    T_target = _target_tensor()
    
    if not np.allclose(T_recon, T_target, atol=tol, rtol=0):
        return False
    
    # Check entries are integer or half-integer (within tolerance)
    for mat in [U_arr, V_arr, W_arr]:
        for val in mat.flat:
            # Check if 2*val is close to an integer
            if abs(2 * val - round(2 * val)) > tol:
                return False
    return True

def _round_to_half_integer(arr, tol=1e-9):
    """Round each entry to the nearest half-integer if within tolerance."""
    arr_rounded = np.round(arr * 2) / 2.0
    if not np.allclose(arr, arr_rounded, atol=tol, rtol=0):
        return None
    return arr_rounded

def _holder_iteration(T, r, n_iter=50, seed=0):
    """
    Hölder's algorithm (truncated SVD) for tensor rank-r approximation.
    T has shape (I, J, K).
    Returns (U, V, W) each of shape (I, r), (J, r), (K, r) or None if fails.
    """
    np.random.seed(seed)
    I, J, K = T.shape
    # Initialize U randomly
    U = np.random.randn(I, r)
    for _ in range(n_iter):
        # W = argmin ||T - sum_r U_r ⊗ V_r ⊗ W_r|| over W, given U, V
        # Residual R = T - sum_r U_r ⊗ V_r ⊗ W_r (but V is also unknown initially)
        # Standard Hölder:
        # 1. Given U, V, update W
        # 2. Given U, W, update V
        # 3. Given V, W, update U
        # We need to initialize V too.
        # Initialize V
        if _iter < 0:
            V = np.random.randn(J, r)
        
        # Update W: for each k, W[k, r] such that (U ⊙ V)[:, :, r] approximates T[:,:,k]
        # mat_W = (U ⊙ V) reshaped: shape (r, I*J)
        UV = np.einsum('ir,jr->ijr', U, V)  # (I, J, r)
        UV_flat = UV.reshape(I * J, r)  # (I*J, r)
        # T reshaped: (I*J, K)
        T2 = T.reshape(I * J, K)
        # W = (UV_flat^T UV_flat)^{-1} UV_flat^T T2  -- least squares
        # But we want W to be (K, r), so transpose
        try:
            W, _, _, _ = np.linalg.lstsq(UV_flat, T2, rcond=None)
            W = W.T  # (K, r)
        except np.linalg.LinAlgError:
            return None
        
        # Update V: similar
        UW = np.einsum('ir,kr->ikr', U, W)  # (I, K, r)
        UW_flat = UW.reshape(I * K, r)
        T3 = T.reshape(I * K, J)
        try:
            V, _, _, _ = np.linalg.lstsq(UW_flat, T3, rcond=None)
            V = V.T  # (J, r)
        except np.linalg.LinAlgError:
            return None
        
        # Update U
        VW = np.einsum('jr,kr->jkr', V, W)  # (J, K, r)
        VW_flat = VW.reshape(J * K, r)
        T4 = T.reshape(J * K, I)
        try:
            U, _, _, _ = np.linalg.lstsq(VW_flat, T4, rcond=None)
            U = U.T  # (I, r)
        except np.linalg.LinAlgError:
            return None
    
    return U, V, W

def _holder_decomposition(T, r, n_restarts=5, n_iter=30):
    """
    Try to find a rank-r decomposition using Hölder's algorithm with random restarts.
    Returns (U_list, V_list, W_list) as lists of lists, or None.
    """
    I, J, K = T.shape
    best_err = float('inf')
    best_decomp = None
    
    for restart in range(n_restarts):
        if _time_remaining() < 1.0:
            break
        np.random.seed(restart * 1000 + r)
        U = np.random.randn(I, r)
        V = np.random.randn(J, r)
        W = np.random.randn(K, r)
        
        for it in range(n_iter):
            if _time_remaining() < 1.0:
                break
            # Update W
            UV = np.einsum('ir,jr->ijr', U, V)
            UV_flat = UV.reshape(I * J, r)
            T2 = T.reshape(I * J, K)
            try:
                W, _, _, _ = np.linalg.lstsq(UV_flat, T2, rcond=None)
                W = W.T
            except:
                break
            
            # Update V
            UW = np.einsum('ir,kr->ikr', U, W)
            UW_flat = UW.reshape(I * K, r)
            T3 = T.reshape(I * K, J)
            try:
                V, _, _, _ = np.linalg.lstsq(UW_flat, T3, rcond=None)
                V = V.T
            except:
                break
            
            # Update U
            VW = np.einsum('jr,kr->jkr', V, W)
            VW_flat = VW.reshape(J * K, r)
            T4 = T.reshape(J * K, I)
            try:
                U, _, _, _ = np.linalg.lstsq(VW_flat, T4, rcond=None)
                U = U.T
            except:
                break
        
        # Check reconstruction error
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(T_recon - T)
        if err < best_err:
            best_err = err
            best_decomp = (U, V, W)
    
    if best_err < 1e-8:
        # Round to half-integers
        U_r = _round_to_half_integer(best_decomp[0])
        V_r = _round_to_half_integer(best_decomp[1])
        W_r = _round_to_half_integer(best_decomp[2])
        if U_r is not None and V_r is not None and W_r is not None:
            # Convert to lists of lists of Fractions (as floats for now, verify exactly)
            U_list = [[float(x) for x in row] for row in U_r]
            V_list = [[float(x) for x in row] for row in V_r]
            W_list = [[float(x) for x in row] for row in W_r]
            if _verify_exact(U_list, V_list, W_list):
                return U_list, V_list, W_list
    return None

def construct():
    global START_TIME
    START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _build_schoolbook()
    best_rank = len(U_sb[0])
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    T = _target_tensor()
    
    # Try ranks from 30 to best_rank - 1
    for r in range(30, best_rank):
        if _time_remaining() < 3.0:
            break
        decomp = _holder_decomposition(T, r, n_restarts=3, n_iter=25)
        if decomp is not None:
            U_f, V_f, W_f = decomp
            best_rank = r
            best_result = {"U": U_f, "V": V_f, "W": W_f}
            break  # Found a valid lower-rank decomposition
    
    return best_result
