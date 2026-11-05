"""Matrix multiplication tensor decomposition for 3x3 * 3x8.

Strategy:
  1. Build a schoolbook decomposition (rank 72) as a safe fallback.
  2. Use randomised iterative compression (rank-revealing QR via SVD) to
     search for a valid decomposition at decreasing ranks.
  3. Verify exactness with rational reconstruction (fractions.Fraction) for
     integer/half-integer entries.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 8]
N, M, P = 3, 3, 8
DEADLINE = 150.0
# We want to finish construct() well before the 150s kill, so target 130s.
TARGET = 130.0


def _check_exact(U, V, W, tol=1e-12):
    """Verify that U, V, W exactly decompose the 3x3 * 3x8 multiplication tensor.
    Returns True if valid. U: (nm, r), V: (mp, r), W: (pn, r).
    Tensor T[i,j,k] should equal sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    where U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    """
    r = U.shape[1]
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We check a set of entries. For 3x3x8 = 72 entries, check all.
    U3 = U.reshape(N, M, r)
    V3 = V.reshape(M, P, r)
    W3 = W.reshape(P, N, r)
    
    for i in range(N):
        for j in range(M):
            for k in range(P):
                val = 0.0
                for c in range(r):
                    val += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                expected = 1.0 if (i == k or j == k) else 0.0
                # Wait, the multiplication tensor: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
                # The tensor T_{i,j,k} has T=1 iff the (i,k) entry of A*B depends on A[i,j]*B[j,k]
                # Actually for matrix mult (n x m) * (m x p) -> (n x p):
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k) in the product.
                # There's no "expected = 1 if ..." -- it's 1 for ALL (i,j,k).
                # Wait no. The multiplication tensor is defined as T_{i,j,k} = 1 for all i,j,k.
                # Because C_{ik} = sum_j A_{ij} B_{jk}, so every triple (i,j,k) contributes.
                # Let me re-examine.
                
                # Actually I need to be more careful. The tensor for matrix multiplication
                # C = A * B where A is n x m and B is m x p:
                # C_{ik} = sum_{j=0}^{m-1} A_{ij} B_{jk}
                # The "multiplication tensor" T has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
                # Because each term A[i,j]*B[j,k] contributes to C[i,k].
                
                # So the expected value is 1.0 for all (i,j,k).
                if abs(val - 1.0) > tol:
                    return False
    return True


def _check_exact_rational(U, V, W):
    """Check exactness using Fraction for entries that are integers or half-integers."""
    r = U.shape[1]
    U3 = U.reshape(N, M, r)
    V3 = V.reshape(M, P, r)
    W3 = W.reshape(P, N, r)
    
    for i in range(N):
        for j in range(M):
            for k in range(P):
                val = Fraction(0, 1)
                for c in range(r):
                    uf = Fraction(int(round(U3[i, j, c] * 2)), 2)
                    vf = Fraction(int(round(V3[j, k, c] * 2)), 2)
                    wf = Fraction(int(round(W3[k, i, c] * 2)), 2)
                    val += uf * vf * wf
                if val != 1:
                    return False
    return True


def _snap_half_int(arr):
    """Snap array entries to nearest half-integer."""
    return np.round(arr * 2) / 2.0


def _schoolbook():
    """Build the schoolbook decomposition of rank 72."""
    rank = N * M * P
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1.0
                V[j * P + k, col] = 1.0
                W[k * N + i, col] = 1.0
                col += 1
    return U, V, W


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k]=1 for all i,j,k."""
    return np.ones((n, m, p), dtype=np.float64)


def _try_rank(rank, time_budget=5.0, seed=0):
    """Try to find a valid rank-`rank` decomposition using iterative compression.
    Returns (U, V, W) if found, else None.
    """
    start = time.time()
    T = _build_tensor(N, M, P)  # shape (N, M, P), all ones
    
    # We need to find U (NM x rank), V (MP x rank), W (PN x rank)
    # such that for all (i,j,k): sum_c U[i*M+j,c] * V[j*P+k,c] * W[k*N+i,c] = T[i,j,k] = 1
    
    # Approach: fix two of the matrices randomly, solve for the third.
    # Actually, a common approach for tensor decomposition search:
    # 1. Randomly initialize U, V, W
    # 2. Iteratively: fix U, V -> solve for W (least squares), fix U, W -> solve for V, etc.
    # 3. Check if the reconstruction is exact.
    
    rng = np.random.RandomState(seed)
    
    for trial in range(30):
        if time.time() - start > time_budget:
            break
        
        # Random initialization with small values
        U = rng.randn(N * M, rank) * 0.1
        V = rng.randn(M * P, rank) * 0.1
        W = rng.randn(P * N, rank) * 0.1
        
        # Iterative refinement
        for it in range(50):
            if time.time() - start > time_budget:
                break
            
            # Fix U, V: solve for W
            # For each (k, i): sum_c W[k*N+i, c] * U[i*M+j, c] * V[j*P+k, c] = 1 for all j
            # This is a system: for fixed (k,i), we have M equations (one per j) and rank unknowns W[k*N+i,:].
            # A_{j,c} = U[i*M+j, c] * V[j*P+k, c], then A @ W[k*N+i,:] = ones(M)
            
            for k in range(P):
                for i in range(N):
                    # Build M x rank matrix
                    A = np.zeros((M, rank))
                    for j in range(M):
                        A[j, :] = U[i * M + j, :] * V[j * P + k, :]
                    b = np.ones(M)
                    # Solve least squares
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k * N + i, :] = sol
                    except np.linalg.LinAlgError:
                        pass
            
            # Fix U, W: solve for V
            for j in range(M):
                for k in range(P):
                    # For fixed (j,k): sum_c V[j*P+k,c] * U[i*M+j,c] * W[k*N+i,c] = 1 for all i
                    # N equations, rank unknowns
                    A = np.zeros((N, rank))
                    for i in range(N):
                        A[i, :] = U[i * M + j, :] * W[k * N + i, :]
                    b = np.ones(N)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j * P + k, :] = sol
                    except np.linalg.LinAlgError:
                        pass
            
            # Fix V, W: solve for U
            for i in range(N):
                for j in range(M):
                    # For fixed (i,j): sum_c U[i*M+j,c] * V[j*P+k,c] * W[k*N+i,c] = 1 for all k
                    # P equations, rank unknowns
                    A = np.zeros((P, rank))
                    for k in range(P):
                        A[k, :] = V[j * P + k, :] * W[k * N + i, :]
                    b = np.ones(P)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i * M + j, :] = sol
                    except np.linalg.LinAlgError:
                        pass
            
            # Check convergence
            U3 = U.reshape(N, M, rank)
            V3 = V.reshape(M, P, rank)
            W3 = W.reshape(P, N, rank)
            max_err = 0.0
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        val = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            if max_err < 1e-10:
                # Snap to half-integers and verify
                U_s = _snap_half_int(U)
                V_s = _snap_half_int(V)
                W_s = _snap_half_int(W)
                if _check_exact_rational(U_s, V_s, W_s):
                    return U_s, V_s, W_s
                # Try without snapping
                if _check_exact(U, V, W, tol=1e-12):
                    return U, V, W
        
        # Also try a different randomization strategy: use the SVD of a random perturbation
        # of the schoolbook decomposition to get a compressed version.
        if trial % 10 == 0:
            # Start from schoolbook and compress
            U0, V0, W0 = _schoolbook()
            # Add small random noise
            U0 = U0 + rng.randn(N * M, N * M * P) * 1e-6
            V0 = V0 + rng.randn(M * P, N * M * P) * 1e-6
            W0 = W0 + rng.randn(P * N, N * M * P) * 1e-6
            
            # Use truncated SVD on a "contracted" tensor
            # Flatten the tensor and use ALS
            U = U0[:, :rank]
            V = V0[:, :rank]
            W = W0[:, :rank]
            
            for it in range(30):
                if time.time() - start > time_budget:
                    break
                for k in range(P):
                    for i in range(N):
                        A = np.zeros((M, rank))
                        for j in range(M):
                            A[j, :] = U[i * M + j, :] * V[j * P + k, :]
                        b = np.ones(M)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            W[k * N + i, :] = sol
                        except:
                            pass
                for j in range(M):
                    for k in range(P):
                        A = np.zeros((N, rank))
                        for i in range(N):
                            A[i, :] = U[i * M + j, :] * W[k * N + i, :]
                        b = np.ones(N)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            V[j * P + k, :] = sol
                        except:
                            pass
                for i in range(N):
                    for j in range(M):
                        A = np.zeros((P, rank))
                        for k in range(P):
                            A[k, :] = V[j * P + k, :] * W[k * N + i, :]
                        b = np.ones(P)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            U[i * M + j, :] = sol
                        except:
                            pass
                
                U3 = U.reshape(N, M, rank)
                V3 = V.reshape(M, P, rank)
                W3 = W.reshape(P, N, rank)
                max_err = 0.0
                for i in range(N):
                    for j in range(M):
                        for k in range(P):
                            val = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                if max_err < 1e-10:
                    U_s = _snap_half_int(U)
                    V_s = _snap_half_int(V)
                    W_s = _snap_half_int(W)
                    if _check_exact_rational(U_s, V_s, W_s):
                        return U_s, V_s, W_s
                    if _check_exact(U, V, W, tol=1e-12):
                        return U, V, W
    
    return None


def construct():
    """Construct a tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U': ..., 'V': ..., 'W': ...} with integer/half-integer entries.
    """
    start_time = time.time()
    
    # Safe fallback: schoolbook rank 72
    U_fallback, V_fallback, W_fallback = _schoolbook()
    
    # Try to find lower-rank decompositions
    # Theoretical lower bounds: max(nm, mp, pn) = max(9, 24, 24) = 24
    # Known best for 3x3x3 is 23, so for 3x3x8 we might be able to do better than 72.
    # Try ranks from a reasonable range down.
    
    # Candidate ranks to try (in order)
    candidate_ranks = [30, 32, 35, 40, 45, 50, 55, 60, 65, 72]
    
    for rank in candidate_ranks:
        if time.time() - start_time > TARGET:
            break
        remaining = TARGET - (time.time() - start_time)
        if remaining < 2.0:
            break
        # Allocate time per rank attempt
        time_per_rank = min(10.0, remaining / max(1, len(candidate_ranks) - candidate_ranks.index(rank)))
        result = _try_rank(rank, time_budget=time_per_rank, seed=rank * 1000)
        if result is not None:
            U, V, W = result
            # Final verification
            if _check_exact_rational(U, V, W) or _check_exact(U, V, W):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist(),
                }
    
    # Fallback to schoolbook
    return {
        "U": U_fallback.tolist(),
        "V": V_fallback.tolist(),
        "W": W_fallback.tolist(),
    }


# EVOLVE-BLOCK-END
