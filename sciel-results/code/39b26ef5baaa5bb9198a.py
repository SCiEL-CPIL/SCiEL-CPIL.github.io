"""Matrix multiplication 3x5x7: search for low-rank exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]
n, m, p = DIMENSIONS
N = n * m  # 15
M = m * p  # 35
P = p * n  # 21
RANK_SCHOOLBOOK = n * m * p  # 105

# Deadline guard
T_START = time.time()
DEADLINE = 120.0  # seconds total budget; leave margin


def _elapsed():
    return time.time() - T_START


def _time_left():
    return DEADLINE - _elapsed() - 5.0  # 5s safety margin


def _build_target_tensor():
    """
    Build the target matrix multiplication tensor T_target of shape (N, M, P).
    T_target[i*m+j, j*p+k, k*n+i] = 1
    """
    T_target = np.zeros((N, M, P), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    return T_target


def _check_exact(U, V, W, tol=1e-9):
    """Check if the decomposition exactly reproduces the matmul tensor."""
    # T_decomposed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = _build_target_tensor()
    diff = np.abs(T_decomp - T_target)
    return np.max(diff) < tol


def _verify_half_int(U, V, W, tol=1e-6):
    """Verify that all entries are half-integers and the decomposition is exact."""
    for A in [U, V, W]:
        A2 = 2 * A
        if np.max(np.abs(A2 - np.round(A2))) > tol:
            return False
    return _check_exact(U, V, W)


def _snap_to_half_ints(A, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    B = np.array(A, dtype=float)
    B2 = 2 * B
    B2_rounded = np.round(B2)
    if np.max(np.abs(B2 - B2_rounded)) < tol:
        return B2_rounded / 2.0
    return B


def _solve_for_W(U, V, T_target, tol=1e-9):
    """
    Given U (N, r) and V (M, r), solve for W (P, r) such that
    sum_r U[i,r] V[j,r] W[k,r] = T_target[i,j,k]
    
    This is a linear system:
    Let A be a matrix of size (N*M*P, r*P) ? No.
    The equation is linear in W.
    For each r, let u_r be col r of U, v_r be col r of V.
    The term is u_r \otimes v_r \otimes w_r.
    
    We can flatten the tensor equation.
    T_target_flat = sum_r (u_r \otimes v_r) \otimes w_r? No.
    
    Let's use the property:
    T_target[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    
    This is equivalent to:
    For each pair (i,j), we have a vector over k:
    T_target[i,j,:] = sum_r (U[i,r] V[j,r]) W[:,r]
    
    Let C_{ij} be the vector T_target[i,j,:].
    Let x_{ij,r} = U[i,r] V[j,r].
    Then C_{ij} = X_{ij} W, where X_{ij} is a row vector of length r, and W is (P, r)??
    No, W is (P, r). The sum is over r.
    C_{ij}[k] = sum_r X_{ij}[r] W[k,r].
    
    This can be written as a large linear system for all (i,j) pairs.
    Number of equations: N * M * P = 15 * 35 * 21 = 11025.
    Number of unknowns: P * r = 21 * r.
    
    Let A_sys be a matrix of size (N*M*P, P*r).
    Let b_sys be the flattened T_target.
    
    For each (i,j,k) and each r:
    The variable is W[k,r].
    The coefficient is U[i,r] * V[j,r].
    
    So, row index idx = (i*M + j)*P + k.
    Col index cidx = k*r + r_idx? No, W is (P, r).
    Let's flatten W to (P*r,). W_flat[k*r + r_idx] = W[k, r_idx].
    
    A_sys[idx, k*r + r_idx] += U[i,r_idx] * V[j,r_idx]
    
    Then solve A_sys * W_flat = b_sys.
    """
    r = U.shape[1]
    num_eqs = N * M * P
    num_unk = P * r
    
    A_sys = np.zeros((num_eqs, num_unk), dtype=np.float64)
    b_sys = T_target.flatten()
    
    # Fill A_sys
    # idx corresponds to (i, j, k)
    # cidx corresponds to (k, r)
    for i in range(N):
        for j in range(M):
            # Extract relevant parts of U and V
            u_row = U[i, :]  # (r,)
            v_row = V[j, :]  # (r,)
            coeff = u_row * v_row  # (r,)
            
            for k in range(P):
                base_idx = (i * M + j) * P + k
                base_cidx = k * r
                A_sys[base_idx, base_cidx:base_cidx+r] = coeff
    
    # Solve least squares
    try:
        W_flat, residuals, rank, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
    except np.linalg.LinAlgError:
        return None
    
    W = W_flat.reshape(P, r)
    
    # Check residual
    if residuals is not None and len(residuals) > 0:
        if residuals[0] > 1e-6:
            return None
    
    return W


def _try_rank_linear(r, max_attempts=200, seed=42):
    """
    Try to find a rank r decomposition by random sparse U, V and solving for W.
    """
    T_target = _build_target_tensor()
    rng = np.random.RandomState(seed)
    
    for attempt in range(max_attempts):
        if _time_left() < 5:
            break
            
        # Generate sparse U and V with entries in {-1, 0, 1}
        # Density should be low to allow for exact solutions
        density = 0.2
        U = (rng.rand(N, r) < density) * 2 - 1
        U[rng.rand(N, r) >= density] = 0
        
        V = (rng.rand(M, r) < density) * 2 - 1
        V[rng.rand(M, r) >= density] = 0
        
        # Solve for W
        W = _solve_for_W(U, V, T_target)
        if W is None:
            continue
            
        # Check if W is close to half-integers
        W_s = _snap_to_half_ints(W, tol=1e-5)
        if _verify_half_int(U, V, W_s, tol=1e-5):
            # Found a valid decomposition
            return U, V, W_s, r
            
    return None, None, None, None


def _schoolbook():
    """Return the schoolbook decomposition of rank 105."""
    rank = RANK_SCHOOLBOOK
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    # Start with schoolbook as fallback
    result = _schoolbook()
    best_rank = RANK_SCHOOLBOOK
    best_result = result
    
    # Try linear algebra search for lower ranks
    # We try ranks from 100 down to 50
    # Note: This is a heuristic search. It may not find a solution.
    ranks_to_try = [104, 103, 102, 101, 100, 95, 90, 85, 80]
    
    for r in ranks_to_try:
        if _time_left() < 10:
            break
        if r >= best_rank:
            continue
            
        U, V, W, found_rank = _try_rank_linear(r, max_attempts=50, seed=int(r*100))
        if U is not None:
            # Convert to list of lists with exact values
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            # Verify once more
            U_np = np.array(U_list)
            V_np = np.array(V_list)
            W_np = np.array(W_list)
            if _verify_half_int(U_np, V_np, W_np):
                best_rank = found_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                # Continue to try lower ranks
                continue
    
    return best_result
