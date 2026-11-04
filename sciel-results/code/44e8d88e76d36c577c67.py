"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import time
import random
import sys

DIMENSIONS = [4, 5, 5]
TIME_LIMIT = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return (U, V, W) for Strassen 2x2x2 decomposition, each shape (4, 7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[0*2+1, 0] = -1; W[1*2+0, 0] = 1; W[1*2+1, 0] = -1
    
    # M2 = (A21+A22)B11
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[1*2+0, 1] = 1; W[1*2+1, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    W[0*2+1, 2] = 1; W[1*2+1, 2] = 1
    
    # M4 = A22(B21-B11)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1; W[1*2+0, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1; W[1*2+0, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[0*2+1, 6] = 1
    
    return U, V, W


def _build_48_rank_decomp(n, m, p):
    """Build the 48-rank decomposition: two 2x5x5 blocks, each using Strassen for 2x2x2 sub-block."""
    r_total = 48
    U = np.zeros((n*m, r_total))
    V = np.zeros((m*p, r_total))
    W = np.zeros((p*n, r_total))
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    for block in range(2):
        i_offset = block * 2
        c_offset = block * 24
        
        # Sub-block a: 2x2x2 Strassen (j in 0..1, k in 0..1)
        for i in range(2):
            for j in range(2):
                U[(i_offset+i)*m + j, c_offset:c_offset+7] = U_s[i*2+j, 0:7]
        for j in range(2):
            for k in range(2):
                V[j*p + k, c_offset:c_offset+7] = V_s[j*2+k, 0:7]
        for k in range(2):
            for i in range(2):
                W[k*n + (i_offset+i), c_offset:c_offset+7] = W_s[k*2+i, 0:7]
        
        # Sub-block b: 2x1x3 (j=0, k in 2..4): rank 2*3=6
        for i in range(2):
            for kk in range(3):
                k = 2 + kk
                c = c_offset + 7 + i*3 + kk
                U[(i_offset+i)*m + 0, c] = 1.0
                V[0*p + k, c] = 1.0
                W[k*n + (i_offset+i), c] = 1.0
        
        # Sub-block c: 2x1x3 (j=1, k in 2..4): rank 2*3=6
        for i in range(2):
            for kk in range(3):
                k = 2 + kk
                c = c_offset + 13 + i*3 + kk
                U[(i_offset+i)*m + 1, c] = 1.0
                V[1*p + k, c] = 1.0
                W[k*n + (i_offset+i), c] = 1.0
        
        # Sub-block d: 2x3x1 (j in 2..4, k=0): rank 2*3=6
        for i in range(2):
            for jj in range(3):
                j = 2 + jj
                c = c_offset + 19 + i*3 + jj
                U[(i_offset+i)*m + j, c] = 1.0
                V[j*p + 0, c] = 1.0
                W[0*n + (i_offset+i), c] = 1.0
        
        # Sub-block e: 2x3x1 (j in 2..4, k=1): rank 2*3=6
        for i in range(2):
            for jj in range(3):
                j = 2 + jj
                c = c_offset + 25 + i*3 + jj
                U[(i_offset+i)*m + j, c] = 1.0
                V[j*p + 1, c] = 1.0
                W[1*n + (i_offset+i), c] = 1.0
        
        # Sub-block f: 2x3x3 schoolbook (j in 2..4, k in 2..4): rank 2*9=18
        for i in range(2):
            for jj in range(3):
                for kk in range(3):
                    j = 2 + jj
                    k = 2 + kk
                    c = c_offset + 31 + i*9 + jj*3 + kk
                    U[(i_offset+i)*m + j, c] = 1.0
                    V[j*p + k, c] = 1.0
                    W[k*n + (i_offset+i), c] = 1.0
    
    return U, V, W, r_total


def _is_valid_fast(U, V, W, r, n, m, p):
    """Vectorized validity check."""
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    return np.allclose(T, 1.0, atol=1e-10)


def _try_remove_term(U, V, W, r, n, m, p, term_idx):
    """Try removing term term_idx by solving for re-weighting of remaining terms.
    Returns (success, new_U, new_V, new_W, new_r) or (False, None, None, None, None)."""
    # The tensor equation: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # If we fix all terms except term_idx, we get a linear system in the free variables of term_idx.
    # But actually, we want to REMOVE term_idx entirely, so we need to check if the remaining
    # r-1 terms can be re-weighted (scaled) to still produce the correct tensor.
    #
    # Actually, the proper approach: each term c contributes a rank-1 tensor T_c[i,j,k] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # We need sum_c T_c = identity tensor (1 for all i,j,k).
    # If we remove term t, we need sum_{c != t} alpha_c * T_c = identity tensor, for some scalars alpha_c.
    # This is a linear system: 100 equations, r-1 unknowns.
    
    if r - 1 < 100:
        # Underdetermined or just barely determined - might work
        pass
    
    # Build the 100x(r-1) matrix A where A[eq, col] = T_{c_col}[eq]
    # and solve A * alpha = ones(100)
    
    remaining = [c for c in range(r) if c != term_idx]
    r_rem = len(remaining)
    
    # Build A: 100 rows, r_rem columns
    # A[e, col] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for the (i,j,k) corresponding to row e
    A = np.zeros((n*m*p, r_rem))
    for col, c in enumerate(remaining):
        U_c = U[:, c]
        V_c = V[:, c]
        W_c = W[:, c]
        # Reshape
        U_t = U_c.reshape(n, m)
        V_t = V_c.reshape(m, p)
        W_t = W_c.reshape(p, n)
        # T[i,j,k] = U_t[i,j] * V_t[j,k] * W_t[k,i]
        # Flatten in order i*m*p + j*p + k
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = U_t[i,j] * V_t[j,k] * W_t[k,i]
        A[:, col] = T.flatten()
    
    b = np.ones(n*m*p)
    
    # Solve A * alpha = b
    try:
        alpha, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
    except:
        return False, None, None, None, None
    
    # Check if solution is good
    if np.max(np.abs(A @ alpha - b)) > 1e-8:
        return False, None, None, None, None
    
    # Check if alpha values are integer or half-integer (within tolerance)
    for a in alpha:
        if abs(a * 2 - round(a * 2)) > 1e-6:
            return False, None, None, None, None
    
    # Build new decomposition
    U_new = np.zeros((n*m, r_rem))
    V_new = np.zeros((m*p, r_rem))
    W_new = np.zeros((p*n, r_rem))
    for col, c in enumerate(remaining):
        U_new[:, col] = U[:, c] * alpha[col]
        V_new[:, col] = V[:, c]
        W_new[:, col] = W[:, c]
    
    # Verify
    if not _is_valid_fast(U_new, V_new, W_new, r_rem, n, m, p):
        return False, None, None, None, None
    
    return True, U_new, V_new, W_new, r_rem


def _greedy_rank_reduction(U, V, W, r, n, m, p, time_limit):
    """Greedy: repeatedly try to remove the 'most redundant' term."""
    start = time.time()
    for iteration in range(r):
        if time.time() - start > time_limit:
            break
        best_removal = None
        best_result = None
        for term_idx in range(r):
            if time.time() - start > time_limit:
                break
            success, U_new, V_new, W_new, r_new = _try_remove_term(U, V, W, r, n, m, p, term_idx)
            if success:
                best_removal = term_idx
                best_result = (U_new, V_new, W_new, r_new)
                break  # Take first successful removal
        if best_result is None:
            break
        U, V, W, r = best_result
    return U, V, W, r


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition, rank n*m*p."""
    r = n * m * p
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    for c, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
        U[i*m+j, c] = 1.0
        V[j*p+k, c] = 1.0
        W[k*n+i, c] = 1.0
    return U, V, W, r


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Start with schoolbook as ultimate fallback
    U_sb, V_sb, W_sb, r_sb = _build_schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb, r_sb)
    
    # Build the 48-rank decomposition
    U48, V48, W48, r48 = _build_48_rank_decomp(n, m, p)
    
    if _is_valid_fast(U48, V48, W48, r48, n, m, p):
        best_result = (U48, V48, W48, r48)
    else:
        # Fallback to schoolbook
        return {"U": U_sb[:, :r_sb].tolist(), "V": V_sb[:, :r_sb].tolist(), "W": W_sb[:, :r_sb].tolist()}
    
    # Now try to reduce rank via greedy term removal
    remaining_time = TIME_LIMIT - (time.time() - start_time) - 15  # Leave 15s safety margin
    if remaining_time > 5:
        U_red, V_red, W_red, r_red = _greedy_rank_reduction(U48, V48, W48, r48, n, m, p, remaining_time)
        if r_red < r48 and _is_valid_fast(U_red, V_red, W_red, r_red, n, m, p):
            best_result = (U_red, V_red, W_red, r_red)
    
    U_best, V_best, W_best, r_best = best_result
    return {"U": U_best[:, :r_best].tolist(), "V": V_best[:, :r_best].tolist(), "W": W_best[:, :r_best].tolist()}
# EVOLVE-BLOCK-END
