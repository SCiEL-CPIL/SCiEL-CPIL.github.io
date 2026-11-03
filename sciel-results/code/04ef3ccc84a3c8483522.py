"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # M1
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            # M2
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] -= 1
            col += 1
            # M3
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            # M4
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            # M5
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1
            W[k1*n + 0, col] += 1
            col += 1
            # M6
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            # M7
            U[0*m + j1, col] += 1
            U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
    
    # Remaining j=4: standard products
    for k in range(p):
        U[0*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 0, col] += 1
        col += 1
        U[1*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 1, col] += 1
        col += 1
        
    assert col == rank, f"Expected col={rank}, got {col}"
    return U, V, W

def _verify_full_fast(U, V, W, n, m, p, tol=1e-9):
    """
    Fast vectorized verification.
    """
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
        
    # Pre-compute V^T for efficiency? 
    # S_x = (U * diag(W[x,:])) @ V^T
    # U is (nm, R), V is (mp, R)
    # W[x,:] is (R,)
    
    # We can pre-compute U * V^T for all R? No, that's huge.
    # But we can compute U @ V^T once? 
    # No, the contraction is sum_r U[:,r] V[:,r]^T W[x,r].
    # This is element-wise product of (U @ V^T) with W? No.
    # It is a tensor contraction.
    
    # Let's just loop over x (12 iterations) and do matrix mult.
    # U: (10, R), V: (30, R), W: (12, R)
    # For each x:
    #   w = W[x] (R,)
    #   A = U * w  (10, R) -> broadcast
    #   S = A @ V.T (10, 30)
    
    V_T = V.T  # (R, mp)
    
    for x in range(p * n):
        w = W[x, :]  # (R,)
        # A = U * w[None, :]  (10, R)
        # S = A @ V_T  (10, 30)
        S = (U * w[None, :]) @ V_T
        
        # Construct target slice T_x
        # i = x % n, k = x // n
        i = x % n
        k = x // n
        
        # Target is 10x30 matrix.
        # Non-zeros at (y, z) where y = i*m + j, z = j*p + k
        # j ranges 0..m-1
        # We can check specific entries instead of building full matrix
        # But S is already computed.
        
        # Check valid entries
        for j in range(m):
            y = i * m + j
            z = j * p + k
            if abs(S[y, z] - 1.0) > tol:
                return False
        
        # Check invalid entries
        # We need to ensure all other entries are 0.
        # S is 10x30.
        # Create a mask of valid positions
        # valid_mask = np.zeros((n*m, m*p), dtype=bool)
        # for j in range(m):
        #     valid_mask[i*m+j, j*p+k] = True
        # if not np.allclose(S[~valid_mask], 0.0, atol=tol):
        #     return False
        
        # Faster: sum of squares of invalid entries
        # valid_mask construction is cheap (10x30)
        valid_mask = np.zeros((n*m, m*p), dtype=bool)
        for j in range(m):
            valid_mask[i*m+j, j*p+k] = True
        
        if np.any(np.abs(S[~valid_mask]) > tol):
            return False
            
    return True

def _try_eliminate_terms_fast(U, V, W, n, m, p, max_eliminations=20):
    """
    Try to eliminate redundant terms one by one using fast verification.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    norms = []
    for r in range(current_rank):
        norm_sq = np.sum(current_U[:, r]**2) + np.sum(current_V[:, r]**2) + np.sum(current_W[:, r]**2)
        norms.append(norm_sq)
        
    order = np.argsort(norms)
    
    for r in order:
        if _time_left() < 30:
            break
        if current_rank <= 1:
            break
            
        # Check if term r can be removed
        # We don't need to modify arrays, just verify with r ignored?
        # No, we need to verify the reduced set.
        # Masking is cheap.
        
        mask = np.ones(current_rank, dtype=bool)
        mask[r] = False
        
        U_test = current_U[:, mask]
        V_test = current_V[:, mask]
        W_test = current_W[:, mask]
        
        if _verify_full_fast(U_test, V_test, W_test, n, m, p):
            current_U = U_test
            current_V = V_test
            current_W = W_test
            current_rank -= 1
            
    return current_U, current_V, current_W, current_rank

def _try_pairwise_elimination_fast(U, V, W, n, m, p):
    """
    Try to eliminate pairs of terms simultaneously.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    norms = []
    for r in range(current_rank):
        norm_sq = np.sum(current_U[:, r]**2) + np.sum(current_V[:, r]**2) + np.sum(current_W[:, r]**2)
        norms.append(norm_sq)
        
    order = np.argsort(norms)
    candidates = order[:30]  # Consider more candidates
    
    for idx1 in range(len(candidates)):
        if _time_left() < 20:
            break
        r1 = candidates[idx1]
        for idx2 in range(idx1 + 1, len(candidates)):
            r2 = candidates[idx2]
            
            mask = np.ones(current_rank, dtype=bool)
            mask[r1] = False
            mask[r2] = False
            
            U_test = current_U[:, mask]
            V_test = current_V[:, mask]
            W_test = current_W[:, mask]
            
            if _verify_full_fast(U_test, V_test, W_test, n, m, p):
                return U_test, V_test, W_test, current_rank - 2
    
    return current_U, current_V, current_W, current_rank

def construct():
    n, m, p = DIMENSIONS
    
    U, V, W = _build_54_rank(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = 54
    
    if not _verify_full_fast(U, V, W, n, m, p):
        # Fallback
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Single elimination
    if _time_left() > 50:
        U_reduced, V_reduced, W_reduced, new_rank = _try_eliminate_terms_fast(U, V, W, n, m, p)
        if new_rank < best_rank:
            best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
            best_rank = new_rank
            U, V, W = U_reduced, V_reduced, W_reduced
    
    # Pairwise elimination
    if _time_left() > 20:
        U_reduced, V_reduced, W_reduced, new_rank = _try_pairwise_elimination_fast(U, V, W, n, m, p)
        if new_rank < best_rank:
            best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
            best_rank = new_rank
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
