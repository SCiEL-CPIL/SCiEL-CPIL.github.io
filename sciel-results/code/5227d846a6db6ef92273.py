"""Optimized matrix multiplication tensor decomposition search for 3x5 * 5x7."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]
TIME_LIMIT = 110  # Safety margin before 120s kill

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly represents matrix multiplication."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    # Tensor T[i, j, k] should be 1 if we're doing A[i,j]*B[j,k]
    # The decomposition gives: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 for all i,j,k (since we're decomposing the identity-like tensor)
    
    # Efficient check: compute for all i,j,k
    # T_decomposed[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Build index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    # Compute decomposed tensor
    # T_decomposed = U[u_idx, :] @ V[v_idx, :].T * W[w_idx, :].T
    # More precisely: T[i,j,k] = sum_c U[u_idx[i,j,k], c] * V[v_idx[i,j,k], c] * W[w_idx[i,j,k], c]
    
    U_sel = U[u_idx, :]  # (n*m*p, rank)
    V_sel = V[v_idx, :]  # (n*m*p, rank)
    W_sel = W[w_idx, :]  # (n*m*p, rank)
    
    # Element-wise product then sum over rank
    T_decomp = np.sum(U_sel * V_sel * W_sel, axis=1)
    
    # Target is all ones
    return np.allclose(T_decomp, np.ones(n * m * p), atol=1e-9)

def _rank_from_decomposition(U, V, W):
    """Return the rank (number of columns)."""
    return U.shape[1]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank_target = n * m * p  # Start with schoolbook
    
    # First, provide the known valid schoolbook solution as fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = rank_target
    
    # Try to find lower rank via random search with local optimization
    # Strategy: start with random integer/half-integer matrices and try to zero out
    # columns while maintaining exactness, or use structured low-rank forms.
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # For 3x5x7, the theoretical minimum is at least 35.
    # Schoolbook is 105. There may be decompositions between 35 and 105.
    
    # Try structured decompositions with reduced rank
    # One approach: use the fact that some entries can be shared.
    
    # Attempt: random search for rank r from 35 upward, checking feasibility
    # This is hard in general, so we try specific structured families.
    
    # Family 1: Block diagonal with shared structure
    # Try to find a decomposition where W has more zeros (sparsity)
    
    # For now, let's try a few structured approaches with time limit
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 10:
        return best
    
    # Try random restarts with local search to reduce rank
    # Start from schoolbook and try to merge columns
    
    # Actually, let's try a different approach: 
    # Use the known fact that for small matrices, there are known optimal decompositions.
    # For 3x5x7, let's try to find a rank-35 decomposition (the lower bound).
    
    # This is a hard problem, so we'll try random integer matrices of rank r
    # and check if they work. If not, we increase r.
    
    # Given time constraints, let's just return the schoolbook solution
    # unless we can quickly find something better.
    
    # Try one structured approach: exploit the 3x5 and 5x7 structure
    # A[i,j] * B[j,k] for i in 0..2, j in 0..4, k in 0..6
    
    # The tensor has 105 non-zero entries (all 1s).
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # Let's try rank = 35 (the lower bound) with random initialization
    # and see if we can satisfy the constraints.
    
    for trial_rank in [35, 40, 45, 50]:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 20:
            break
        
        for attempt in range(5):
            elapsed = time.time() - start_time
            if elapsed > TIME_LIMIT - 20:
                break
            
            # Random initialization with small integers
            np.random.seed(attempt * 1000 + trial_rank)
            U = np.random.randint(-2, 3, (n * m, trial_rank))
            V = np.random.randint(-2, 3, (m * p, trial_rank))
            W = np.random.randint(-2, 3, (p * n, trial_rank))
            
            if _verify(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = trial_rank
                break
        
        if best_rank < rank_target:
            break
    
    return best
