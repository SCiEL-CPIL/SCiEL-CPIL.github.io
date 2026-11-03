"""Matrix multiplication 2x4 by 4x8: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = T[i,j,k]
    
    # Start with schoolbook
    rank_schoolbook = n * m * p  # 64
    U = np.zeros((n*m, rank_schoolbook))
    V = np.zeros((m*p, rank_schoolbook))
    W = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook
    
    # Try to reduce rank via rank-1 updates / row-column operations
    # Key insight: if we can find linear dependencies among the rank-1 components,
    # we can eliminate some.
    
    # Approach: Try to find a lower rank by solving for coefficients
    # that express some components as combinations of others.
    
    # For matrix multiplication, the known lower bounds are:
    # For 2x4 * 4x8: lower bound is at least n*m + m*p + n*p = 8 + 32 + 16 = 56
    # (this is a weak bound)
    # Better: the rank must be at least max(n*m, m*p, n*p) = 32
    # Actually for tensor rank of matmul, known results suggest it's around 50-60 for these dims.
    
    # Let's try a systematic approach: random linear combinations
    # Try to express the tensor as sum of fewer rank-1 terms.
    
    # Strategy: Use the fact that for matmul, we can sometimes merge
    # components by finding non-trivial linear relations.
    
    # Try rank from 56 down to 48
    for target_rank in range(56, 47, -1):
        if time.time() > deadline - 10:
            break
        
        # Try random projection approach
        # Generate random U, V, W of rank target_rank and check if they work
        # This is unlikely to work randomly, so use structured approach.
        
        # Structured approach: try to find if schoolbook can be compressed
        # by identifying linearly dependent rows/columns.
        
        # For 2x4*4x8, let's try a specific structured decomposition.
        # The matrix multiplication tensor for (n,m,p) has rank at most n*m*p.
        # For small cases, we can try to find exact lower rank.
        
        # Try: can we use rank = n*p + m*p + n*m - something?
        # Actually, let's just try to verify if rank 56 works with a structured approach.
        
        # Construct a candidate: try to merge some of the 64 components.
        # If two components (i1,j1,k1) and (i2,j2,k2) can be merged when
        # they share indices, we might reduce rank.
        
        # For our case, let's try a direct search for rank reduction.
        # We'll try to find if there's a linear dependency among the 64 basis vectors.
        
        # Represent each rank-1 component as a vector in the space of (i,j,k) tuples
        # Component c corresponds to (i_c, j_c, k_c) where it contributes 1 to T[i_c,j_c,k_c]
        # Two components can be "merged" if their contributions can be combined.
        
        # Actually, the simplest approach: just try smaller ranks with random initialization
        # and check validity.
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random structured initialization
            np.random.seed(attempt * 1000 + target_rank)
            
            U_try = np.random.randn(n*m, target_rank)
            V_try = np.random.randn(m*p, target_rank)
            W_try = np.random.randn(p*n, target_rank)
            
            # Check: does sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k?
            # This is a 3-way tensor product. Check a few entries.
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_try[i*m+j], V_try[j*p+k] * W_try[k*n+i])
                        # Wait, that's not right. The tensor decomposition is:
                        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        # So for each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        val = 0
                        for r in range(target_rank):
                            val += U_try[i*m+j, r] * V_try[j*p+k, r] * W_try[k*n+i, r]
                        if abs(val - 1) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
                break
    
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
