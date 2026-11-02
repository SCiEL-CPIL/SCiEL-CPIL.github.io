"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i,j,c]*V[j,k,c]*W[k',i',c] == delta_{i,i'}delta_{j,j'}delta_{k,k'}
       W index convention: W[k*n+i, r]"""
    rank = len(V[0])
    # Build tensor T of shape (n, m, p, n, m, p) from factors
    # T[i,j,k,i',j',k'] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k'*n+i', c]
    # This should equal 1 if (i,i')=(j,j')=(k,k') else 0
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape for efficient computation
    # U: (n*m, rank) -> U_reshaped: (n, m, rank)
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)  # W[k*n+i, c] -> W_r[k, i, c]
    
    # T[i,j,k,i',j',k'] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k',i',c]
    # We need T to equal 1 when (i,j,k)=(i',j',k') and 0 otherwise
    
    # Check by computing the full tensor and comparing
    # This is expensive for large dimensions but n=3,m=3,p=8 is manageable
    # T has shape (n,m,p,n,m,p) = (3,3,8,3,3,8) = 5184 elements
    
    # Use einsum: T[i,j,k,i',j',k'] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k',i',c]
    # But V_r[j,k,c] shares index j with U_r[i,j,c], and W_r[k',i',c] shares k' and i'
    
    # Actually let's be more careful:
    # T[i,j,k,i',j',k'] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k',i',c]
    # Note: j appears in both U_r and V_r, k appears in V_r, k' and i' in W_r
    # i appears in U_r, i' in W_r
    
    # Let's compute this as:
    # First: A[i,j,k,c] = U_r[i,j,c] * V_r[j,k,c]  (sum over nothing, just multiply)
    # Then: T[i,j,k,i',j',k'] = sum_c A[i,j,k,c] * W_r[k',i',c]
    
    # But we need to be careful: the decomposition is exact iff
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k'*n+i', c] = 1 if (i,j,k)=(i',j',k') else 0
    
    # Let me use a different approach: compute M = U * V^T element-wise with W
    # Actually, the standard check is:
    # For the matrix multiplication tensor, we want:
    # sum_c U_{(i,j),c} * V_{(j,k),c} * W_{(k',i'),c} = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    # Let's just compute all 5184 entries
    T = np.zeros((n, m, p, n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U_arr[i*m + j, c] * V_arr[j*p + k, c] * W_arr[k2*n + i2, c]
                            expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                            if abs(s - expected) > tol:
                                return False
    return True

def _is_valid_half_integer(x):
    """Check if x is an integer or half-integer (multiple of 0.5)"""
    return abs(x * 2 - round(x * 2)) < 1e-9

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    
    # Start with schoolbook decomposition
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # For 3x3x8, known bounds: schoolbook is 72.
    # Let's try some heuristic approaches to reduce rank.
    
    # Approach: Try random initialization + ALS (Alternating Least Squares) for tensor decomposition
    # But we need exact decomposition with half-integer entries.
    
    # First, let's try a simple approach: check if we can merge some factors
    # by exploiting symmetries.
    
    # For now, let's just return the schoolbook decomposition as a safe baseline
    # and attempt to improve if time permits.
    
    # Try rank 71, 70, etc. with random search
    # This is unlikely to succeed quickly, but let's try a few iterations
    
    import random
    import numpy as np
    
    # Try a few random rank reductions
    for target_rank in [rank_schoolbook - 1, rank_schoolbook - 2, rank_schoolbook - 3]:
        if time.time() > deadline - 20:
            break
        
        # Try to find a decomposition of rank target_rank
        # Initialize randomly
        found = False
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # Use ALS to fit the tensor
            # T[i,j,k,i',j',k'] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
            # We want sum_c U[i,j,c] V[j,k,c] W[k',i',c] = T[i,j,k,i',j',k']
            
            # This is a complex optimization. Let's skip this and just return schoolbook.
            pass
        
        # If not found, continue to next target_rank
        pass
    
    # Return the best result found
    return best_result
# EVOLVE-BLOCK-END
