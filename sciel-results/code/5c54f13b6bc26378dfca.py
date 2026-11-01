"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": rank_schoolbook}
    
    # Try to find a lower-rank decomposition
    # The tensor T has entries T[i,j,k] = 1 for the multiplication tensor
    # We want to find rank-r decomposition: T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    # Try ranks from 53 down, but that's expensive. Instead, try a few candidate ranks.
    # Known: for (2,3,3) rank is 11 < 18. Let's try to see if (2,3,9) has rank < 54.
    
    # Attempt: use a structured decomposition
    # Idea: For each pair (i, j), we need to compute A[i,j] * B[j, :] which contributes to C[i, :]
    # So for fixed i,j, we have a rank-9 term: A[i,j] * (B[j,0], B[j,1], ..., B[j,8])
    # This is already the schoolbook.
    
    # Alternative: can we share across different j for the same i?
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is like a matrix-vector product for each i.
    
    # Let's try a random search for lower rank
    rng = np.random.RandomState(42)
    
    # Try a few candidate ranks below 54
    for target_rank in [53, 52, 50, 48, 45, 40, 36]:
        if time.time() > deadline - 20:
            break
        
        # Try to find a rank-target_rank decomposition
        # Use a simple iterative approach: initialize randomly and try to fit
        # This is a hard problem, so we'll use a heuristic
        
        # Actually, let's try a different approach: use the fact that
        # the tensor might have a low-rank structure due to the small n=2
        
        # For n=2, we can think of this as two "layers":
        # Layer 0: C[0,k] = sum_j A[0,j] * B[j,k]
        # Layer 1: C[1,k] = sum_j A[1,j] * B[j,k]
        
        # The tensor is T[i,j,k] = 1 for all i,j,k (in the appropriate indexing)
        # Wait, no. The tensor for matrix multiplication is:
        # T[A_indices, B_indices, C_indices] = 1 if C_index corresponds to the product
        
        # More precisely, for matrix multiplication M_{n×m} * M_{m×p} = M_{n×p}:
        # The tensor T has T[i,j,k] = 1 where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
        # and the multiplication is: C[k*n + i] = sum_j A[i*m+j] * B[j*p+k]
        
        # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
        
        # This is exactly what I built above.
        
        # Let's try to see if we can find a lower rank by exploiting symmetries.
        # For now, return the schoolbook if we can't find better quickly.
        
        pass
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
