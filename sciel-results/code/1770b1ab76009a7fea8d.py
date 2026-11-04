"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time as _time

def construct():
    n, m, p = DIMENSIONS
    deadline = _time.time() + 140.0  # safety margin before 150s kill
    
    # Build the rank-42 Strassen decomposition
    rank0 = 42
    U0 = [[0] * rank0 for _ in range(n * m)]
    V0 = [[0] * rank0 for _ in range(m * p)]
    W0 = [[0] * rank0 for _ in range(p * n)]
    
    def add_strassen_block(i_start, j_start, k_start, offset):
        col = offset
        U0[(i_start+0)*m + (j_start+0)][col] += 1
        U0[(i_start+1)*m + (j_start+1)][col] += 1
        V0[(j_start+0)*p + (k_start+0)][col] += 1
        V0[(j_start+1)*p + (k_start+1)][col] += 1
        W0[(k_start+0)*n + (i_start+0)][col] += 1
        W0[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 1
        U0[(i_start+1)*m + (j_start+0)][col] += 1
        U0[(i_start+1)*m + (j_start+1)][col] += 1
        V0[(j_start+0)*p + (k_start+0)][col] += 1
        W0[(k_start+0)*n + (i_start+1)][col] += 1
        W0[(k_start+1)*n + (i_start+1)][col] -= 1
        
        col = offset + 2
        U0[(i_start+0)*m + (j_start+0)][col] += 1
        V0[(j_start+0)*p + (k_start+1)][col] += 1
        V0[(j_start+1)*p + (k_start+1)][col] -= 1
        W0[(k_start+1)*n + (i_start+0)][col] += 1
        W0[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 3
        U0[(i_start+1)*m + (j_start+1)][col] += 1
        V0[(j_start+1)*p + (k_start+0)][col] += 1
        V0[(j_start+0)*p + (k_start+0)][col] -= 1
        W0[(k_start+0)*n + (i_start+0)][col] += 1
        W0[(k_start+0)*n + (i_start+1)][col] += 1
        
        col = offset + 4
        U0[(i_start+0)*m + (j_start+0)][col] += 1
        U0[(i_start+0)*m + (j_start+1)][col] += 1
        V0[(j_start+1)*p + (k_start+1)][col] += 1
        W0[(k_start+0)*n + (i_start+0)][col] -= 1
        W0[(k_start+1)*n + (i_start+0)][col] += 1
        
        col = offset + 5
        U0[(i_start+1)*m + (j_start+0)][col] += 1
        U0[(i_start+0)*m + (j_start+0)][col] -= 1
        V0[(j_start+0)*p + (k_start+0)][col] += 1
        V0[(j_start+0)*p + (k_start+1)][col] += 1
        W0[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 6
        U0[(i_start+0)*m + (j_start+1)][col] += 1
        U0[(i_start+1)*m + (j_start+1)][col] -= 1
        V0[(j_start+1)*p + (k_start+0)][col] += 1
        V0[(j_start+1)*p + (k_start+1)][col] += 1
        W0[(k_start+0)*n + (i_start+0)][col] += 1

    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 0, 2, 7)
    add_strassen_block(0, 0, 4, 14)
    add_strassen_block(0, 2, 0, 21)
    add_strassen_block(0, 2, 2, 28)
    add_strassen_block(0, 2, 4, 35)
    
    U_arr = np.array(U0, dtype=np.float64)
    V_arr = np.array(V0, dtype=np.float64)
    W_arr = np.array(W0, dtype=np.float64)
    
    def verify(U_v, V_v, W_v, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_v[i*m+j], V_v[j*p+k] * W_v[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify(U_arr, V_arr, W_arr):
        return {"U": U0, "V": V0, "W": W0}
    
    # Try to find a lower-rank decomposition using the structure of the problem.
    # The tensor is T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,...,5}
    
    # Strategy: Use a parameterized family of decompositions and optimize.
    # For rank R, we have R terms. Each term r has:
    #   U[:,r] in R^{n*m} = R^8
    #   V[:,r] in R^{m*p} = R^{24}  
    #   W[:,r] in R^{p*n} = R^{12}
    # Constraint: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # This is a system of 48 nonlinear equations in 40*R unknowns (for R terms).
    # For R=36: 40*36=1440 unknowns, 48 equations. Very underdetermined.
    
    # Better approach: Fix two of the three matrices and solve for the third.
    # Or use the fact that the tensor is separable in a specific way.
    
    # Key insight: T[i,j,k] = 1 = (sum over something)
    # We can write 1 = a_i * b_j * c_k for appropriate choices, but that's rank 1 per (i,j,k)
    # which doesn't help directly.
    
    # Alternative: Use the middle index j as the "bottleneck".
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For fixed j, this is a 2x6 matrix that must equal the all-ones 2x6 matrix.
    
    # Let's try: for each j, the contribution from term r is:
    #   (U[i,j,r] for i=0,1) outer (V[j,k,r]*W[k,i,r] for k=0..5, i=0,1)
    # This is complicated because W depends on both k and i.
    
    # Simpler: Let's try to construct a rank-36 decomposition directly.
    # Use the fact that n=2 is small.
    
    # For n=2, W[k*n+i] = W[2k+i], so W has 12 entries indexed by (k,i).
    
    # Let's try a construction where we use "sum" and "difference" terms
    # across the n=2 dimension to reduce rank.
    
    # Try to reduce rank by 1 or 2 using a direct construction.
    # Idea: The 6 Strassen blocks each use 7 terms = 42 total.
    # If we can find a way to share one term between two blocks, we get rank 41.
    
    # Let's try: can we merge the M1 terms of adjacent k-blocks for the same j-block?
    # Block (j=0:2, k=0:2) M1: U[0,0]+U[1,1], V[0,0]+V[1,1], W[0,0]+W[1,1]
    # Block (j=0:2, k=2:4) M1: U[0,0]+U[1,1], V[0,2]+V[1,3], W[2,0]+W[3,1]
    
    # These use the same U entries but different V and W entries, so they can't be merged directly.
    
    # Let's try a completely different construction for rank 40.
    # Use 5 Strassen blocks (35 terms) + 5 additional terms for the remaining structure.
    
    # Actually, the cleanest approach: just return rank 42.
    # The search for lower rank requires more sophisticated algebraic techniques
    # that may not be implementable in this context.
    
    return {"U": U0, "V": V0, "W": W0}
# EVOLVE-BLOCK-END
