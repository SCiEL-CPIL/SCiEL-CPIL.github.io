"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 48
    
    # Try to find a lower rank decomposition
    # The multiplication tensor T has entries T[i,j,k,l] where:
    # i in [0,n), j in [0,m), k in [0,p), l in [0,m)
    # T[i,j,k,l] = 1 if j==l (the shared index)
    # Actually, let me set up the tensor properly.
    
    # For matrix multiplication C = A * B where A is n×m, B is m×p, C is n×p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T in the space (n*m) x (m*p) x (n*p) has:
    # T[(i,j), (j',k), (i',k')] = 1 if i==i', j==j', k==k'
    # Wait, that's not right. Let me think again.
    
    # The bilinear map: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # This is a 3-linear form. The tensor T in R^{n*m} x R^{m*p} x R^{n*p}
    # T[a, b, c] = sum over all i,j,k: delta[i,a_row] * delta[j,a_col] * delta[j,b_row] * delta[k,b_col] * delta[i,c_row] * delta[k,c_col]
    # More simply: T is such that for basis vectors e_a in first space, e_b in second, e_c in third:
    # T(e_a, e_b, e_c) = 1 if a corresponds to (i,j), b to (j',k), c to (i',k') and i==i', j==j', k==k'
    
    # Let me index:
    # First space (A): index = i*m + j, for i in [0,n), j in [0,m)
    # Second space (B): index = j*p + k, for j in [0,m), k in [0,p)  
    # Third space (C): index = i*p + k, for i in [0,n), k in [0,p)
    
    # The tensor T satisfies: for all A, B, C:
    # sum T[(i,j),(j',k),(i',k')] * A[i,j] * B[j',k] * C[i',k'] = sum_{i,k} C[i,k] * (sum_j A[i,j] B[j,k])
    # So T[(i,j),(j',k),(i',k')] = 1 if i==i' and j==j' and k==k', else 0
    
    # The rank-1 decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r is in R^{n*m}, V_r in R^{m*p}, W_r in R^{n*p}
    # For schoolbook: for each (i,j,k), U has 1 at (i*m+j), V has 1 at (j*p+k), W has 1 at (i*p+k)
    
    # To find lower rank, we need to solve:
    # T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # This is equivalent to: for all (a,b), sum_r u_r[a] * v_r[b] = T_a_b (flattened)
    # But that's not quite right either. The tensor is 3-way.
    
    # Let me use the standard approach: flatten T to a matrix of size (n*m*(n*p)) x (m*p) or similar.
    # T as a 3-tensor can be matricized. Let me flatten the first and third indices.
    
    # Actually, for finding low-rank decompositions, I'll use the following approach:
    # Assume a decomposition of rank R. The equations are:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[i*p+k, r] = 1
    # And for all other combinations, the sum should be 0.
    
    # This is a system of polynomial equations. Hard to solve directly.
    
    # Alternative: use known results. For 2×3×8, the best known rank might be 44 or so.
    # Let me try a constructive approach with rank 44.
    
    # Actually, let me just return the schoolbook decomposition for now, as finding 
    # a lower rank decomposition reliably within the time limit is challenging.
    # But let me try one simple improvement first.
    
    # Simple improvement: for the 3×8 multiplication part, we know that 
    # the tensor rank of 1×3×8 is 8, and we have 2 such multiplications.
    # If we could share some factors, we might save a few terms.
    
    # Let me try: for each (i,k), C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # This requires 3 terms per (i,k) pair, so 2*8*3 = 48 terms total.
    
    # Can we do better? Note that B[0,k], B[1,k], B[2,k] are the same for both i=0 and i=1.
    # So we can write:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # If we define for each k:
    # X_k = A[0,0]*B[0,k] + A[1,0]*B[0,k] = (A[0,0]+A[1,0])*B[0,k]
    # But that doesn't directly help.
    
    # Let me just return schoolbook for safety.
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
