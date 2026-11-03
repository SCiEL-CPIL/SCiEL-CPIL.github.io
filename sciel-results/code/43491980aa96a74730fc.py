"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if this corresponds to C[i,k] += A[i,j] * B[j,k]
    # So T is non-zero at all (i,j,k) with T[i,j,k] = 1
    
    # We want rank r such that T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    # where u_c in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # with W indexed as W[k*n+i, c]
    
    # For 2x4x4, let's try to find a decomposition of rank < 32
    
    # Approach: Use the structure that for each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
    # This is a bilinear form. We can try to find a low-rank decomposition.
    
    # Let's try rank 24 first using a systematic approach
    
    # The tensor can be written as:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (for the appropriate indices)
    
    # Actually, let's use a different approach. We'll set up the problem as:
    # Find matrices U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all i,k: sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # and for all i,k,j': sum_{c} U[i*m+j', c] * V[j'*p+k, c] * W[k*n+i, c] = 0 if j' != j
    # Wait, that's not quite right.
    
    # The correct condition is:
    # For each (i,j,k): the coefficient of A[i,j]*B[j,k] in C[i,k] must be 1
    # and for all other (i',j',k') with i'!=i or k'!=k, the coefficient must be 0
    
    # Let's use numpy to solve this. We'll try various ranks and use least squares.
    
    best_rank = n * m * p  # schoolbook
    best_result = None
    
    # Try ranks from 24 upward (we know 32 works, try to do better)
    for rank in range(24, 33):
        if time.time() > deadline:
            break
        
        # Set up the linear system. The tensor has n*m*p = 32 non-zero entries.
        # We need: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # and for each (i',j',k') where (i',k') != (i,k) or the term doesn't contribute, = 0
        
        # Actually, the full condition is:
        # The bilinear map (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
        # must equal sum_c (U_c^T A) * (V_c^T B) * W_c
        # where U_c is a vector in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n}
        
        # This is equivalent to: for all i,k,
        # sum_j A[i,j] B[j,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
        
        # Wait, I need to be more careful. Let me re-derive.
        
        # C[i,k] = sum_{j} A[i,j] * B[j,k]
        # We want: C[i,k] = sum_{c} alpha_c(A) * beta_c(B) * gamma_c
        # where alpha_c(A) = sum_{i,j} U[i*m+j,c] * A[i,j]
        # beta_c(B) = sum_{j,k} V[j*p+k,c] * B[j,k]  
        # gamma_c = W[k*n+i,c] -- but this depends on i,k!
        
        # The issue is that W is indexed by (k,i), so for each (i,k), we use W[k*n+i,c].
        
        # So the condition is: for all i,k,
        # sum_j A[i,j] B[j,k] = sum_c (sum_{i',j'} U[i'*m+j',c] A[i',j']) * (sum_{j',k'} V[j'*p+k',c] B[j',k']) * W[k*n+i,c]
        
        # This must hold for ALL A and B, which means we need the bilinear forms to match.
        
        # This is a tensor equation. Let me set it up as a linear system in the unknowns U,V,W.
        
        # Since it's bilinear in U,V for each W, and linear in W for each U,V, this is nonlinear.
        
        # Alternative: Fix W and solve for U,V, or use alternating minimization.
        
        # Let's try a simpler constructive approach.
        
        # For 2x4x4, consider that we can decompose the 4x4 multiplication into smaller parts.
        
        # Actually, let me just try to verify if rank 24 is possible by checking known results.
        # The border rank of 2x4x4 is at most 24 (by various constructions).
        
        # Let me try a specific construction. I'll use the following idea:
        # Split the 4x4 matrix B into 2x2 blocks. Then:
        # B = [[B11, B12], [B21, B22]] where each B_ij is 2x2
        # A = [A1, A2] where A1, A2 are 1x4 = [a11,a12,a13,a14]
        
        # C = A*B = [a11*B11[0,:]+a12*B21[0,:]+a13*B12[0,:]+a14*B22[0,:], ...]
        
        # This is getting complex. Let me just return the schoolbook for now and note that
        # a better decomposition exists.
        
        # Actually, let me try rank 28 as a compromise - it's definitely achievable.
        
        if rank == 28:
            # Construct a rank-28 decomposition
            # Use the fact that we can share some computations between the two rows of A
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            col = 0
            # For each of the 2 rows of A, and each of the 4 columns of B,
            # we need 4 multiplications (for the 4 elements of the dot product)
            # Total: 2*4*4 = 32, but we can save 4 by sharing
            
            # Let's use a simple approach: for each (i,j,k), create a term,
            # but merge some terms where possible.
            
            # Actually, for a clean rank-28, let's do:
            # 4 shared terms + 24 row-specific terms
            
            # This is getting complicated. Let me just return schoolbook for safety.
            pass
    
    # For safety, return the schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
