"""Matrix multiplication 3x3x7 - rank-minimal tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    # Output: C[i,k] = sum_j A[i,j] * B[j,k]
    # U: shape (n*m, rank) = (9, rank)
    # V: shape (m*p, rank) = (21, rank)
    # W: shape (p*n, rank) = (21, rank)
    # W index: k*n + i
    
    # Strategy: Use rank-23 decomposition for each 3x3x3 block
    # and rank-9 for the 3x3x1 block.
    # Total: 23 + 23 + 9 = 55
    
    # Actually, let me try to use the known rank-23 algorithm for 3x3x3.
    # The Strassen-based approach gives rank 23 for 3x3x3.
    
    # For 3x3x3, the standard schoolbook is rank 27.
    # Known best: rank 23 (Bürgisser 2011, etc.)
    
    # Let me use a concrete rank-23 decomposition for 3x3x3.
    # I'll use the decomposition from the literature.
    
    # Actually, let me think about this more carefully.
    # The task says "rank-minimal exact tensor decomposition".
    # For 3x3x3, the minimal rank is 23.
    # For 3x3x1, the minimal rank is 9 (it's just a sum of 9 outer products).
    
    # So total rank = 23 + 23 + 9 = 55.
    
    # Let me construct the rank-23 decomposition for 3x3x3.
    # I'll use the algorithm from https://arxiv.org/abs/1011.5944 or similar.
    
    # A simpler approach: use the known 7-term Strassen for 2x2x2,
    # then extend to 3x3x3.
    
    # Let me use a different approach: hardcode a known rank-23 decomposition.
    
    # The 23 terms for 3x3x3 (0-indexed i,j,k in 0..2):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # I'll use the decomposition from the "rank 23" algorithm.
    
    # Let me try a different strategy: use the Kronecker product approach.
    # 3x3x3 can be seen as 2x2x2 (rank 7) + cross terms.
    
    # Actually, let me just use schoolbook for now and see if I can find a better approach.
    # Schoolbook: 27 + 27 + 9 = 63.
    
    # Let me try to use a rank-22 or rank-23 decomposition for 3x3x3.
    
    # I found a reference: the minimal rank for 3x3x3 is 23.
    # Let me hardcode the 23 terms.
    
    # From the literature, here's a rank-23 decomposition for 3x3x3:
    # The terms are (u, v, w) where:
    #   u[i,j] = coefficient for A[i,j]
    #   v[j,k] = coefficient for B[j,k]  
    #   w[i,k] = coefficient for C[i,k]
    #   C[i,k] = sum_t u_t[i,j] * v_t[j,k] * w_t[i,k] (summed over j)
    
    # Actually, the tensor decomposition is:
    # T = sum_t U_t ⊗ V_t ⊗ W_t
    # where U_t is a 3x3 matrix, V_t is a 3x3 matrix, W_t is a 3x3 matrix
    # and T[i,j,k] = sum_t U_t[i,j] * V_t[j,k] * W_t[k,i]  (note the index convention)
    
    # Wait, let me re-read the task.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank) = (21, rank)
    # W[k*n + i, t] is the coefficient for the (k,i) entry of C in term t.
    
    # The tensor is: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # This should equal sum_j A[i,j] * B[j,k] for the matrix multiplication.
    
    # So: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = sum_j A[i,j] * B[j,k]
    
    # This means: for each (i,k), we need:
    # sum_j [sum_t U[i*m+j, t] * W[k*n+i, t]] * V[j*p+k, t] ... 
    
    # Actually, the standard tensor decomposition for matrix multiplication M_{n,m,p} is:
    # (A ⊗ B ⊗ C) where the tensor T has entries T[i,j,k] = delta_{i+k, j} (or similar)
    # No wait, for matrix multiplication: C = A * B
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor has T[i,j,k] = 1 if the indices match appropriately.
    
    # Let me think again. The tensor for matrix multiplication M_{n,m,p} is:
    # T in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}
    # T[i*m+j, j*p+k, k*n+i] = 1 (this is the "multiplication tensor")
    # No, that's not right either.
    
    # The standard formulation: the multiplication tensor M_{n,m,p} is defined as:
    # M = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # where e_{i,j} is the basis vector for the (i,j) entry of A,
    # e_{j,k} is the basis vector for the (j,k) entry of B,
    # e_{k,i} is the basis vector for the (k,i) entry of C.
    
    # So the tensor has rank at most n*m*p (schoolbook).
    # A rank-r decomposition is: M = sum_{t=1}^r u_t ⊗ v_t ⊗ w_t
    # where u_t in R^{n*m}, v_t in R^{m*p}, w_t in R^{p*n}
    
    # And the condition is: for all A, B:
    # C = A*B = sum_t (u_t^T * A_flat) * (v_t^T * B_flat) * w_t
    # where A_flat is the vectorized A, etc.
    
    # More precisely: C_flat = sum_t (U[:,t]^T * A_flat) * (V[:,t]^T * B_flat) * W[:,t]
    # where U, V, W are the factor matrices.
    
    # So: C[i,k] = sum_t (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k]) * W[k*n+i, t]
    
    # This should equal sum_j A[i,j] * B[j,k].
    
    # So we need: for all i, k:
    # sum_t (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k]) * W[k*n+i, t] = sum_j A[i,j] * B[j,k]
    
    # This is a polynomial identity in A and B.
    
    # For schoolbook: each term t corresponds to a unique (i,j,k) triple:
    # U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1
    # Then: sum_t ... = sum_{i',j',k'} A[i',j'] * B[j',k'] * delta_{i',i} * delta_{k',k}
    # = sum_{j'} A[i,j'] * B[j',k] ✓
    
    # OK so the current program is correct for schoolbook.
    
    # Now, for rank-23 decomposition of 3x3x3, I need to find 23 terms
    # such that the above identity holds.
    
    # Let me use a known rank-23 algorithm. I'll use the one from:
    # "A rank 23 algorithm for 3x3 matrix multiplication"
    
    # After research, I'll use the following approach:
    # Split 3x3 into 2x2 + 2x1 + 1x2 + 1x1 blocks.
    # Use Strassen (rank 7) for 2x2x2.
    # Use 6 terms for 2x1x1, 6 terms for 1x2x1, 3 terms for 1x1x1.
    # Total: 7 + 6 + 6 + 3 = 22? No, that's not quite right.
    
    # Actually, the cross terms complicate things.
    
    # Let me just use schoolbook (rank 63) for now, since I'm not confident
    # in implementing a correct rank-23 decomposition.
    
    rank = 63
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # First 3x3 block: i in 0..2, j in 0..2, k in 0..2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Second 3x3 block: i in 0..2, j in 0..2, k in 3..5
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Third 3x1 block: i in 0..2, j in 0..2, k in 6
    for i in range(n):
        for j in range(m):
            k = 6
            U[i*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
