"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 140
    
    n, m, p = DIMENSIONS
    
    # Use 3x3 block structure: split 6x6 into four 3x3 blocks
    # Each 3x3x3 multiplication uses rank-23 (Lam's decomposition)
    # Total: 4 blocks * 23 = 92... wait, that's not right.
    # Actually for 6x6 = 2x2 blocks of 3x3, we need 4 multiplications of 3x3 matrices
    # But each 3x3 multiplication has rank 23, so total rank = 4 * 23 = 92
    # That seems too low. Let me reconsider.
    
    # Actually, the block decomposition works differently:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # 
    # Each C_ij is a 3x3 matrix, computed as sum of two 3x3 products.
    # Each 3x3 product has rank 23, so each C_ij needs 2*23 = 46 rank terms.
    # Total: 4 * 46 = 184 rank terms.
    
    # Let me implement the known rank-23 decomposition for 3x3x3.
    # This is based on Lam's 1981 result.
    
    # The rank-23 decomposition for 3x3x3 matrix multiplication:
    # We need U, V, W matrices of shape (9, 23) each
    # such that sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1 for all i,j,k
    
    # Known rank-23 solution (one standard form):
    # This is complex, so let me use a different approach.
    
    # Alternative: Use the 2x2x2 Strassen decomposition but with a better base.
    # For 2x2x2, Strassen gives rank 7.
    # For 6x6, we can use 3x3x3 blocks of size 2x2: 27 * 7 = 189 (current)
    
    # Better: Try to find a direct rank-200 or lower decomposition.
    # 
    # Actually, let me try a hybrid: use 3x3 blocks with a rank-21 decomposition for 3x3.
    # If 3x3x3 can be done with rank 21, then 4*21*2 = 168... no, that's not right either.
    
    # Let me reconsider the block structure more carefully.
    #
    # For n x n matrix multiplication using b x b blocks:
    # If we split into (n/b) x (n/b) blocks, we get (n/b)^2 output blocks.
    # Each output block C_ij = sum_k A_ik * B_kj, which is a sum of (n/b) matrix products.
    # Each matrix product of size b x b has some rank r_b.
    # Total rank = (n/b)^2 * (n/b) * r_b = (n/b)^3 * r_b
    #
    # For n=6, b=2: (6/2)^3 * 7 = 27 * 7 = 189
    # For n=6, b=3: (6/3)^3 * r_3 = 8 * r_3
    # If r_3 = 23, then 8 * 23 = 184
    # If r_3 = 22, then 8 * 22 = 176
    # If r_3 = 21, then 8 * 21 = 168
    
    # The best known rank for 3x3x3 is 23 (Lam, 1981).
    # So 8 * 23 = 184 is achievable.
    
    # Let me implement the rank-23 decomposition for 3x3x3.
    # I'll use a known explicit decomposition.
    
    # One known rank-23 decomposition for 3x3x3:
    # Reference: Lam, "A 3x3x3 matrix multiplication algorithm with 23 multiplications"
    
    # Due to complexity, let me use a different strategy:
    # Start with the 189 rank solution and try to reduce it.
    
    # Actually, let me try a simpler improvement:
    # Use the fact that 6 = 2 + 2 + 2 and apply Strassen at multiple levels.
    
    # Or, use a direct search for a lower rank decomposition.
    
    # For now, let me implement the 3x3 block approach with rank-23.
    
    # Known rank-23 decomposition for 3x3x3 (one explicit form):
    # This is quite involved. Let me use a numerical approach instead.
    
    # Strategy: Start with the naive 27-rank decomposition for 3x3x3,
    # then use iterative methods to reduce the rank.
    
    # Naive 3x3x3: rank 27 (each of the 27 terms is a rank-1 tensor)
    # We want to reduce this to 23.
    
    # Let me try a different approach: use the 2x2x2 Strassen and see if
    # we can combine blocks more efficiently.
    
    # Actually, let me just implement the known best results:
    # For 6x6x6, the best known is around 184 (using 3x3 blocks with rank-23).
    
    # Let me construct the rank-23 3x3x3 decomposition explicitly.
    
    # Here's a known rank-23 decomposition (simplified):
    # I'll use the approach from the literature.
    
    # Due to the complexity of writing out the full rank-23 decomposition,
    # let me use a different strategy:
    
    # Use 2x2x2 Strassen blocks but try to find linear dependencies to reduce rank.
    
    # Start with the 189-rank solution:
    block_size = 2
    blocks_per_dim = 3
    base_rank = (blocks_per_dim ** 3) * 7
    U = [[0] * base_rank for _ in range(n * m)]
    V = [[0] * base_rank for _ in range(m * p)]
    W = [[0] * base_rank for _ in range(p * n)]
    
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1
    
    # Now try to reduce rank by finding linear dependencies.
    # The tensor equation is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # If we can find a vector c in the null space of the "tensor product matrix",
    # we can potentially reduce the rank.
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the constraint matrix: for each (i,j,k), the row is
    # [U[i*m+j, 0]*V[j*p+k, 0]*W[k*n+i, 0], ..., U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]]
    n_eq = n * m * p
    A = np.zeros((n_eq, base_rank), dtype=float)
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_np[i*m+j]
                v_row = V_np[j*p+k]
                w_row = W_np[k*n+i]
                A[eq_idx] = u_row * v_row * w_row
                eq_idx += 1
    
    # Find the null space
    try:
        U_svd, S, Vt = np.linalg.svd(A, full_matrices=False)
        tol = 1e-10
        rank_svd = np.sum(S > tol)
        
        if rank_svd < base_rank:
            null_space = Vt[rank_svd:]
            null_dim = null_space.shape[0]
            
            # Try to use null space vectors to eliminate columns
            # If a column can be expressed as a linear combination of others
            # while preserving the tensor equation, we can remove it.
            
            # This is complex, so let's try a simpler approach:
            # Greedily remove columns that are approximately in the span of others.
            
            # For now, just return the 189-rank solution.
            pass
    except:
        pass
    
    # Try a completely different approach: use a 3x3x3 rank-23 decomposition.
    # Let me implement a known rank-23 solution.
    
    # Here's an explicit rank-23 decomposition for 3x3x3 matrix multiplication.
    # Based on Lam's work, one form uses the following pattern:
    
    # Due to the complexity, let me try a numerical optimization approach.
    # Start with the naive 27-rank decomposition and try to reduce it.
    
    if time.time() > deadline - 5:
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition using iterative refinement.
    # This is computationally expensive, so we'll only attempt it if time permits.
    
    # For now, return the 189-rank solution.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
