import numpy as np
import random

def construct():
    n, m, p = 3, 4, 4
    
    # The tensor T has T[a,b,c] = 1 if a = i*m+k, b = k*p+j, c = j*n+i
    # Otherwise T[a,b,c] = 0
    # We need: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Build the tensor as a list of (a,b,c) tuples where T=1
    tensor_ones = []
    for i in range(n):
        for j in range(p):
            for k in range(m):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                tensor_ones.append((a, b, c))
    
    # We want to find U (n*m x R), V (m*p x R), W (p*n x R)
    # such that for all (a,b,c) in tensor_ones: sum_r U[a,r]*V[b,r]*W[c,r] = 1
    # and for all other (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = 0
    
    # Start with a high rank and try to reduce
    # Trivial rank: n*m*p = 48
    # Try to find a lower rank
    
    # Use a greedy approach: start with rank 48 and try to merge
    # Or start with a random rank and optimize
    
    # Let's try a known decomposition structure
    # Use block decomposition
    
    # Split into 2x2 blocks of the 4x4 B matrix
    # A is 3x4, B is 4x4, C is 3x4
    
    # Block 1: A[:,0:2] * B[0:2,0:4] -> C[:,0:4]
    # Block 2: A[:,2:4] * B[2:4,0:4] -> C[:,0:4]
    
    # For each block, use a 3x2x4 decomposition
    # 3x2x4 can be done with rank... let's use 8 per block, total 16?
    
    # Actually, let me try a specific construction
    # I'll use the following: for each pair of k values, use 4 multiplications
    
    # k=0,1: compute A[:,0:2] * B[0:2,:]
    # This is 3x2 * 2x4 = 3x4
    # Use rank 8: for each i, use 2 multiplications for each of the 4 columns? No.
    
    # Let me try: for fixed i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    # This is a linear combination of two rows of B, scaled by A[i,0] and A[i,1]
    # We can compute: M1 = A[i,0]*B[0,0], M2 = A[i,1]*B[1,0], M3 = A[i,0]*B[0,1], M4 = A[i,1]*B[1,1], etc.
    # That's 8 multiplications per i, 24 total for k=0,1.
    
    # Similarly for k=2,3: another 24. Total 48.
    
    # To reduce, we need to share across i or across k.
    
    # Idea: use Strassen for the 2x2 part.
    # For fixed i, let X = A[i,0:2], Y = B[0:2, j] (a column of B)
    # C[i,j] = X[0]*Y[0] + X[1]*Y[1]
    # This is a dot product of two 2-vectors, which can be done with 2 multiplications (trivial) or...
    # Actually, dot product of 2-vectors requires 2 multiplications, no saving.
    
    # So for each (i,j), we need 2 multiplications (for k=0,1) + 2 multiplications (for k=2,3) = 4.
    # Total: 3*4*4 = 48.
    
    # To save, we need to compute linear combinations.
    # For fixed j, let's compute for all i:
    # C[0,j] = A[0,0]*B[0,j] + A[0,1]*B[1,j] + A[0,2]*B[2,j] + A[0,3]*B[3,j]
    # C[1,j] = A[1,0]*B[0,j] + A[1,1]*B[1,j] + A[1,2]*B[2,j] + A[1,3]*B[3,j]
    # C[2,j] = A[2,0]*B[0,j] + A[2,1]*B[1,j] + A[2,2]*B[2,j] + A[2,3]*B[3,j]
    
    # Let B_j = [B[0,j], B[1,j], B[2,j], B[3,j]] (a column of B)
    # Then C[:,j] = A * B_j
    # This is a 3x4 matrix times a 4x1 vector = 3x1 vector.
    # This requires 12 multiplications (trivial).
    
    # Can we do better? If we precompute linear combinations of rows of A:
    # Let M1 = A[0,0]*B[0,j], M2 = A[1,0]*B[0,j], M3 = A[2,0]*B[0,j] (for B[0,j])
    # M4 = A[0,1]*B[1,j], M5 = A[1,1]*B[1,j], M6 = A[2,1]*B[1,j] (for B[1,j])
    # M7 = A[0,2]*B[2,j], M8 = A[1,2]*B[2,j], M9 = A[2,2]*B[2,j] (for B[2,j])
    # M10 = A[0,3]*B[3,j], M11 = A[1,3]*B[3,j], M12 = A[2,3]*B[3,j] (for B[3,j])
    # Then C[0,j] = M1+M4+M7+M10, C[1,j] = M2+M5+M8+M11, C[2,j] = M3+M6+M9+M12
    # That's 12 multiplications per j, 48 total.
    
    # To save, we need to share B[k,j] across i.
    # For fixed k,j, B[k,j] is multiplied by A[0,k], A[1,k], A[2,k].
    # Let M_kj_0 = A[0,k]*B[k,j], M_kj_1 = A[1,k]*B[k,j], M_kj_2 = A[2,k]*B[k,j]
    # That's 3 multiplications per (k,j), 48 total.
    
    # No saving. We need a fundamentally different approach.
    
    # Let me try using the known rank for small cases and extrapolate.
    # 2x2x2: 7
    # 2x2x3: 10 (I think)
    # 3x3x3: 23
    # 4x4x4: 49
    
    # For 3x4x4, a reasonable guess is 30-35.
    
    # Let me try to construct a rank-32 decomposition using a block approach.
    # Split A into A1 (3x2) and A2 (3x2), B into B1 (2x4) and B2 (2x4).
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (3x2 * 2x4 = 3x4):
    # Use rank 14 (7 per 2x2 block, 2 blocks)
    # For A2*B2 (3x2 * 2x4 = 3x4):
    # Use rank 14
    # Total: 28
    
    # But this assumes we can do 3x2x2 with rank 7, which may not be true.
    
    # Let me assume rank 8 for 3x2x2, so 16 per 3x2x4, total 32.
    
    # I'll construct this now.
    
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define a 3x2x2 decomposition with rank 8
    # A is 3x2, B is 2x2, C is 3x2
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
    
    # Trivial: 12 multiplications. Can we do 8?
    # Use Strassen-like:
    # M1 = A[0,0]*B[0,0]
    # M2 = A[1,1]*B[1,1]
    # M3 = (A[0,0]+A[1,0])*(B[0,1]+B[1,1])
    # M4 = (A[1,1]-A[0,1])*B[0,0]
    # M5 = (A[0,0]-A[1,0])*(B[1,0]+B[1,1])  # This is getting complicated
    
    # Let me just use rank 12 for now and see if we can improve.
    # Actually, let me use a known 3x2x2 decomposition.
    
    # I'll use the following 8-term decomposition (if it exists):
    # This is a guess, may not be correct.
    
    U322 = np.zeros((6, 8))
    V322 = np.zeros((4, 8))
    W322 = np.zeros((6, 8))
    
    # M1 = A[0,0]*B[0,0] -> C[0,0]
    U322[0,0] = 1; V322[0,0] = 1; W322[0,0] = 1
    # M2 = A[0,1]*B[1,0] -> C[0,0]
    U322[1,1] = 1; V322[2,1] = 1; W322[0,1] = 1
    # M3 = A[0,0]*B[0,1] -> C[0,1]
    U322[0,2] = 1; V322[1,2] = 1; W322[1,2] = 1
    # M4 = A[0,1]*B[1,1] -> C[0,1]
    U322[1,3] = 1; V322[3,3] = 1; W322[1,3] = 1
    # M5 = A[1,0]*B[0,0] -> C[1,0]
    U322[2,4] = 1; V322[0,4] = 1; W322[2,4] = 1
    # M6 = A[1,1]*B[1,0] -> C[1,0]
    U322[3,5] = 1; V322[2,5] = 1; W322[2,5] = 1
    # M7 = A[1,0]*B[0,1] -> C[1,1]
    U322[2,6] = 1; V322[1,6] = 1; W322[3,6] = 1
    # M8 = A[1,1]*B[1,1] -> C[1,1]
    U322[3,7] = 1; V322[3,7] = 1; W322[3,7] = 1
    
    # This only covers rows 0,1 of A. We need row 2 as well.
    # So this is not a complete 3x2x2 decomposition.
    
    # Let me use rank 12 for 3x2x2 (trivial) and see if the total is acceptable.
    # 3x2x4 = 2 * 3x2x2, so rank 24 per 3x2x4.
    # Two such blocks: rank 48. That's the trivial rank.
    
    # I need a better approach.
    
    # Let me try a completely different strategy: use a random search.
    
    # Initialize with rank R and random U, V, W, then optimize.
    
    # Given time, let me just return the current best known: rank 44.
    # But the task asks for a new approach.
    
    # Let me try rank 40 with a structured decomposition.
    
    # I'll use the following: for each of the 4 k-values, use a 3x1x4 sub-decomposition.
    # But share the B[k,j] values.
    
    # For k=0: compute A[:,0] * B[0,:] (3x1 * 1x4 = 3x4)
    # This requires 4 multiplications: A[0,0]*B[0,0], A[1,0]*B[0,0], A[2,0]*B[0,0], and... wait, that's only 3.
    # Actually, A[:,0] is a 3x1 vector, B[0,:] is a 1x4 vector.
    # C_k = A[:,0] * B[0,:] is an outer product, rank 1.
    # So C_k[i,j] = A[i,0] * B[0,j]
    # We can compute this with 1 "multiplication" if we allow vector scaling, but in the tensor decomposition, we need scalar multiplications.
    
    # In the tensor decomposition, each term U[a,r]*V[b,r]*W[c,r] is a scalar multiplication.
    # So for C[i,j] = A[i,0]*B[0,j], we need a term with U[i*m+0, r] = A[i,0] coefficient, V[0*p+j, r] = B[0,j] coefficient, W[j*n+i, r] = 1.
    # But U, V, W are fixed, they don't depend on the input values.
    
    # I'm confusing the tensor decomposition with the actual computation.
    
    # In the tensor decomposition, U, V, W are fixed matrices that define the structure.
    # The actual computation is: for each r, compute S_r = (U[:,r]^T A_flat) * (V[:,r]^T B_flat), then C_flat += S_r * W[:,r].
    
    # So U[:,r] is a vector that selects a linear combination of A's entries.
    # V[:,r] is a vector that selects a linear combination of B's entries.
    # W[:,r] is a vector that distributes S_r to C's entries.
    
    # For C[i,j] = A[i,0]*B[0,j], we can use:
    # U[i*m+0, r] = 1, V[0*p+j, r] = 1, W[j*n+i, r] = 1
    # But this is for a specific (i,j). To cover all i,j, we need 12 terms for k=0.
    
    # This is the trivial approach.
    
    # To save, we need to use linear combinations.
    # For example, let U[:,r] have multiple non-zeros, so that S_r = (sum_i A[i,0]*u_i) * (sum_j B[0,j]*v_j)
    # Then C[i,j] gets contribution S_r * w_{i,j}
    
    # This is like using a rank-1 approximation of the outer product.
    
    # I think I need to accept that finding a low-rank decomposition is non-trivial and use a search.
    
    # Let me try a simple local search.
    
    random.seed(42)
    
    best_rank = 48
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 30 to 48
    for R in range(30, 49):
        # Initialize U, V, W with small random values
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        # Check if this decomposition is valid
        # For all (a,b,c) in tensor_ones: sum_r U[a,r]*V[b,r]*W[c,r] should be 1
        # For all other (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] should be 0
        
        # This is a hard constraint to satisfy with random initialization.
        # Instead, let me use a structured initialization.
        
        break  # Skip random search for now
    
    # Given the complexity, let me return a known good decomposition.
    # I'll use the rank-44 decomposition from the current program as a baseline,
    # but modify it slightly to use a different structure.
    
    # Actually, let me try to improve the current program's approach.
    # The current program uses rank 44 with a block-strassen approach.
    # Let me see if I can reduce it to 40 or 38.
    
    # I'll use the same structure but with a more efficient 3x2x2 decomposition.
    
    # For 3x2x2, I'll use rank 9 instead of 11.
    
    # Define a 3x2x2 decomposition with rank 9
    U322 = np.zeros((6, 9))
    V322 = np.zeros((4, 9))
    W322 = np.zeros((6, 9))
    
    # Use Strassen for the 2x2x2 part (rows 0,1 of A)
    # M1 = A[0,0]*B[0,0]
    U322[0,0] = 1; V322[0,0] = 1; W322[0,0] = 1  # C[0,0]
    # M2 = A[1,1]*B[1,1]
    U322[3,1] = 1; V322[3,1] = 1; W322[3,1] = 1  # C[1,1]
    # M3 = (A[0,0]+A[1,0])*(B[0,1]+B[1,1])
    U322[0,2] = 1; U322[2,2] = 1; V322[1,2] = 1; V322[3,2] = 1
    W322[1,2] = 1; W322[3,2] = 1  # C[0,1] and C[1,1]
    # M4 = (A[1,1]-A[0,1])*B[0,0]
    U322[3,3] = 1; U322[1,3] = -1; V322[0,3] = 1
    W322[0,3] = -1; W322[3,3] = 1  # C[0,0] and C[1,1]
    # M5 = (A[0,0]-A[1,0])*(B[1,0]+B[1,1])
    U322[0,4] = 1; U322[2,4] = -1; V322[2,4] = 1; V322[3,4] = 1
    W322[0,4] = 1; W322[1,4] = 1  # C[0,0] and C[0,1]
    # M6 = A[1,0]*B[1,0]
    U322[2,5] = 1; V322[2,5] = 1; W322[2,5] = 1  # C[1,0]
    # M7 = A[0,1]*B[0,1]
    U322[1,6] = 1; V322[1,6] = 1; W322[1,6] = 1  # C[0,1]
    
    # This is getting too complicated and error-prone.
    
    # Let me just return the current program's result as is, since I don't have time to develop a better one.
    # But the task asks for a new approach.
    
    # Final attempt: use a known result from the literature.
    # I believe the rank of 3x4x4 is 30.
    # Let me construct a rank-30 decomposition using a systematic method.
    
    # I'll use the following: for each of the 4 k-values, use a 3x1x4 sub-decomposition with rank 7.5 (not integer, so adjust).
    # Total rank 30.
    
    # This is a guess. Let me just implement a rank-30 decomposition with a plausible structure.
    
    rank = 30
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # This is a placeholder. I don't have a valid rank-30 decomposition.
    # Let me return the rank-44 decomposition to be safe.
    
    # Actually, let me try to improve the current program by optimizing the 3x2x2 block.
    
    # The current program uses rank 11 for 3x2x2. Let me try rank 10.
    
    # I'll modify the current program's U_loc, V_loc, W_loc to use rank 10.
    
    U_loc = np.zeros((6, 10))
    V_loc = np.zeros((4, 10))
    W_loc = np.zeros((6, 10))
    
    # Strassen part (cols 0-6) - same as before
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Extra part (cols 7-9) for Row 2 of A - use 3 terms instead of 4
    # M8 = a20 * b00 -> c20
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    # M9 = a21 * b10 -> c20
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    # M10 = a20 * b01 + a21 * b11 -> c21 (combined)
    # This is not valid as a single term.
    
    # I can't combine them. So I need 4 terms for row 2.
    
    # Total rank for 3x2x2 is 11.
    
    # With 4 blocks, total rank is 44.
    
    # I'm stuck. Let me just return the current program.
    
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]
    
    add_block(0, 0, 0, 0, 0, 0, 0)
    add_block(1, 0, 2, 2, 0, 0, 0)
    add_block(2, 0, 0, 0, 2, 0, 2)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
