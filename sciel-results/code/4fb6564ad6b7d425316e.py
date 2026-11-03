"""Matrix multiplication 2x3x8 with Strassen-based rank-44 decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # We'll build the rank-44 decomposition.
    # A is 2x3, B is 3x8.
    # Split B into 4 blocks of width 2: B = [B_0 | B_1 | B_2 | B_3]
    # For each block b (0..3):
    #   A_1 = A[:, 0:2]  (2x2)
    #   A_2 = A[:, 2:3]  (2x1)
    #   B_{b,1} = B[0:2, 2b:2b+2]  (2x2)
    #   B_{b,2} = B[2:3, 2b:2b+2]  (1x2)
    #   C[:, 2b:2b+2] = A_1 * B_{b,1} + A_2 * B_{b,2}
    #
    # Strassen on A_1 * B_{b,1} uses 7 terms.
    # Schoolbook on A_2 * B_{b,2} uses 2 terms.
    # Total per block: 9 terms.
    # Total: 4 * 9 = 36? Wait, that's not 44.
    #
    # Let me reconsider. The known result for 2x3x8 is rank 44? 
    # Actually, let me think about this differently.
    #
    # Alternative: Think of it as 2x3 * 3x8.
    # We can write this as a sum of Kronecker-like structures.
    #
    # Actually, let me use a known construction. 
    # The border rank of 2x3x8 is known to be around 44-48.
    # Let me try a direct rank-44 construction.
    #
    # Approach: Use the fact that matmul(2,3,8) can be embedded in matmul(4,4,8) 
    # or use recursive Strassen.
    #
    # Let me try: pad A to 4x4, pad B to 4x4, use Strassen twice?
    # That would give 7^2 = 49 for 4x4x4, but we need 2x3x8.
    #
    # Better approach: 
    # Write the tensor as sum of rank-1 tensors.
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where T[i,j,k] = delta(i + j == k) in appropriate indexing.
    #
    # For 2x3x8: T has indices (i,j,k) with i in {0,1}, j in {0,1,2}, k in {0,...,7}.
    # The "output" C[i,k] = sum_j A[i,j]*B[j,k].
    # So T[i,j,k] = 1 for all valid (i,j,k) in the standard embedding.
    #
    # Actually the tensor for matmul is T[i,j,k] which when contracted gives C[i,k].
    # The rank-1 decomposition is: T = sum over (i,j,k) of e_i ⊗ e_j ⊗ e_k, which is rank 48.
    #
    # To reduce rank, we need linear combinations. 
    # The idea is that some of these basis tensors can be combined.
    #
    # Let me try a Strassen-inspired approach more carefully.
    # Consider multiplying a 2x2 matrix by a 2x2 matrix (rank 7 via Strassen).
    # For 2x3 * 3x8, think of it as:
    #   [A_left (2x2)] [A_right (2x1)] * [[B_top (2x8)] ; [B_bottom (1x8)]]
    #   = A_left * B_top + A_right * B_bottom
    #
    # A_left * B_top is 2x2 * 2x8. Split B_top into 4 blocks of 2x2.
    # Each 2x2 * 2x2 can be done with rank 7 (Strassen).
    # So A_left * B_top uses 4 * 7 = 28 terms.
    #
    # A_right * B_bottom is 2x1 * 1x8. This is just 2 * 8 = 16 scalar products,
    # but as a tensor decomposition, each term A_right[i] * B_bottom[k] contributes
    # to C[i,k]. This is rank 2*8 = 16? No, it's rank 2 since A_right is a column
    # vector and B_bottom is a row vector... wait.
    #
    # A_right * B_bottom: A_right is 2x1, B_bottom is 1x8.
    # C_contribution[i,k] = A_right[i,0] * B_bottom[0,k].
    # This is an outer product, so it's rank 1 as a matrix, but as a tensor 
    # decomposition of the matmul operation, each (i,k) pair needs its own term
    # unless we can combine them.
    #
    # Actually, for the tensor T, the contribution from A_right * B_bottom is:
    # For each i in {0,1} and k in {0,...,7}, we have T[i, 2, k] = 1 (where j=2 is the third row).
    # These are 16 separate rank-1 tensors: e_i ⊗ e_2 ⊗ e_k for i=0,1 and k=0,...,7.
    # Can we combine any of them? 
    # sum_i sum_k alpha[i,k] * (e_i ⊗ e_2 ⊗ e_k) = (sum_i alpha[i,0] e_i + ...) 
    # No, these are all distinct basis elements in the tensor space, so they can't be
    # combined into fewer rank-1 terms. So this part has rank 16.
    #
    # Total: 28 + 16 = 44. That matches!
    #
    # So the construction is:
    # - 28 terms from Strassen on A[:,0:2] * B[0:2, :] (split into 4 blocks of width 2)
    # - 16 terms from A[:,2:3] * B[2:3, :] (schoolbook/outer product)
    #
    # Let me construct this explicitly.
    
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: Strassen for A[:,0:2] * B[0:2, :]
    # A_left is 2x2: indices (i, j) with i in {0,1}, j in {0,1}
    # B_top is 2x8: indices (j, k) with j in {0,1}, k in {0,...,7}
    # Split k into 4 blocks of size 2.
    
    for b in range(4):  # block index 0,1,2,3
        k0 = 2 * b  # first column in this block
        
        # A_left blocks:
        # A_00 = A[0,0], A_01 = A[0,1]
        # A_10 = A[1,0], A_11 = A[1,1]
        
        # B_top block: B[j, k0], B[j, k0+1] for j in {0,1}
        # B_00 = B[0, k0], B_01 = B[0, k0+1]
        # B_10 = B[1, k0], B_11 = B[1, k0+1]
        
        # Strassen 7 products:
        # M1 = (A_00 + A_11) * (B_00 + B_11)
        # M2 = (A_10 + A_11) * B_00
        # M3 = A_00 * (B_10 - B_11)
        # M4 = A_11 * (B_00 - B_01)
        # M5 = (A_00 + A_01) * B_11
        # M6 = (A_10 - A_00) * (B_00 + B_01)
        # M7 = (A_01 - A_11) * (B_10 + B_11)
        
        # M1
        U[0][col] += 1  # A[0,0]
        U[3][col] += 1  # A[1,1]
        V[0][col] += 1  # B[0,k0]
        V[2][col] += 1  # B[1,k0+1]
        # M1 contributes to C[0,0] and C[1,1] in the 2x2 block
        # In global coords: C[0, k0] and C[1, k0+1]
        W[k0 * n + 0][col] += 1    # C[0, k0]
        W[(k0+1) * n + 1][col] += 1  # C[1, k0+1]
        col += 1
        
        # M2
        U[3][col] += 1  # A[1,0]
        U[4][col] += 1  # A[1,1]
        V[0][col] += 1  # B[0,k0]
        # M2 contributes to C[1,0] in the 2x2 block -> C[1, k0]
        W[k0 * n + 1][col] += 1
        col += 1
        
        # M3
        U[0][col] += 1  # A[0,0]
        V[1][col] += 1  # B[1,k0]
        V[2][col] -= 1  # B[1,k0+1]
        # M3 contributes to C[0,1] in the 2x2 block -> C[0, k0+1]
        W[(k0+1) * n + 0][col] += 1
        col += 1
        
        # M4
        U[4][col] += 1  # A[1,1]
        V[0][col] += 1  # B[0,k0]
        V[1][col] -= 1  # B[0,k0+1]
        # M4 contributes to C[0,0] in the 2x2 block -> C[0, k0]
        W[k0 * n + 0][col] += 1
        col += 1
        
        # M5
        U[0][col] += 1  # A[0,0]
        U[1][col] += 1  # A[0,1]
        V[2][col] += 1  # B[1,k0+1]
        # M5 contributes to C[0,1] in the 2x2 block -> C[0, k0+1]
        W[(k0+1) * n + 0][col] += 1
        col += 1
        
        # M6
        U[3][col] += 1  # A[1,0]
        U[0][col] -= 1  # A[0,0]
        V[0][col] += 1  # B[0,k0]
        V[1][col] += 1  # B[0,k0+1]
        # M6 contributes to C[1,1] in the 2x2 block -> C[1, k0+1]
        W[(k0+1) * n + 1][col] += 1
        col += 1
        
        # M7
        U[1][col] += 1  # A[0,1]
        U[4][col] -= 1  # A[1,1]
        V[1][col] += 1  # B[1,k0]
        V[2][col] += 1  # B[1,k0+1]
        # M7 contributes to C[0,0] in the 2x2 block -> C[0, k0]
        W[k0 * n + 0][col] += 1
        col += 1
    
    # Part 2: Schoolbook for A[:,2:3] * B[2:3, :]
    # A_right is 2x1: A[0,2] and A[1,2]
    # B_bottom is 1x8: B[2,0], ..., B[2,7]
    # C[i,k] += A[i,2] * B[2,k] for i in {0,1}, k in {0,...,7}
    
    for i in range(n):  # i = 0, 1
        for k in range(p):  # k = 0, ..., 7
            U[i * m + 2][col] = 1  # A[i, 2]
            V[2 * p + k][col] = 1  # B[2, k]
            W[k * n + i][col] = 1  # C[i, k]
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
