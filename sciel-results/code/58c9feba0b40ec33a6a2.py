"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Winograd F(4) for the 4x4 core, extended to 4x6 output.
    # Strategy: Use Winograd F(4) on the 4x4 part of B (columns 0-3),
    # and handle columns 4-5 separately with a smaller Winograd or direct method.
    # 
    # Winograd F(4) for C = A * B where A is 4x4, B is 4x4:
    # Precompute:
    #   M = A * A (but with specific linear combinations)
    # Actually, Winograd F(4) uses:
    #   Let a = A (4x4), b = B (4x4)
    #   Precompute 16 linear combinations of a:
    #     m0 = a00 + a01 + a02 + a03
    #     m1 = a00 - a01 + a02 - a03
    #     m2 = a00 - a01 - a02 + a03
    #     m3 = a00 + a01 - a02 - a03
    #     ... (similar for other rows)
    #   Similarly for b:
    #     n0 = b00 + b10 + b20 + b30
    #     n1 = b00 - b10 + b20 - b30
    #     ...
    #   Then C entries are products of m and n combinations.
    #
    # For 4x4x4, Winograd F(4) achieves rank 16 (vs 64 schoolbook).
    # For 4x4x6, we can split B into B1 (4x4) and B2 (4x2).
    # 
    # Let's use a known Winograd F(4) decomposition for the 4x4 part,
    # and a simple decomposition for the 4x2 part.
    #
    # Winograd F(4) for 4x4x4:
    # The 4x4 matrix multiplication tensor can be decomposed with rank 16
    # using the Winograd scheme. The key idea is:
    #   C = A * B
    #   Define 16 intermediate values from A (linear combinations of rows)
    #   Define 16 intermediate values from B (linear combinations of columns)
    #   C entries are sums of products of these intermediates.
    #
    # Let me construct this explicitly.
    #
    # For Winograd F(4):
    # A is 4x4: A[i,j] for i,j in 0..3
    # B is 4x4: B[j,k] for j,k in 0..3
    # C = A*B: C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Winograd precomputes:
    # From A: 
    #   For each row i, compute 4 values:
    #     a_i0 = A[i,0] + A[i,1] + A[i,2] + A[i,3]
    #     a_i1 = A[i,0] - A[i,1] + A[i,2] - A[i,3]
    #     a_i2 = A[i,0] - A[i,1] - A[i,2] + A[i,3]
    #     a_i3 = A[i,0] + A[i,1] - A[i,2] - A[i,3]
    #   Total: 16 values from A
    #
    # From B:
    #   For each column k, compute 4 values:
    #     b_0k = B[0,k] + B[1,k] + B[2,k] + B[3,k]
    #     b_1k = B[0,k] - B[1,k] + B[2,k] - B[3,k]
    #     b_2k = B[0,k] - B[1,k] - B[2,k] + B[3,k]
    #     b_3k = B[0,k] + B[1,k] - B[2,k] - B[3,k]
    #   Total: 16 values from B
    #
    # Then C[i,k] = (1/4) * sum_{r,s} coeff[r,s] * a_ir * b_sk
    # where coeff is a 4x4 matrix derived from the Winograd scheme.
    #
    # Actually, the Winograd F(4) scheme works as follows:
    # C[i,k] = sum_{j} A[i,j] * B[j,k]
    # We define:
    #   p_r = sum_j f_r(j) * A[i,j]  for r in 0..3 (4 values per row i)
    #   q_s = sum_j g_s(j) * B[j,k]  for s in 0..3 (4 values per column k)
    # Then C[i,k] = sum_{r,s} h_{r,s} * p_r(i) * q_s(k)
    #
    # For Winograd F(4), the coefficients are:
    #   f_0 = [1, 1, 1, 1]
    #   f_1 = [1, -1, 1, -1]
    #   f_2 = [1, -1, -1, 1]
    #   f_3 = [1, 1, -1, -1]
    #
    #   g_0 = [1, 1, 1, 1]
    #   g_1 = [1, -1, 1, -1]
    #   g_2 = [1, -1, -1, 1]
    #   g_3 = [1, 1, -1, -1]
    #
    # And h_{r,s} = (1/4) * f_r dot g_s (but with appropriate signs)
    #
    # Actually, the correct Winograd F(4) uses:
    #   C[i,k] = (1/4) * sum_{r=0}^{3} sum_{s=0}^{3} w_{r,s} * (sum_j f_r(j) A[i,j]) * (sum_j g_s(j) B[j,k])
    # where w_{r,s} are specific coefficients.
    #
    # Let me just use the known result: Winograd F(4) gives rank 16 for 4x4x4.
    # The 16 rank-1 terms correspond to the 16 products of the 4 A-intermediates
    # with the 4 B-intermediates, but with specific weightings.
    #
    # For simplicity, let me construct the 4x4x4 Winograd decomposition explicitly.
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Build the decomposition by combining:
    # 1. Winograd F(4) for A(4x4) * B1(4x4) -> C1(4x4)
    # 2. Direct multiplication for A(4x4) * B2(4x2) -> C2(4x2)
    #
    # For part 1: Winograd F(4) rank 16
    # For part 2: schoolbook rank 4*4*2 = 32
    # Total: 16 + 32 = 48
    #
    # But we can do better for part 2. A(4x4) * B2(4x2) can be done with
    # rank 4*4*2 = 32 schoolbook, or we can use a smaller Winograd-like scheme.
    #
    # Actually, let me just use schoolbook for both parts first to get a valid solution,
    # then optimize.
    #
    # Schoolbook for 4x4x6: rank = 4*4*6 = 96
    #
    # Let me try a hybrid: Winograd for 4x4x4 (rank 16) + schoolbook for 4x4x2 (rank 32) = 48.
    
    # Define Winograd F(4) coefficients
    # f_r(j) for r,j in 0..3
    f = [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, -1, -1, 1],
        [1, 1, -1, -1]
    ]
    # g_s(j) for s,j in 0..3
    g = [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, -1, -1, 1],
        [1, 1, -1, -1]
    ]
    
    # For Winograd F(4):
    # C[i,k] = (1/4) * sum_{r,s} f_r dot g_s * (sum_j f_r(j) A[i,j]) * (sum_j g_s(j) B[j,k])
    # Wait, that's not quite right. Let me think more carefully.
    #
    # The Winograd F(4) scheme:
    # Define:
    #   M_r(i) = sum_j f_r(j) * A[i,j]   for r=0..3, i=0..3
    #   N_s(k) = sum_j g_s(j) * B[j,k]   for s=0..3, k=0..3
    # Then:
    #   C[i,k] = sum_{r,s} c_{r,s} * M_r(i) * N_s(k)
    # where c_{r,s} are constants.
    #
    # For the standard Winograd F(4), c_{r,s} = (1/4) * delta_{r,s} (diagonal only)?
    # No, that would give rank 4, which is too low.
    #
    # Actually, I recall that Winograd F(4) for 4x4x4 has rank 16, with all 16 products used.
    # The coefficients c_{r,s} are such that the reconstruction is exact.
    #
    # Let me derive: We need
    #   sum_j A[i,j] B[j,k] = sum_{r,s} c_{r,s} (sum_j f_r(j) A[i,j]) (sum_j g_s(j) B[j,k])
    #   = sum_{r,s} c_{r,s} (sum_j f_r(j) g_s(j) A[i,j] B[j,k])  [this doesn't work directly]
    #
    # Actually, the product (sum_j f_r(j) A[i,j]) (sum_j g_s(j) B[j,k]) 
    # = sum_j sum_l f_r(j) g_s(l) A[i,j] B[l,k]
    # This involves A[i,j] and B[l,k] for different j,l, so it's not a simple bilinear form
    # in the original tensor.
    #
    # I think I'm overcomplicating this. Let me just use the known Strassen-like
    # or Winograd decomposition that's well-documented.
    #
    # Alternative: Use Strassen for 4x4x4 (rank 18 or so) + schoolbook for 4x4x2.
    # Strassen 2x2x2 has rank 7. For 4x4x4, we can use 4 Strassen blocks (rank 28)
    # or a more efficient scheme.
    #
    # Actually, the best known for 4x4x4 is rank 49 (from the knowledge in the prompt).
    # Wait, the prompt says "Splitting 4x4x6 into 4x4x4 (rank 49) and 4x4x2 (rank 28) 
    # reduces total rank from 84 to 77". So 4x4x4 has rank 49 and 4x4x2 has rank 28.
    #
    # Let me just use schoolbook for now to ensure correctness, then try to improve.
    
    rank = n * m * p  # 96 for schoolbook
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
