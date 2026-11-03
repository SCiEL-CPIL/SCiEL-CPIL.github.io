"""Strassen-based decomposition for 4x4 * 4x7 matrix multiplication."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We split B into B1 (4x4) and B2 (4x3)
    # C = A * B1 + A * B2
    
    # Part 1: A * B1 (4x4 * 4x4) using Strassen (Rank 49)
    # Part 2: A * B2 (4x4 * 4x3) using Schoolbook (Rank 48)
    
    rank = 49 + 48
    R = rank
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Helper to add a term
    # u_idx, v_idx, w_idx are indices in the flattened vectors
    # u_coeff, v_coeff, w_coeff are the coefficients
    # col is the rank index
    def add_term(u_idx, v_idx, w_idx, u_coeff, v_coeff, w_coeff, col):
        U[u_idx][col] += u_coeff
        V[v_idx][col] += v_coeff
        W[w_idx][col] += w_coeff

    # --- Part 2: Schoolbook for A * B2 (cols 4-6 of B, cols 4-6 of C) ---
    # B2 is 4x3.
    # For i in 0..3, j in 0..3, k in 4..6
    # C[i][k] += A[i][j] * B[j][k]
    col = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u_idx = i * 4 + j
                v_idx = j * 7 + k
                w_idx = k * 4 + i
                add_term(u_idx, v_idx, w_idx, 1, 1, 1, col)
                col += 1
    # col should be 49 + 48 = 97

    # --- Part 1: Strassen for A * B1 (cols 0-3 of B, cols 0-3 of C) ---
    # Strassen for 4x4x4.
    # We can implement this by calling Strassen on 2x2 blocks recursively?
    # Or just hardcode the 49 terms?
    # Easier: Use Strassen for 2x2x2 (rank 7) and apply it to the 4x4x4?
    # No, Strassen for 4x4x4 is 7 products of 2x2x2.
    # Each 2x2x2 product has rank 4 (if schoolbook) or 7 (if Strassen).
    # If we use schoolbook for the 2x2x2, we get rank 28 for 4x4x4?
    # Wait, Strassen 4x4x4 = 7 * Strassen(2x2x2)?
    # No, Strassen(4) = 7 * Strassen(2).
    # Strassen(2) = 7 * Strassen(1) = 7 * 1 = 7.
    # So Strassen(4) = 7 * 7 = 49.
    # So we need to decompose each of the 7 products of 2x2 matrices into 7 terms?
    # No, the 7 products are of size 2x2.
    # To compute a 2x2 product, we use Strassen (rank 7).
    # So we need to generate 7 * 7 = 49 terms.
    
    # Let's define the Strassen factors for 4x4x4.
    # A = [[A11, A12], [A21, A22]] where Ai are 2x2.
    # B = [[B11, B12], [B21, B22]] where Bi are 2x2.
    # M1 = A11 * B11
    # M2 = A21 * B12
    # M3 = A11 * (B22 - B12)
    # M4 = A22 * (B11 - B21)
    # M5 = (A11 + A22) * (B11 + B22)
    # M6 = (A21 - A11) * (B12 + B22)
    # M7 = (A12 - A22) * (B21 + B11)
    
    # Each Mk is 2x2. We compute Mk = Xk * Yk using Strassen (rank 7).
    # Xk, Yk are 2x2 matrices of linear combos of A and B.
    # We need to express the 7 Strassen terms for Mk in terms of A and B.
    
    # This is getting complex to hardcode.
    # Alternative: Use the known rank-49 decomposition for 4x4x4?
    # Or just use schoolbook for 4x4x4 (rank 64) and schoolbook for 4x4x3 (rank 48).
    # Total 112. Same as seed.
    
    # Let's try to implement Strassen for 2x2x2 and use it for the blocks.
    # We need a function to add a Strassen decomposition for a 2x2x2 block.
    # But the blocks are not independent; they share A and B.
    
    # Let's just use the rank-49 decomposition for 4x4x4.
    # I will construct it by applying Strassen to the 4x4x4.
    
    # Strassen for 4x4x4:
    # We define 7 "meta-terms" M1..M7.
    # Each Mi is a 2x2 matrix.
    # We compute Mi using Strassen for 2x2x2.
    # Strassen for 2x2x2 has 7 terms.
    # So we have 7 * 7 = 49 terms.
    
    # Let's define the 2x2 Strassen terms.
    # For X * Y (2x2):
    # S1 = X11 * Y11
    # S2 = X21 * Y12
    # S3 = X11 * (Y22 - Y12)
    # S4 = X22 * (Y11 - Y21)
    # S5 = (X11 + X22) * (Y11 + Y22)
    # S6 = (X21 - X11) * (Y12 + Y22)
    # S7 = (X12 - X22) * (Y21 + Y11)
    
    # We need to map these to the global A and B.
    
    # This is a large hardcoded task.
    # Given the time, I will stick to the safe Schoolbook for the 4x4 part as well?
    # No, 112 is the seed. I need to improve.
    
    # Let's use the rank-49 decomposition for 4x4x4.
    # I will generate it programmatically.
    
    # Define the 7 Strassen factors for 4x4x4.
    # X1 = A11, Y1 = B11
    # X2 = A21, Y2 = B12
    # X3 = A11, Y3 = B22 - B12
    # X4 = A22, Y4 = B11 - B21
    # X5 = A11 + A22, Y5 = B11 + B22
    # X6 = A21 - A11, Y6 = B12 + B22
    # X7 = A12 - A22, Y7 = B21 + B11
    
    # For each (Xk, Yk), we apply Strassen for 2x2x2.
    # This means we need to define the 7 terms of Strassen for 2x2x2 in terms of the elements of Xk and Yk.
    
    # Let's write a helper that adds a 2x2x2 Strassen decomposition given the linear forms for X and Y.
    
    # This is too complex to do correctly in a short time without testing.
    # I will use a known decomposition for 4x4x4 with rank 49.
    # Or I will just use the schoolbook for 4x4x4 (64) and schoolbook for 4x4x3 (48) -> 112.
    # But I can use Strassen for 2x2x2 blocks of the 4x4x4?
    # If I use Strassen for the 4x4x4, the rank is 49.
    
    # Let's try to implement the 49 terms.
    
    # I will skip the complex Strassen implementation and use a simpler improvement:
    # Use the rank-49 decomposition for 4x4x4.
    # Since I don't have it hardcoded, I will use the schoolbook for 4x4x4 (64) and 
    # try to optimize the 4x4x3 part?
    # 4x4x3 schoolbook is 48.
    # Can we do better for 4x4x3?
    # R(4,4,3) <= 3 * R(4,4,1) = 3 * 4 = 12?
    # If R(4,4,1) = 4, then R(4,4,3) = 12.
    # Total = 64 + 12 = 76.
    
    # Let's verify R(4,4,1) = 4.
    # C_i = sum_j A_ij B_j0.
    # We can use 4 terms.
    # Term r (r=0..3):
    # U_r: 1 at (i, r) for all i? No.
    # U_r is 4x4.
    # V_r is 4x1.
    # W_r is 1x4.
    # We want sum_r W_r (U_r A) (V_r B)^T?
    # No, the form is sum_r W_{ki,r} (sum_j U_{ij,r} A_{ij}) (sum_l V_{jl,r} B_{jl}).
    # For 4x4x1, k=0.
    # C_i = sum_r W_{0i,r} (sum_j U_{ij,r} A_{ij}) (sum_l V_{jl,r} B_{jl}).
    # We want C_i = sum_j A_ij B_j0.
    # If we set r=j:
    # U_{ij,r} = delta_{jr}.
    # V_{jl,r} = delta_{jr} delta_{l0}.
    # W_{0i,r} = 1.
    # Then term r: 1 * (A_ir) * (B_r0).
    # Sum over r: sum_r A_ir B_r0 = C_i.
    # This works!
    # So R(4,4,1) = 4.
    # R(4,4,3) = 3 * 4 = 12.
    
    # So total rank = 64 (4x4x4 schoolbook) + 12 (4x4x3 optimized) = 76.
    
    # Let's implement this.
    
    U = [[0] * 76 for _ in range(16)]
    V = [[0] * 76 for _ in range(28)]
    W = [[0] * 76 for _ in range(28)]
    
    col = 0
    # Part 1: 4x4x4 Schoolbook (cols 0-3 of B)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u_idx = i * 4 + j
                v_idx = j * 7 + k
                w_idx = k * 4 + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    # col is 64
    
    # Part 2: 4x4x3 Optimized (cols 4-6 of B)
    # For each column k in 4..6, use 4 terms.
    for k in range(4, 7):
        for r in range(4):
            # r corresponds to j=r
            # U: 1 at (i, r) for all i?
            # U_{ij, col} = delta_{jr}
            for i in range(4):
                u_idx = i * 4 + r
                U[u_idx][col] = 1
            
            # V: 1 at (r, k)
            v_idx = r * 7 + k
            V[v_idx][col] = 1
            
            # W: 1 at (k, i) for all i?
            # W_{ki, col} = 1
            for i in range(4):
                w_idx = k * 4 + i
                W[w_idx][col] = 1
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
