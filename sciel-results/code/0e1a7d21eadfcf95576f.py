"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Block decomposition:
    # A (4x5) = [A4 (4x4) | A1 (4x1)]
    # B (5x5) = [B44 (4x4) B41 (4x1); B14 (1x4) B11 (1x1)]
    # C (4x5) = A4*B44 + A4*B41 + A1*B14 + A1*B11
    
    # Part 1: A4 * B44 (4x4 * 4x4 -> 4x4)
    # Use Strassen for 4x4x4, with Schoolbook for 2x2x2 base.
    # Rank = 7 * 4 = 28.
    
    # Part 2: A4 * B41 (4x4 * 4x1 -> 4x1)
    # Rank = 16 (Schoolbook for each of 16 terms? No, 4x4 terms)
    # C_i4 = sum_j A_ij B_j4. Rank 16.
    
    # Part 3: A1 * B14 (4x1 * 1x4 -> 4x4)
    # C_ik = A_i4 B_4k. Rank 1.
    
    # Part 4: A1 * B11 (4x1 * 1x1 -> 4x1)
    # C_i4 = A_i4 B_44. Rank 1.
    
    total_rank = 28 + 16 + 1 + 1
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 schoolbook term
    # r1, c1: top-left of A block
    # k1: top-left of C block (col in B is c1, row in C is r1)
    # The block is A[r1..r1+1, c1..c1+1] * B[c1..c1+1, k1..k1+1] -> C[r1..r1+1, k1..k1+1]
    # We need to handle signs for Strassen.
    
    def add_2x2_schoolbook(r1, c1, k1, sign=1):
        nonlocal col
        # C_00 = A_00 B_00 + A_01 B_10
        # C_01 = A_00 B_01 + A_01 B_11
        # C_10 = A_10 B_00 + A_11 B_10
        # C_11 = A_10 B_01 + A_11 B_11
        
        # Term 1: A_00 B_00 -> C_00
        U[r1 * m + c1][col] = sign
        V[c1 * p + k1][col] = sign
        W[k1 * n + r1][col] = 1
        col += 1
        
        # Term 2: A_01 B_10 -> C_00
        U[r1 * m + c1 + 1][col] = sign
        V[c1 + 1 * p + k1][col] = sign
        W[k1 * n + r1][col] = 1
        col += 1
        
        # Term 3: A_00 B_01 -> C_01
        U[r1 * m + c1][col] = sign
        V[c1 * p + k1 + 1][col] = sign
        W[(k1 + 1) * n + r1][col] = 1
        col += 1
        
        # Term 4: A_01 B_11 -> C_01
        U[r1 * m + c1 + 1][col] = sign
        V[c1 + 1 * p + k1 + 1][col] = sign
        W[(k1 + 1) * n + r1][col] = 1
        col += 1
        
        # Term 5: A_10 B_00 -> C_10
        U[r1 + 1 * m + c1][col] = sign
        V[c1 * p + k1][col] = sign
        W[k1 * n + r1 + 1][col] = 1
        col += 1
        
        # Term 6: A_11 B_10 -> C_10
        U[r1 + 1 * m + c1 + 1][col] = sign
        V[c1 + 1 * p + k1][col] = sign
        W[k1 * n + r1 + 1][col] = 1
        col += 1
        
        # Term 7: A_10 B_01 -> C_11
        U[r1 + 1 * m + c1][col] = sign
        V[c1 * p + k1 + 1][col] = sign
        W[(k1 + 1) * n + r1 + 1][col] = 1
        col += 1
        
        # Term 8: A_11 B_11 -> C_11
        U[r1 + 1 * m + c1 + 1][col] = sign
        V[c1 + 1 * p + k1 + 1][col] = sign
        W[(k1 + 1) * n + r1 + 1][col] = 1
        col += 1

    # Strassen for 4x4x4
    # M1 = (M11+M22)(N11+N22)
    # M2 = (M21+M22)N11
    # M3 = M11(N12-N22)
    # M4 = M22(N21-N11)
    # M5 = (M11+M12)N22
    # M6 = (M21-M11)(N11+N12)
    # M7 = (M12-M22)(N21+N22)
    
    # We need to decompose the sums/differences into elementary terms.
    # This is complex. Instead, we can just use the 28 rank terms directly.
    # The Strassen algorithm defines 7 products of 2x2 matrices.
    # Each 2x2 product is rank 4.
    # We need to define the 2x2 matrices for each of the 7 steps.
    
    # Step 1: M1 = (A11+A22) * (B11+B22)
    # A_block = A11 + A22. This is a sum of two 2x2 blocks.
    # This requires expanding.
    
    # Alternative: Just use the known rank 28 decomposition coefficients.
    # Or, simpler: Use the schoolbook for 4x4x4? Rank 64.
    # 64 + 16 + 1 + 1 = 82. Still better than 100.
    # But 46 is much better.
    
    # Let's try to implement the Strassen steps by adding/subtracting terms.
    # Actually, the "rank" of the Strassen algorithm is 7 * rank(2x2x2).
    # If we use schoolbook for 2x2x2, rank is 28.
    # We can implement this by iterating over the 7 Strassen terms.
    # For each term, we have a 2x2 matrix multiplication.
    # We can just call add_2x2_schoolbook for each of the 7 terms?
    # No, the inputs to the 2x2 multiplications are sums/differences of blocks.
    # E.g., M1 = (A11+A22)(B11+B22).
    # This is not a single 2x2 multiplication of constant matrices.
    # It is a bilinear form.
    # We need to decompose (A11+A22)(B11+B22) into rank-4 terms?
    # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # This is 4 separate 2x2 multiplications. Rank 4*4=16.
    # Then M1 contributes 16 to the rank?
    # If we do this for all 7 Strassen terms, the rank will be higher.
    
    # Actually, Strassen's algorithm works by reducing the number of multiplications.
    # The "multiplications" are the 2x2 matrix multiplications.
    # If we implement the 2x2 multiplications with rank 4, and we have 7 of them,
    # the total rank is 28.
    # But the inputs to these 2x2 multiplications are linear combinations of A and B blocks.
    # The tensor decomposition handles this by having U and V entries that are sums of deltas.
    # E.g., for M1, the U matrix for the first 2x2 term would have 1s for A11 and A22 blocks.
    
    # Let's implement the 7 Strassen steps.
    # Each step is a 2x2 * 2x2 -> 2x2 multiplication.
    # We will add 4 terms for each step (Schoolbook 2x2x2).
    
    # We need a helper that adds a 2x2 multiplication where the A and B blocks are specified by their offsets and signs.
    # This is getting complicated.
    
    # Let's fall back to Rank 82 (Schoolbook 4x4x4 + others) to ensure correctness and improvement.
    # 82 < 100.
    
    # Reset col
    col = 0
    total_rank = 64 + 16 + 1 + 1
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 Schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: 4x4x1 Schoolbook
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    # Part 3: 4x1x4 Rank 1
    for i in range(4):
        for k in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
    col += 1
    
    # Part 4: 4x1x1 Rank 1
    for i in range(4):
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
    col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
