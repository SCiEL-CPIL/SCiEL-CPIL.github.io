"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # The 7 classical Strassen 2x2 multiplication terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    # 
    # Each term is a product of two sums of submatrices.
    # We define the 28 possible 4x4 blocks (4 A-sums * 7 B-sums)
    # A-sums:
    # S_A0: A00
    # S_A1: A11
    # S_A2: A00 + A11
    # S_A3: A00 + A10 (Note: Strassen uses specific combos, but we can define a basis)
    # Actually, let's stick to the specific Strassen combinations to ensure we find the 56-rank solution.
    
    # For 2x4x8, we can think of A as 2x(4x2), B as (4x2)x8.
    # This is getting complex to generalize Strassen directly.
    # Let's use a known 56-rank decomposition for 2x4x8 if one exists, or verify if 56 is possible.
    # Actually, the rank of 2x4x8 is known to be 56? 
    # Let's check: rank(2x2x2)=7. rank(2x2x4) >= 14? 
    # There is a known decomposition of rank 56 for 2x4x8?
    # Wait, the inspiration said "Strassen decomposition reduces rank from 64 to 56".
    # This implies a specific construction.
    
    # Let's try to construct the 56-rank decomposition directly.
    # The 56-rank decomposition for 2x4x8 is often derived by applying Strassen to the 2x2x2 blocks.
    # If we view the 2x4x8 multiplication as a 2x2x2 block multiplication where each "scalar" is a 2x4 matrix? No.
    # A is 2x4, B is 4x8.
    # We can partition A into [A0 A1] where A0, A1 are 2x2.
    # We can partition B into [B0 B1] where B0, B1 are 2x4? No, B is 4x8.
    # Partition B into 4 rows, 8 cols.
    # Let's partition B into 4 blocks of 1x8? No.
    # Let's partition B into 2 blocks of 2x8: B = [B_top; B_bottom].
    # Then A = [A0 A1] (2x2 each).
    # C = A0 * B_top + A1 * B_bottom.
    # This is just standard multiplication of two 2x2-by-2x8 matrices?
    # No, A0 is 2x2, B_top is 2x8. Product is 2x8.
    # A1 is 2x2, B_bottom is 2x8. Product is 2x8.
    # Sum is 2x8.
    # This requires 2 * (rank of 2x2x8).
    # Rank of 2x2x8?
    # If we use Strassen for 2x2x2, it's 7.
    # For 2x2x4, rank is 14? (2 * 7)
    # For 2x2x8, rank is 28? (4 * 7)
    # Total rank = 2 * 28 = 56.
    # Yes! This matches the "56" score.
    
    # So the strategy is:
    # 1. Split A into two 2x2 matrices: A = [A0 A1].
    # 2. Split B into two 2x8 matrices: B = [B0; B1] (top 2 rows, bottom 2 rows).
    # 3. Compute C0 = A0 * B0 and C1 = A1 * B1.
    # 4. C = C0 + C1.
    # 5. Each 2x2x8 multiplication can be done with rank 28 (4 Strassen 2x2x2 blocks).
    # 6. Total rank = 28 + 28 = 56.
    
    # Let's construct the 28-rank decomposition for 2x2x8 first.
    # A is 2x2, B is 2x8.
    # Split B into 4 2x2 blocks: B = [B00 B01 B02 B03].
    # C = A00*B00 + A01*B01 + A00*B02 + A01*B03 ... wait.
    # A = [[a00, a01], [a10, a11]].
    # B = [[b00, b01, b02, b03], [b10, b11, b12, b13]] where each bij is 2x2? No, B is 2x8.
    # Let's make each bij a 2x2 matrix.
    # B is 2x8. Split into 4 columns of 2x2.
    # B = [B_0, B_1, B_2, B_3] where B_i is 2x2.
    # C = A * B = [A*B_0, A*B_1, A*B_2, A*B_3].
    # Each A*B_i is a 2x2x2 multiplication.
    # Use Strassen for each.
    # Strassen terms for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4 + M7
    # C10 = M1 + M2 - M3 - M5
    # C11 = M2 - M1 + M3 + M6
    
    # We need to map this to the global indices.
    # Global A: 2x4. A[i, j]. i in 0..1, j in 0..3.
    # Global B: 4x8. B[j, k]. j in 0..3, k in 0..7.
    # Global C: 2x8. C[i, k]. i in 0..1, k in 0..7.
    # Trace convention: W[k*n+i, r].
    
    # Step 1: Split A into A0 (cols 0,1) and A1 (cols 2,3).
    # A0[i, j'] where j' in 0..1. A0[i, j'] = A[i, j'].
    # A1[i, j'] where j' in 0..1. A1[i, j'] = A[i, j'+2].
    
    # Step 2: Split B into B0 (rows 0,1) and B1 (rows 2,3).
    # B0[j', k] where j' in 0..1. B0[j', k] = B[j', k].
    # B1[j', k] where j' in 0..1. B1[j', k] = B[j'+2, k].
    
    # Step 3: Compute C0 = A0 * B0.
    # A0 is 2x2, B0 is 2x8.
    # Split B0 into B0_0, B0_1, B0_2, B0_3 (each 2x2).
    # B0_0[k' in 0..1, l in 0..1] = B0[k', l] = B[k', l].
    # B0_1[k' in 0..1, l in 0..1] = B0[k', l+2] = B[k', l+2].
    # ...
    # C0 is 2x8.
    # C0[i, k] = sum_{j' in 0..1} A0[i, j'] * B0[j', k].
    
    # We will build the U, V, W matrices for the 56 terms.
    # There are 2 (for A0/A1) * 4 (for B0_0..B0_3) * 7 (Strassen terms) = 56 terms.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    # Strassen coefficients for C00, C01, C10, C11
    # M1, M2, M3, M4, M5, M6, M7
    # C00: M1, -M4, M5, M7 -> coeffs: 1, 0, 0, -1, 1, 0, 1
    # C01: M3, M4, M7 -> coeffs: 0, 0, 1, 1, 0, 0, 1
    # C10: M1, M2, -M3, -M5 -> coeffs: 1, 1, -1, 0, -1, 0, 0
    # C11: -M1, M2, M3, M6 -> coeffs: -1, 1, 1, 0, 0, 1, 0
    
    strassen_coeffs = [
        [1, 0, 0, -1, 1, 0, 1],   # C00
        [0, 0, 1, 1, 0, 0, 1],    # C01
        [1, 1, -1, 0, -1, 0, 0],  # C10
        [-1, 1, 1, 0, 0, 1, 0]    # C11
    ]
    
    # Strassen factors for A and B
    # M1: (A00+A11), (B00+B11)
    # M2: (A10+A11), B00
    # M3: A00, (B01-B11)
    # M4: A11, (B10-B00)
    # M5: (A00+A10), B11
    # M6: (A01-A00), (B00+B10)
    # M7: (A10-A11), (B01+B11)
    
    # A factors (for 2x2 A)
    # A00, A01, A10, A11
    a_factors = [
        [[1, 0], [0, 1]],   # M1: A00+A11
        [[0, 0], [1, 1]],   # M2: A10+A11
        [[1, 0], [0, 0]],   # M3: A00
        [[0, 0], [0, 1]],   # M4: A11
        [[1, 0], [1, 0]],   # M5: A00+A10
        [[0, 1], [-1, 0]],  # M6: A01-A00
        [[0, 0], [1, -1]]   # M7: A10-A11
    ]
    
    # B factors (for 2x2 B)
    b_factors = [
        [[1, 1], [1, 1]],   # M1: B00+B11
        [[1, 0], [0, 0]],   # M2: B00
        [[0, 1], [0, -1]],  # M3: B01-B11
        [[0, 0], [1, -1]],  # M4: B10-B00
        [[0, 0], [0, 1]],   # M5: B11
        [[1, 1], [1, 0]],   # M6: B00+B10
        [[0, 1], [0, 1]]    # M7: B01+B11
    ]
    
    # Loop over the 2 main blocks (A0*B0 and A1*B1)
    for main_block in range(2):
        # main_block 0: A0, B0
        # main_block 1: A1, B1
        
        # A0 is A[:, 0:2], A1 is A[:, 2:4]
        # B0 is B[0:2, :], B1 is B[2:4, :]
        
        for b_block in range(4):
            # b_block 0: B_0 (cols 0:2)
            # b_block 1: B_1 (cols 2:4)
            # b_block 2: B_2 (cols 4:6)
            # b_block 3: B_3 (cols 6:8)
            
            for s in range(7):
                # s is the Strassen term index
                
                # Update U
                # U index: i*m + j. i in 0..1, j in 0..3.
                # For main_block 0: j is 0 or 1. A0[i, j] = A[i, j].
                # For main_block 1: j is 2 or 3. A1[i, j-2] = A[i, j].
                
                for i in range(2):
                    for j_prime in range(2):
                        val = a_factors[s][i][j_prime]
                        if val != 0:
                            j = j_prime + main_block * 2
                            U[i * 4 + j][col] += val
                
                # Update V
                # V index: j*p + k. j in 0..3, k in 0..7.
                # For main_block 0: j is 0 or 1. B0[j, k] = B[j, k].
                # For main_block 1: j is 2 or 3. B1[j-2, k] = B[j, k].
                
                # b_block determines k range.
                # k_prime in 0..1. k = k_prime + b_block * 2.
                
                for j_prime in range(2):
                    for k_prime in range(2):
                        val = b_factors[s][j_prime][k_prime]
                        if val != 0:
                            j = j_prime + main_block * 2
                            k = k_prime + b_block * 2
                            V[j * 8 + k][col] += val
                
                # Update W
                # W index: k*n + i. k in 0..7, i in 0..1.
                # C[i, k] = sum C0[i, k] + C1[i, k].
                # C0[i, k] is from A0*B0.
                # C1[i, k] is from A1*B1.
                # The Strassen result gives C00, C01, C10, C11 for the 2x2 result.
                # Here the "2x2 result" is actually 2x2 submatrix of the 2x8 result.
                # So C0[i, k] = sum_{j'} A0[i, j'] B0[j', k].
                # The Strassen decomposition computes this 2x2 (in terms of the 2x2 blocks) product.
                # The result indices are i (0..1) and k_prime (0..1).
                # So W[k*n + i] gets the coefficient.
                
                # The Strassen term s contributes to C00, C01, C10, C11 with coeffs.
                # C00 -> i=0, k_prime=0
                # C01 -> i=0, k_prime=1
                # C10 -> i=1, k_prime=0
                # C11 -> i=1, k_prime=1
                
                for i in range(2):
                    for k_prime in range(2):
                        val = strassen_coeffs[i][k_prime][s]
                        if val != 0:
                            k = k_prime + b_block * 2
                            W[k * 2 + i][col] += val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
