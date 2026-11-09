"""Matrix multiplication 4x4x4: recursive Strassen decomposition of rank 49."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Recursive Strassen decomposition for 4x4x4 matrix multiplication.
    
    4x4 matrices are viewed as 2x2 blocks of 2x2 matrices.
    Level 1: Strassen on 2x2 block matrices -> 7 block multiplications
    Level 2: Each block multiplication is 2x2x2x2, done via Strassen -> 7 scalar mults
    
    Total rank = 7 * 7 = 49
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # We'll build U, V, W as lists of lists (integers or half-integers)
    # U: shape (n*m, rank) = (16, 49)
    # V: shape (m*p, rank) = (16, 49)  
    # W: shape (p*n, rank) = (16, 49)
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Strassen 2x2 coefficients:
    # For multiplying 2x2 matrices A*B = C:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # For the outer Strassen (4x4 as 2x2 blocks of 2x2):
    # Let A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # Let B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = [[C00, C01], [C10, C11]] where each Ci is 2x2
    #
    # Outer Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mk is a 2x2 matrix product, decomposed via inner Strassen (7 mults each)
    # Total: 7 * 7 = 49 rank-1 terms
    
    # Define the 7 Strassen multiplications for 2x2:
    # For A*B where A, B are 2x2:
    # The 7 products are:
    # s1: (a00+a11) * (b00+b11)
    # s2: (a10+a11) * b00
    # s3: a00 * (b01-b11)
    # s4: a11 * (b10-b00)
    # s5: (a00+a01) * b11
    # s6: (a10-a00) * (b00+b01)
    # s7: (a01-a11) * (b10+b11)
    
    # Coefficients for each Strassen multiplication:
    # (left_coeffs, right_coeffs) for each of the 7
    strassen_coeffs = [
        # (left: [a00, a01, a10, a11], right: [b00, b01, b10, b11])
        ([1, 0, 0, 1], [1, 0, 0, 1]),   # s1
        ([0, 0, 1, 1], [1, 0, 0, 0]),   # s2
        ([1, 0, 0, 0], [0, 1, 0, -1]),  # s3
        ([0, 0, 0, 1], [0, 0, 1, -1]),  # s4
        ([1, 1, 0, 0], [0, 0, 0, 1]),   # s5
        ([-1, 0, 1, 0], [1, 1, 0, 0]),  # s6
        ([0, 1, 0, -1], [0, 0, 1, 1]),  # s7
    ]
    
    # Output coefficients for Strassen 2x2:
    # C00 = s1 + s4 - s5 + s7
    # C01 = s3 + s5
    # C10 = s2 + s4
    # C11 = s1 - s2 + s3 + s6
    output_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ]
    
    # Outer Strassen coefficients (same structure, for block matrices):
    # M1 = (A00+A11)(B00+B11) -> left: A00+A11, right: B00+B11
    # M2 = (A10+A11)B00 -> left: A10+A11, right: B00
    # M3 = A00(B01-B11) -> left: A00, right: B01-B11
    # M4 = A11(B10-B00) -> left: A11, right: B10-B00
    # M5 = (A00+A01)B11 -> left: A00+A01, right: B11
    # M6 = (A10-A00)(B00+B01) -> left: A10-A00, right: B00+B01
    # M7 = (A01-A11)(B10+B11) -> left: A01-A11, right: B10+B11
    
    # Outer output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    outer_output_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ]
    
    # Now build the decomposition
    # Indexing: A is 4x4, indexed as A[row][col] where row, col in 0..3
    # Block structure: A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    
    # For U: index is (i, j) where i is row of A (0..3), j is col of A (0..3)
    # A[i,j] is in block: block_row = i//2, block_col = j//2, local_row = i%2, local_col = j%2
    
    # For V: index is (j, k) where j is col of A (0..3), k is row of B (0..3)
    # B[k,j]... wait, let me be careful with conventions.
    
    # The tensor for matrix multiplication: T[i,j,k] = A[i,j] * B[j,k]
    # U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
    # So: U indexed by (i,j), V indexed by (j,k), W indexed by (k,i)
    
    # A[i,j]: block (i//2, j//2), local (i%2, j%2)
    # B[j,k]: block (j//2, k//2), local (j%2, k%2)  -- wait, B is m x p = 4x4, B[j,k]
    # Actually B[j,k] where j is from A's column space, k is output row
    # C[k,i] = sum_j A[i,j] * B[j,k]
    
    # Let me define:
    # A[i,j] is in outer block (ar, ac) = (i//2, j//2), inner position (lr, lc) = (i%2, j%2)
    # B[j,k] is in outer block (br, bc) = (j//2, k//2), inner position (lr, lc) = (j%2, k%2)
    # C[k,i] is in outer block (cr, cc) = (k//2, i//2), inner position (lr, lc) = (k%2, i%2)
    
    # For each outer Strassen multiplication M_m (m=0..6):
    # M_m = (linear combo of A blocks) * (linear combo of B blocks)
    # Then for each inner Strassen multiplication s_s (s=0..6):
    # s_s = (linear combo of A sub-block elements) * (linear combo of B sub-block elements)
    # The rank-1 term contributes to W with coefficient = outer_output[ci, m] * inner_output[li, s]
    
    # Let's build it:
    for m in range(7):  # outer Strassen multiplication index
        for s in range(7):  # inner Strassen multiplication index
            c = m * 7 + s  # rank-1 term index
            
            # Get the outer coefficients for A blocks and B blocks
            # Outer Strassen left side (A blocks):
            # M1: A00 + A11 -> blocks (0,0) and (1,1) with coeff 1
            # M2: A10 + A11 -> blocks (1,0) and (1,1) with coeff 1
            # M3: A00 -> block (0,0) with coeff 1
            # M4: A11 -> block (1,1) with coeff 1
            # M5: A00 + A01 -> blocks (0,0) and (0,1) with coeff 1
            # M6: A10 - A00 -> blocks (1,0) and (0,0) with coeff 1, -1
            # M7: A01 - A11 -> blocks (0,1) and (1,1) with coeff 1, -1
            
            outer_A_blocks = {
                0: [(0,0,1), (1,1,1)],   # M1
                1: [(1,0,1), (1,1,1)],   # M2
                2: [(0,0,1)],            # M3
                3: [(1,1,1)],            # M4
                4: [(0,0,1), (0,1,1)],   # M5
                5: [(1,0,1), (0,0,-1)],  # M6
                6: [(0,1,1), (1,1,-1)],  # M7
            }
            
            # Outer Strassen right side (B blocks):
            # M1: B00 + B11 -> blocks (0,0) and (1,1)
            # M2: B00 -> block (0,0)
            # M3: B01 - B11 -> blocks (0,1) and (1,1)
            # M4: B10 - B00 -> blocks (1,0) and (0,0)
            # M5: B11 -> block (1,1)
            # M6: B00 + B01 -> blocks (0,0) and (0,1)
            # M7: B10 + B11 -> blocks (1,0) and (1,1)
            
            outer_B_blocks = {
                0: [(0,0,1), (1,1,1)],   # M1
                1: [(0,0,1)],            # M2
                2: [(0,1,1), (1,1,-1)],  # M3
                3: [(1,0,1), (0,0,-1)],  # M4
                4: [(1,1,1)],            # M5
                5: [(0,0,1), (0,1,1)],   # M6
                6: [(1,0,1), (1,1,1)],   # M7
            }
            
            # Inner Strassen coefficients (for 2x2 matrix multiplication)
            # A local: [a00, a01, a10, a11] indexed as (lr, lc)
            # B local: [b00, b01, b10, b11] indexed as (lr, lc)
            # s1: (a00+a11)(b00+b11)
            # s2: (a10+a11)b00
            # s3: a00(b01-b11)
            # s4: a11(b10-b00)
            # s5: (a00+a01)b11
            # s6: (a10-a00)(b00+b01)
            # s7: (a01-a11)(b10+b11)
            
            inner_A_elems = {
                0: [(0,0,1), (1,1,1)],   # s1
                1: [(1,0,1), (1,1,1)],   # s2
                2: [(0,0,1)],            # s3
                3: [(1,1,1)],            # s4
                4: [(0,0,1), (0,1,1)],   # s5
                5: [(1,0,1), (0,0,-1)],  # s6
                6: [(0,1,1), (1,1,-1)],  # s7
            }
            
            inner_B_elems = {
                0: [(0,0,1), (1,1,1)],   # s1
                1: [(0,0,1)],            # s2
                2: [(0,1,1), (1,1,-1)],  # s3
                3: [(1,0,1), (0,0,-1)],  # s4
                4: [(1,1,1)],            # s5
                5: [(0,0,1), (0,1,1)],   # s6
                6: [(1,0,1), (1,1,1)],   # s7
            }
            
            # Inner output coefficients:
            # C00 = s1 + s4 - s5 + s7
            # C01 = s3 + s5
            # C10 = s2 + s4
            # C11 = s1 - s2 + s3 + s6
            inner_output = {
                (0,0): [1, 0, 0, 1, -1, 0, 1],
                (0,1): [0, 0, 1, 0, 1, 0, 0],
                (1,0): [0, 1, 0, 1, 0, 0, 0],
                (1,1): [1, -1, 1, 0, 0, 1, 0],
            }
            
            # Outer output coefficients:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            outer_output = {
                (0,0): [1, 0, 0, 1, -1, 0, 1],
                (0,1): [0, 0, 1, 0, 1, 0, 0],
                (1,0): [0, 1, 0, 1, 0, 0, 0],
                (1,1): [1, -1, 1, 0, 0, 1, 0],
            }
            
            # Now fill in U, V, W for this rank-1 term
            # U[i*m + j, c]: i is row of A, j is col of A
            # For each A block (ar, ac) with outer coeff oa, and inner elem (lr, lc) with inner coeff ia:
            #   i = ar*2 + lr, j = ac*2 + lc
            #   U[i*4 + j, c] += oa * ia
            
            for (ar, ac, oa) in outer_A_blocks[m]:
                for (lr, lc, ia) in inner_A_elems[s]:
                    i = ar * 2 + lr
                    j = ac * 2 + lc
                    U[i * 4 + j][c] += oa * ia
            
            # V[j*p + k, c]: j is col of A (row of B), k is row of C (col of B)
            # B[j,k]: block (br, bc) = (j//2, k//2), local (lr, lc) = (j%2, k%2)
            for (br, bc, ob) in outer_B_blocks[m]:
                for (lr, lc, ib) in inner_B_elems[s]:
                    j = br * 2 + lr
                    k = bc * 2 + lc
                    V[j * 4 + k][c] += ob * ib
            
            # W[k*n + i, c]: k is row of C, i is col of C
            # C[k,i]: block (cr, cc) = (k//2, i//2), local (lr, lc) = (k%2, i%2)
            # The coefficient is outer_output[(cr,cc)][m] * inner_output[(lr,lc)][s]
            for cr in range(2):
                for cc in range(2):
                    outer_coeff = outer_output[(cr, cc)][m]
                    if outer_coeff == 0:
                        continue
                    for lr in range(2):
                        for lc in range(2):
                            inner_coeff = inner_output[(lr, lc)][s]
                            if inner_coeff == 0:
                                continue
                            k = cr * 2 + lr
                            i = cc * 2 + lc
                            W[k * 4 + i][c] += outer_coeff * inner_coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
