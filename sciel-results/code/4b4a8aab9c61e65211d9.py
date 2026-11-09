import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS
    # We split the multiplication into two parts:
    # 1. 4x4 * 4x4 (first 4 columns of the 4x7 matrix) using Strassen (rank 49)
    # 2. 4x4 * 4x3 (last 3 columns of the 4x7 matrix) using Schoolbook (rank 48)
    # Total rank = 49 + 48 = 97
    
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: Strassen for 4x4 * 4x4 ---
    # Indices for 4x4 blocks:
    # A is 4x4, B is 4x4 (first 4 cols of the 4x7 matrix)
    # We use the standard Strassen algorithm for 4x4 by applying 2x2 Strassen to the 2x2 blocks.
    
    # 2x2 Strassen basis for a single 2x2 multiplication:
    # Inputs: A (2x2), B (2x2)
    # Outputs: C (2x2)
    # The 7 products are:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # For 4x4, we have 4 blocks of A: A00, A01, A10, A11 (each 2x2)
    # And 4 blocks of B: B00, B01, B10, B11 (each 2x2)
    
    # The 7 products for the 4x4 Strassen are:
    # Q1 = (A00 + A11)(B00 + B11)
    # Q2 = (A10 + A11)B00
    # Q3 = A00(B01 - B11)
    # Q4 = A11(B10 - B00)
    # Q5 = (A00 + A01)B11
    # Q6 = (A10 - A00)(B00 + B01)
    # Q7 = (A01 - A11)(B10 + B11)
    
    # Each Qk is a product of two 2x2 matrices, which can be done with 7 rank-1 terms.
    # So total 7 * 7 = 49 terms.
    
    # Let's define the 2x2 Strassen coefficients.
    # For a 2x2 multiplication C = A * B:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We will generate the 49 terms for the 4x4 part.
    # Term index t from 0 to 48.
    # t = q * 7 + s, where q is the 4x4 Strassen product (0-6) and s is the 2x2 Strassen term (0-6).
    
    # Define the 2x2 Strassen basis vectors for U, V, W.
    # For a 2x2 multiplication, the U part (for A) and V part (for B) are:
    # s=0: P1 = (A00+A11)(B00+B11)
    #   U: A00:1, A11:1
    #   V: B00:1, B11:1
    # s=1: P2 = (A10+A11)B00
    #   U: A10:1, A11:1
    #   V: B00:1
    # s=2: P3 = A00(B01-B11)
    #   U: A00:1
    #   V: B01:1, B11:-1
    # s=3: P4 = A11(B10-B00)
    #   U: A11:1
    #   V: B10:1, B00:-1
    # s=4: P5 = (A00+A01)B11
    #   U: A00:1, A01:1
    #   V: B11:1
    # s=5: P6 = (A10-A00)(B00+B01)
    #   U: A10:1, A00:-1
    #   V: B00:1, B01:1
    # s=6: P7 = (A01-A11)(B10+B11)
    #   U: A01:1, A11:-1
    #   V: B10:1, B11:1
    
    # 2x2 Strassen U coefficients (for A00, A01, A10, A11)
    u_2x2 = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    
    # 2x2 Strassen V coefficients (for B00, B01, B10, B11)
    v_2x2 = [
        [1, 0, 0, 1],  # P1
        [1, 0, 0, 0],  # P2
        [0, 1, 0, -1], # P3
        [0, 0, 1, -1], # P4
        [0, 0, 0, 1],  # P5
        [1, 1, 0, 0],  # P6
        [0, 0, 1, 1]   # P7
    ]
    
    # 2x2 Strassen W coefficients (for C00, C01, C10, C11)
    w_2x2 = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, -1], # P2
        [0, 1, 0, 1],  # P3
        [1, 0, 1, 0],  # P4
        [-1, 1, 0, 0], # P5
        [0, 0, 0, 1],  # P6
        [1, 1, 0, 0]   # P7
    ]
    
    # 4x4 Strassen products:
    # Q1 = (A00 + A11)(B00 + B11)
    # Q2 = (A10 + A11)B00
    # Q3 = A00(B01 - B11)
    # Q4 = A11(B10 - B00)
    # Q5 = (A00 + A01)B11
    # Q6 = (A10 - A00)(B00 + B01)
    # Q7 = (A01 - A11)(B10 + B11)
    
    # For each Qk, we need to form the sum of A blocks and B blocks, then apply 2x2 Strassen.
    # The A block for Qk is a linear combination of A00, A01, A10, A11.
    # The B block for Qk is a linear combination of B00, B01, B10, B11.
    
    # Let's define the 4x4 Strassen coefficients for the A and B blocks.
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    
    # Q1: A = A00 + A11, B = B00 + B11
    # Q2: A = A10 + A11, B = B00
    # Q3: A = A00, B = B01 - B11
    # Q4: A = A11, B = B10 - B00
    # Q5: A = A00 + A01, B = B11
    # Q6: A = A10 - A00, B = B00 + B01
    # Q7: A = A01 - A11, B = B10 + B11
    
    # Coefficients for A blocks (A00, A01, A10, A11) for each Qk:
    a_4x4 = [
        [1, 0, 0, 1],  # Q1
        [0, 0, 1, 1],  # Q2
        [1, 0, 0, 0],  # Q3
        [0, 0, 0, 1],  # Q4
        [1, 1, 0, 0],  # Q5
        [-1, 0, 1, 0], # Q6
        [0, 1, 0, -1]  # Q7
    ]
    
    # Coefficients for B blocks (B00, B01, B10, B11) for each Qk:
    b_4x4 = [
        [1, 0, 0, 1],  # Q1
        [1, 0, 0, 0],  # Q2
        [0, 1, 0, -1], # Q3
        [0, 0, 1, -1], # Q4
        [0, 0, 0, 1],  # Q5
        [1, 1, 0, 0],  # Q6
        [0, 0, 1, 1]   # Q7
    ]
    
    # W coefficients for the 4x4 output:
    # C00 = Q1 + Q4 - Q5 + Q7
    # C01 = Q3 + Q5
    # C10 = Q2 + Q4
    # C11 = Q1 - Q2 + Q3 + Q6
    
    w_4x4 = [
        [1, 0, 0, 1],  # Q1
        [0, 0, 1, -1], # Q2
        [0, 1, 0, 1],  # Q3
        [1, 0, 1, 0],  # Q4
        [-1, 1, 0, 0], # Q5
        [0, 0, 0, 1],  # Q6
        [1, 1, 0, 0]   # Q7
    ]
    
    # Now, for each term t in 0..48:
    # t = q * 7 + s
    # q is the 4x4 Strassen product (0-6)
    # s is the 2x2 Strassen term (0-6)
    
    # The U part for this term:
    # The A block for Qq is sum(a_4x4[q][b] * A_b for b in 0..3)
    # Then we apply 2x2 Strassen term s to this A block.
    # The 2x2 Strassen term s uses u_2x2[s] for the 2x2 A block.
    # So the U coefficients for the 4x4 A matrix (A00, A01, A10, A11, each 2x2) are:
    # For each 2x2 block b of A, the coefficient is a_4x4[q][b] * u_2x2[s][c] for each 2x2 element c.
    
    # Similarly for V:
    # The B block for Qq is sum(b_4x4[q][b] * B_b for b in 0..3)
    # Then we apply 2x2 Strassen term s to this B block.
    # The V coefficients for the 4x4 B matrix (B00, B01, B10, B11, each 2x2) are:
    # For each 2x2 block b of B, the coefficient is b_4x4[q][b] * v_2x2[s][c] for each 2x2 element c.
    
    # And for W:
    # The output C block for Qq is sum(w_4x4[q][c] * C_c for c in 0..3)
    # Then we apply 2x2 Strassen term s to this C block.
    # The W coefficients for the 4x4 C matrix (C00, C01, C10, C11, each 2x2) are:
    # For each 2x2 block c of C, the coefficient is w_4x4[q][c] * w_2x2[s][d] for each 2x2 element d.
    
    for q in range(7):
        for s in range(7):
            t = q * 7 + s
            if t >= 49:
                break
            
            # U coefficients for the 4x4 A matrix (16 elements)
            # A is indexed as A[i, j] for i, j in 0..3
            # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
            # For each 2x2 block b (0..3), and each 2x2 element c (0..3):
            #   block b corresponds to A[2*(b//2) : 2*(b//2)+2, 2*(b%2) : 2*(b%2)+2]
            #   element c corresponds to row c//2, col c%2 within the block
            for b in range(4):
                for c in range(4):
                    coeff = a_4x4[q][b] * u_2x2[s][c]
                    if coeff != 0:
                        # Block b: row block = b//2, col block = b%2
                        row_block = b // 2
                        col_block = b % 2
                        row = 2 * row_block + c // 2
                        col = 2 * col_block + c % 2
                        U[row * m + col][t] += coeff
            
            # V coefficients for the 4x4 B matrix (16 elements)
            # B is indexed as B[i, j] for i, j in 0..3 (first 4 cols of the 4x7 matrix)
            # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4], B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
            for b in range(4):
                for c in range(4):
                    coeff = b_4x4[q][b] * v_2x2[s][c]
                    if coeff != 0:
                        # Block b: row block = b//2, col block = b%2
                        row_block = b // 2
                        col_block = b % 2
                        row = 2 * row_block + c // 2
                        col = 2 * col_block + c % 2
                        # B is 4x7, so col is 0..3 for the first 4 cols
                        V[row * p + col][t] += coeff
            
            # W coefficients for the 4x4 C matrix (16 elements)
            # C is indexed as C[i, j] for i, j in 0..3 (first 4 cols of the 4x7 output)
            # C00 = C[0:2, 0:2], C01 = C[0:2, 2:4], C10 = C[2:4, 0:2], C11 = C[2:4, 2:4]
            for c_block in range(4):
                for d in range(4):
                    coeff = w_4x4[q][c_block] * w_2x2[s][d]
                    if coeff != 0:
                        # Block c_block: row block = c_block//2, col block = c_block%2
                        row_block = c_block // 2
                        col_block = c_block % 2
                        row = 2 * row_block + d // 2
                        col = 2 * col_block + d % 2
                        # W is indexed as W[k*n + i, r] where k is col of C, i is row of C
                        # So W[col * n + row, r]
                        W[col * n + row][t] += coeff
    
    # --- Part 2: Schoolbook for 4x4 * 4x3 ---
    # The last 3 columns of the 4x7 matrix (cols 4, 5, 6)
    # Schoolbook: for each i, j, k, add a term
    # U[i*m + j, t] = 1
    # V[j*p + k, t] = 1
    # W[k*n + i, t] = 1
    # t is from 49 to 96 (48 terms)
    
    t = 49
    for i in range(n):
        for j in range(m):
            for k in range(3):  # Last 3 columns (k=4,5,6 in the full 4x7 matrix)
                k_full = k + 4  # 4, 5, 6
                U[i * m + j][t] = 1
                V[j * p + k_full][t] = 1
                W[k_full * n + i][t] = 1
                t += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
