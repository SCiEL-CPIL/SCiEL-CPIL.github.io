"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    import time
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # 2x2x2 Strassen rank-7 decomposition kernel
    # U: 4x7, rows (i,j) for i,j in {0,1}
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],   # (0,0)
        [0, 0, 0, 0, 1, 0, 1],    # (0,1)
        [0, 1, 0, 0, 0, 1, 0],    # (1,0)
        [1, 1, 0, 1, 0, 0, -1],   # (1,1)
    ]
    # V: 4x7, rows (j,k) for j,k in {0,1}
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],   # (0,0)
        [0, 0, 1, 0, 0, 1, 0],    # (0,1)
        [0, 0, 0, 1, 0, 0, 1],    # (1,0)
        [1, 0, -1, 0, 1, 0, 1],   # (1,1)
    ]
    # W: 4x7, rows (k,i) for k,i in {0,1}
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],   # (0,0)
        [0, 1, 0, 1, 0, 0, 0],    # (0,1)
        [0, 0, 1, 0, 1, 0, 0],    # (1,0)
        [1, -1, 1, 0, 0, 1, 0],   # (1,1)
    ]
    
    total_rank = 77
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # Strassen products M1..M7
    # Each Mr is a 2x2x2 multiplication.
    # We need to map the linear combinations of blocks to the global U, V, W.
    
    # Define the 7 Strassen steps for 4x4x4
    # Each step consists of:
    # - A linear combination of A blocks (2x2)
    # - A linear combination of B blocks (2x2)
    # - A linear combination of C blocks (2x2)
    
    # Blocks of A: A00, A01, A10, A11
    # Blocks of B: B00, B01, B10, B11
    # Blocks of C: C00, C01, C10, C11
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A11 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 + M6 + M7
    # C01 = M2 + M4
    # C10 = M3 + M5
    # C11 = M1 - M5 + M2 - M6
    
    # We will iterate through the 7 products.
    # For each product, we have 7 rank-1 terms from the 2x2x2 kernel.
    
    # Helper to add a 2x2x2 kernel contribution
    # a_blocks: dict mapping block_name to coefficient
    # b_blocks: dict mapping block_name to coefficient
    # c_blocks: dict mapping block_name to coefficient
    
    def add_kernel(a_coeffs, b_coeffs, c_coeffs, col_start):
        # a_coeffs: { (bi, bj): coeff }
        # b_coeffs: { (bj, bk): coeff }
        # c_coeffs: { (bk, bi): coeff }
        
        for c in range(7):
            col = col_start + c
            # U part
            for bi in range(2):
                for bj in range(2):
                    coeff_a = a_coeffs.get((bi, bj), 0)
                    if coeff_a != 0:
                        for di in range(2):
                            for dj in range(2):
                                i = 2 * bi + di
                                j = 2 * bj + dj
                                U[i * m + j][col] += coeff_a * S_U[di * 2 + dj][c]
                                
            # V part
            for bj in range(2):
                for bk in range(2):
                    coeff_b = b_coeffs.get((bj, bk), 0)
                    if coeff_b != 0:
                        for dj in range(2):
                            for dk in range(2):
                                j = 2 * bj + dj
                                k = 2 * bk + dk
                                V[j * p + k][col] += coeff_b * S_V[dj * 2 + dk][c]
                                
            # W part
            for bk in range(2):
                for bi in range(2):
                    coeff_c = c_coeffs.get((bk, bi), 0)
                    if coeff_c != 0:
                        for dk in range(2):
                            for di in range(2):
                                k = 2 * bk + dk
                                i = 2 * bi + di
                                W[k * n + i][col] += coeff_c * S_W[dk * 2 + di][c]

    # M1: (A00 + A11)(B00 + B11) -> C00, C11
    a1 = {(0,0): 1, (1,1): 1}
    b1 = {(0,0): 1, (1,1): 1}
    c1 = {(0,0): 1, (1,1): 1} # Contribution to C00 and C11?
    # Wait, the kernel output is a single 2x2 matrix.
    # The Strassen algorithm sums these kernels into the C blocks.
    # The "c_coeffs" should represent which C blocks this kernel's result is added to.
    # M1 contributes to C00 and C11.
    # So for the W matrix, we add the kernel result to C00 and C11.
    # But the kernel S_W is for a single 2x2 output block.
    # So we call add_kernel twice? Or modify c_coeffs?
    # c_coeffs maps (bk, bi) to coeff.
    # For M1, the result is added to C00 (bk=0, bi=0) and C11 (bk=1, bi=1).
    # So c1 = {(0,0): 1, (1,1): 1}.
    
    add_kernel(a1, b1, c1, col_offset)
    col_offset += 7
    
    # M2: (A10 + A11)B00 -> C01, C11
    a2 = {(1,0): 1, (1,1): 1}
    b2 = {(0,0): 1}
    c2 = {(0,1): 1, (1,1): 1} # C01 and C11?
    # C01 = M2 + M4. C11 = M1 - M5 + M2 - M6.
    # So M2 contributes to C01 (bk=0, bi=1) and C11 (bk=1, bi=1).
    add_kernel(a2, b2, c2, col_offset)
    col_offset += 7
    
    # M3: A00(B10 - B11) -> C10
    a3 = {(0,0): 1}
    b3 = {(1,0): 1, (1,1): -1}
    c3 = {(1,0): 1} # C10
    add_kernel(a3, b3, c3, col_offset)
    col_offset += 7
    
    # M4: A11(B01 - B00) -> C00, C01
    a4 = {(1,1): 1}
    b4 = {(0,1): 1, (0,0): -1}
    c4 = {(0,0): -1, (0,1): 1} # C00 (coeff -1), C01 (coeff 1)
    add_kernel(a4, b4, c4, col_offset)
    col_offset += 7
    
    # M5: (A00 + A10)B11 -> C10, C11
    a5 = {(0,0): 1, (1,0): 1}
    b5 = {(1,1): 1}
    c5 = {(1,0): 1, (1,1): -1} # C10 (coeff 1), C11 (coeff -1)
    add_kernel(a5, b5, c5, col_offset)
    col_offset += 7
    
    # M6: (A11 - A00)(B00 + B01) -> C00, C11
    a6 = {(1,1): 1, (0,0): -1}
    b6 = {(0,0): 1, (0,1): 1}
    c6 = {(0,0): 1, (1,1): -1} # C00 (coeff 1), C11 (coeff -1)
    add_kernel(a6, b6, c6, col_offset)
    col_offset += 7
    
    # M7: (A01 - A11)(B10 + B11) -> C00, C01
    a7 = {(0,1): 1, (1,1): -1}
    b7 = {(1,0): 1, (1,1): 1}
    c7 = {(0,0): 1, (0,1): 1} # C00 (coeff 1), C01 (coeff 1)
    add_kernel(a7, b7, c7, col_offset)
    col_offset += 7
    
    # Part 2: 4x4x2 (Rank 28)
    # Columns 4, 5 of B.
    # Split B2 (4x2) into B20 (rows 0,1) and B21 (rows 2,3).
    # Split C2 (4x2) into C20 (rows 0,1) and C21 (rows 2,3).
    # C20 = A00 B20 + A01 B21
    # C21 = A10 B20 + A11 B21
    
    # Term 1: A00 B20 -> C20
    # A00 is block (0,0) of A.
    # B20 is block (0, 2) of B? No.
    # B is 4x6.
    # B2 is columns 4,5.
    # B20 is rows 0,1, cols 4,5.
    # In global V index (j, k), j in 0,1, k in 4,5.
    # This is a 2x2 block of B.
    # Let's map this to the kernel.
    # The kernel expects B to be 2x2.
    # We can treat this as a separate 2x2x2 multiplication.
    # A_block: (0,0)
    # B_block: rows 0,1, cols 4,5.
    # C_block: rows 0,1, cols 4,5.
    
    # We need to be careful with indices.
    # The kernel add_kernel assumes B is 2x2x2 with local indices.
    # Here, the "local" B is at global (j=0..1, k=4..5).
    # The "local" A is at global (i=0..1, j=0..1).
    # The "local" C is at global (k=4..5, i=0..1).
    
    # We can reuse add_kernel if we adjust the index mapping.
    # But add_kernel uses fixed 2x2 block offsets.
    # Let's write a specific add_kernel_2 that takes explicit global block offsets.
    
    def add_kernel_global(a_bi, a_bj, b_bj, b_bk, c_bk, c_bi, col_start):
        # a_bi, a_bj: block indices for A (0 or 1)
        # b_bj, b_bk: block indices for B (0 or 1 for j, 2 or 3 for k? No, 0 or 1 for local)
        # Wait, B is 4x6.
        # For the 4x4x2 part, we are looking at columns 4,5.
        # So k is 4,5.
        # We split rows j into 0,1 and 2,3.
        # So b_bj is 0 or 1.
        # b_bk is 0 or 1 (local index for the 2 columns).
        # Global k = 4 + b_bk.
        
        for c in range(7):
            col = col_start + c
            # U part
            for di in range(2):
                for dj in range(2):
                    i = 2 * a_bi + di
                    j = 2 * a_bj + dj
                    U[i * m + j][col] = S_U[di * 2 + dj][c]
                    
            # V part
            for dj in range(2):
                for dk in range(2):
                    j = 2 * b_bj + dj
                    k = 4 + b_bk * 2 + dk # Global k index
                    V[j * p + k][col] = S_V[dj * 2 + dk][c]
                    
            # W part
            for dk in range(2):
                for di in range(2):
                    k = 4 + c_bk * 2 + dk # Global k index
                    i = 2 * c_bi + di
                    W[k * n + i][col] = S_W[dk * 2 + di][c]

    # Term 1: A00 B20 -> C20
    # A00: bi=0, bj=0
    # B20: bj=0, bk=0 (local) -> global k=4,5
    # C20: bk=0, bi=0 (local) -> global k=4,5, i=0,1
    add_kernel_global(0, 0, 0, 0, 0, 0, col_offset)
    col_offset += 7
    
    # Term 2: A01 B21 -> C20
    # A01: bi=0, bj=1
    # B21: bj=1, bk=0 (local) -> global k=4,5
    # C20: bk=0, bi=0
    add_kernel_global(0, 1, 1, 0, 0, 0, col_offset)
    col_offset += 7
    
    # Term 3: A10 B20 -> C21
    # A10: bi=1, bj=0
    # B20: bj=0, bk=0
    # C21: bk=0, bi=1
    add_kernel_global(1, 0, 0, 0, 0, 1, col_offset)
    col_offset += 7
    
    # Term 4: A11 B21 -> C21
    # A11: bi=1, bj=1
    # B21: bj=1, bk=0
    # C21: bk=0, bi=1
    add_kernel_global(1, 1, 1, 0, 0, 1, col_offset)
    col_offset += 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
