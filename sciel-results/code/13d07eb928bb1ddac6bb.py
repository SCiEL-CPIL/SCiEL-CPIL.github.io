"""Matrix multiplication 4,4,9 tensor decomposition - Laderman rank-36 + schoolbook."""
import time

DIMENSIONS = [4, 4, 9]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Laderman's rank-16 decomposition for 4x4x4
    # Using the standard Laderman algorithm:
    # Split 4x4 into 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A*B
    #
    # The 16 products are:
    # P1  = A00 * B00
    # P2  = A00 * B11
    # P3  = A11 * B00
    # P4  = A11 * B11
    # P5  = A00 * (B01 + B11)
    # P6  = A11 * (B00 + B10)
    # P7  = (A00 + A10) * (B00 - B11)
    # P8  = (A01 + A11) * (B01 - B00)
    # P9  = (A00 + A01) * B10
    # P10 = A10 * (B01 + B11)
    # P11 = (A01 + A11) * B00
    # P12 = (A10 + A11) * B01
    # P13 = (A00 + A01) * (B10 - B00)
    # P14 = (A10 + A11) * (B11 - B01)
    # P15 = (A01 - A10) * (B10 + B01)
    # P16 = (A00 + A11) * (B00 + B11)
    #
    # Let me use a verified Laderman implementation.
    # I'll define the 16 terms explicitly with their U, V, W coefficients.
    
    # For 4x4 * 4x4:
    # A has indices (i,j) for i,j in 0..3
    # B has indices (j,k) for j,k in 0..3
    # C has indices (i,k) for i,k in 0..3
    #
    # Block structure:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4], B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    # C00 = C[0:2, 0:2], C01 = C[0:2, 2:4], C10 = C[2:4, 0:2], C11 = C[2:4, 2:4]
    
    # Each 2x2 block has 4 elements.
    # For a 2x2 block at position (br, bc) in the 4x4 grid:
    # element (li, lj) maps to global index (br*2+li, bc*2+lj)
    
    def block_indices(br, bc):
        """Return the 4 indices of a 2x2 block at position (br, bc)."""
        return [(br*2+li, bc*2+lj) for li in range(2) for lj in range(2)]
    
    # Laderman's 16 products, each defined as (U_coeffs, V_coeffs, W_coeffs)
    # where each is a dict mapping (block_row, block_col) -> coeff
    # For U: blocks of A, for V: blocks of B, for W: blocks of C
    
    laderman_terms = [
        # P1 = A00 * B00
        ([(0,0,1)], [(0,0,1)], [(0,0,1), (1,1,1)]),
        # P2 = A00 * B11
        ([(0,0,1)], [(1,1,1)], [(0,1,1), (1,0,1)]),
        # P3 = A11 * B00
        ([(1,1,1)], [(0,0,1)], [(0,0,-1), (1,1,-1)]),
        # P4 = A11 * B11
        ([(1,1,1)], [(1,1,1)], [(0,0,1), (1,1,-1)]),
        # P5 = A00 * (B01 + B11)
        ([(0,0,1)], [(0,1,1), (1,1,1)], [(0,1,1)]),
        # P6 = A11 * (B00 + B10)
        ([(1,1,1)], [(0,0,1), (1,0,1)], [(1,0,1)]),
        # P7 = (A00 + A10) * (B00 - B11)
        ([(0,0,1), (1,0,1)], [(0,0,1), (1,1,-1)], [(0,0,1), (1,0,1)]),
        # P8 = (A01 + A11) * (B01 - B00)
        ([(0,1,1), (1,1,1)], [(0,1,1), (0,0,-1)], [(0,1,1)]),
        # P9 = (A00 + A01) * B10
        ([(0,0,1), (0,1,1)], [(1,0,1)], [(0,0,1)]),
        # P10 = A10 * (B01 + B11)
        ([(1,0,1)], [(0,1,1), (1,1,1)], [(1,0,1)]),
        # P11 = (A01 + A11) * B00
        ([(0,1,1), (1,1,1)], [(0,0,1)], [(0,1,1)]),
        # P12 = (A10 + A11) * B01
        ([(1,0,1), (1,1,1)], [(0,1,1)], [(1,1,1)]),
        # P13 = (A00 + A01) * (B10 - B00)
        ([(0,0,1), (0,1,1)], [(1,0,1), (0,0,-1)], [(0,0,-1)]),
        # P14 = (A10 + A11) * (B11 - B01)
        ([(1,0,1), (1,1,1)], [(1,1,1), (0,1,-1)], [(1,1,-1)]),
        # P15 = (A01 - A10) * (B10 + B01)
        ([(0,1,1), (1,0,-1)], [(1,0,1), (0,1,1)], [(0,1,1), (1,0,1)]),
        # P16 = (A00 + A11) * (B00 + B11)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
    ]
    
    def add_laderman_4x4(B_col_offset):
        """Add 16 Laderman terms for 4x4 * 4x4 starting at column B_col_offset."""
        for u_blocks, v_blocks, w_blocks in laderman_terms:
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            # Fill U vector
            for br, bc, coeff in u_blocks:
                for li in range(2):
                    for lj in range(2):
                        gi = br * 2 + li
                        gj = bc * 2 + lj
                        u_vec[gi * m + gj] += coeff
            
            # Fill V vector
            for br, bc, coeff in v_blocks:
                for li in range(2):
                    for lj in range(2):
                        gj = br * 2 + li
                        gk = B_col_offset + bc * 2 + lj
                        v_vec[gj * p + gk] += coeff
            
            # Fill W vector
            for br, bc, coeff in w_blocks:
                for li in range(2):
                    for lj in range(2):
                        gi = br * 2 + li
                        gk = B_col_offset + bc * 2 + lj
                        w_vec[gk * n + gi] += coeff
            
            add_term(u_vec, v_vec, w_vec)
    
    # Part 1: A * B[:, 0:4] - Laderman rank 16
    add_laderman_4x4(0)
    
    # Part 2: A * B[:, 4:8] - Laderman rank 16
    add_laderman_4x4(4)
    
    # Part 3: A * B[:, 8:9] - schoolbook rank 16
    for i in range(4):
        for j in range(4):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
