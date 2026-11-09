import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Rank calculation:
    # Split B into B1 (4x4) and B2 (4x2).
    # C = A*B = [A*B1, A*B2]
    # Rank(A*B1) for 4x4x4x4 is 49 (Strassen).
    # Rank(A*B2) for 4x4x4x2 is 28 (4 Strassen blocks of 2x2x2x2).
    # Total Rank = 49 + 28 = 77.
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen coefficients for 2x2x2x2
    alpha = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], 
        [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    beta = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], 
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    recon = [
        [[0, 0, 1], [1, 1, 1]], [[1, 0, 1], [1, 1, -1]], [[0, 1, 1], [1, 1, 1]],
        [[0, 0, 1], [1, 0, 1]], [[0, 0, -1], [0, 1, 1]], [[1, 1, 1]], [[0, 0, 1]]
    ]
    
    col = 0
    
    # Part 1: A * B1 (4x4 * 4x4)
    # B1 is columns 0-3 of B.
    # Use Strassen on 4x4 blocks.
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B1 blocks: B00, B01, B10, B11 (2x2 each)
    for s in range(7):
        for si in range(7):
            # U: A part
            for a in range(4):
                if alpha[si][a] != 0:
                    ar, br = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar2, br2 = r // 2, r % 2
                            u_idx = (ar2 * 2 + ar) * 4 + (br2 * 2 + br)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            # V: B1 part
            # B1 blocks are 2x2.
            # B1 global indices: rows 0-3, cols 0-3.
            for b in range(4):
                if beta[si][b] != 0:
                    br_row, br_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            # B1[c_block][b_local]
                            # Global row: ac*2 + br_row
                            # Global col: bc*2 + br_col
                            v_idx = (ac * 2 + br_row) * 6 + (bc * 2 + br_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            # W: Output C1 (4x4)
            # C1 is cols 0-3 of C.
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    # za, zb are local indices in 2x2 block.
                    # a, b are block indices.
                    # Global row: a*2 + za
                    # Global col: b*2 + zb
                    w_idx = (b * 2 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
    
    # Part 2: A * B2 (4x4 * 4x2)
    # B2 is columns 4-5 of B.
    # B2 blocks: B20, B21 (2x2 each)
    # A blocks: A00, A01, A10, A11
    # C2 = [A00*B20 + A01*B21, A10*B20 + A11*B21]
    # We need 4 products:
    # M1 = A00 * B20
    # M2 = A01 * B21
    # M3 = A10 * B20
    # M4 = A11 * B21
    # Each is 2x2 * 2x2, rank 7.
    # Total 28.
    
    # Define the 4 terms
    # Term 0: A00 * B20 -> C2[0:2, 0:2] (which is C[0:2, 4:6])
    # Term 1: A01 * B21 -> C2[0:2, 0:2]
    # Term 2: A10 * B20 -> C2[2:4, 0:2] (which is C[2:4, 4:6])
    # Term 3: A11 * B21 -> C2[2:4, 0:2]
    
    terms = [
        (0, 0, 0, 0), # A_block, B_block, C_row_block, C_col_block (local to C2)
        (1, 1, 0, 0),
        (2, 0, 1, 0),
        (3, 1, 1, 0)
    ]
    
    for t in terms:
        a_blk, b_blk, c_r, c_c = t
        for si in range(7):
            # U
            for a in range(4):
                if alpha[si][a] != 0:
                    ar, br = a // 2, a % 2
                    # A is 4x4.
                    # Block a_blk: row a_blk*2, col a_blk%2*2?
                    # A00: (0,0), A01: (0,1), A10: (1,0), A11: (1,1).
                    # a_blk=0 -> (0,0). a_blk=1 -> (0,1). a_blk=2 -> (1,0). a_blk=3 -> (1,1).
                    ar_blk, ac_blk = a_blk // 2, a_blk % 2
                    u_idx = (ar_blk * 2 + ar) * 4 + (ac_blk * 2 + br)
                    U[u_idx, col] += alpha[si][a]
            
            # V
            # B2 blocks: B20 (rows 0-1, cols 4-5), B21 (rows 2-3, cols 4-5).
            # b_blk=0 -> B20. b_blk=1 -> B21.
            br_blk = b_blk # 0 or 1.
            # B20 is rows 0-1. B21 is rows 2-3.
            # Cols are 4-5.
            for b in range(4):
                if beta[si][b] != 0:
                    br, bc = b // 2, b % 2
                    # Global row: br_blk*2 + br
                    # Global col: 4 + bc
                    v_idx = (br_blk * 2 + br) * 6 + (4 + bc)
                    V[v_idx, col] += beta[si][b]
            
            # W
            # Output C2.
            # C2 is 4x2.
            # C_r is block row (0 or 1).
            # c_c is block col (always 0 for C2 local, which is 4-5 global).
            for za, zb, zgamma in recon[si]:
                # za, zb local 2x2 indices.
                # Global row: c_r*2 + za
                # Global col: 4 + zb
                w_idx = (4 + zb) * 4 + (c_r * 2 + za)
                W[w_idx, col] += zgamma
            
            col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
