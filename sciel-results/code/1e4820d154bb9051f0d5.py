import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Block size for 2x2 block decomposition
    bs = 3  # Each block is 3x3
    
    # Strassen's algorithm for 2x2 blocks of size 3x3
    # 7 multiplications, each of rank 27 (schoolbook 3x3)
    # Total rank = 7 * 27 = 189
    
    rank = 7 * 27
    
    U = [[0.0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0.0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0.0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Helper: get block index range
    # For a 6x6 matrix split into 2x2 blocks of 3x3:
    # Block (bi, bj) where bi, bj in {0,1}
    # Row indices: bi*bs to (bi+1)*bs - 1
    # Col indices: bj*bs to (bj+1)*bs - 1
    
    def block_rows(bi):
        return list(range(bi * bs, (bi + 1) * bs))
    
    def block_cols(bj):
        return list(range(bj * bs, (bj + 1) * bs))
    
    # For each of the 7 Strassen multiplications, we have a 3x3 schoolbook
    # The 7 Strassen terms:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For each multiplication s (0-6), and each scalar multiplication in the 3x3 block (27 total)
    # We need to set up U, V, W contributions
    
    # Let's define the 7 multiplications with their A-block and B-block coefficients
    # Each is a list of (block_row, block_col, coeff) for A and B
    
    strassen_mults = [
        # M1: (A11 + A22)(B11 + B22)
        {'A': [(0,0,1.0), (1,1,1.0)], 'B': [(0,0,1.0), (1,1,1.0)]},
        # M2: (A21 + A22)B11
        {'A': [(1,0,1.0), (1,1,1.0)], 'B': [(0,0,1.0)]},
        # M3: A11(B12 - B22)
        {'A': [(0,0,1.0)], 'B': [(0,1,1.0), (1,1,-1.0)]},
        # M4: A22(B21 - B11)
        {'A': [(1,1,1.0)], 'B': [(1,0,1.0), (0,0,-1.0)]},
        # M5: (A11 + A12)B22
        {'A': [(0,0,1.0), (0,1,1.0)], 'B': [(1,1,1.0)]},
        # M6: (A21 - A11)(B11 + B12)
        {'A': [(1,0,1.0), (0,0,-1.0)], 'B': [(0,0,1.0), (0,1,1.0)]},
        # M7: (A12 - A22)(B21 + B22)
        {'A': [(0,1,1.0), (1,1,-1.0)], 'B': [(1,0,1.0), (1,1,1.0)]},
    ]
    
    # For each Strassen multiplication, the result M_s is a 3x3 matrix
    # M_s = A_s * B_s where A_s and B_s are 3x3 matrices formed by linear combinations
    # The 3x3 multiplication is schoolbook: M_s[i,j] = sum_k A_s[i,k] * B_s[k,j]
    
    # The final C blocks are linear combinations of M_s:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # C block to M_s coefficients:
    # C[bi, bj] = sum_s coeff[bi,bj,s] * M_s
    c_block_coeffs = {
        (0,0): [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],   # C11 = M1 + M4 - M5 + M7
        (0,1): [(2, 1.0), (4, 1.0)],                          # C12 = M3 + M5
        (1,0): [(1, 1.0), (3, 1.0)],                          # C21 = M2 + M4
        (1,1): [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],    # C22 = M1 - M2 + M3 + M6
    }
    
    # Now build U, V, W
    # For each Strassen multiplication s (0-6):
    #   For each scalar multiplication (i, k, j) in the 3x3 block (i,k,j in 0-2):
    #     This is one rank-1 term
    #     U contribution: For each A-block (ai, aj) with coeff ca in strassen_mults[s]['A']:
    #       U[ai*bs + i, aj*bs + k][r] += ca
    #     V contribution: For each B-block (bk, bj) with coeff cb in strassen_mults[s]['B']:
    #       V[bk*bs + k, bj*bs + j][r] += cb
    #     W contribution: For each C-block (ci, cj) with coeff cc in c_block_coeffs[(ci,cj)]:
    #       If s is in the list for (ci,cj) with coeff cs:
    #         W[ci*bs + i, cj*bs + j][r] += cc * cs
    
    r = 0
    for s in range(7):
        sm = strassen_mults[s]
        for i in range(bs):  # row in 3x3 block
            for k in range(bs):  # inner index in 3x3 block
                for j in range(bs):  # col in 3x3 block
                    # U: A_s[i,k] = sum over A-blocks
                    for (ai, aj, ca) in sm['A']:
                        u_row = ai * bs + i
                        u_col = aj * bs + k
                        U[u_row][r] += ca
                    
                    # V: B_s[k,j] = sum over B-blocks
                    for (bk, bj, cb) in sm['B']:
                        v_row = bk * bs + k
                        v_col = bj * bs + j
                        V[v_row][r] += cb
                    
                    # W: C contribution from M_s
                    # For each C-block (ci, cj), check if s contributes
                    for (ci, cj) in [(0,0), (0,1), (1,0), (1,1)]:
                        coeffs = c_block_coeffs[(ci, cj)]
                        for (s_idx, cs) in coeffs:
                            if s_idx == s:
                                w_row = ci * bs + i
                                w_col = cj * bs + j
                                W[w_row][r] += cs
                        # Note: cs is already the coefficient for this specific s in the C-block formula
                    
                    r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
