import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Rank 28 construction using two Strassen blocks
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen products for 2x2
    # P1: (X00+X11)(Y00+Y11)
    # P2: (X10+X11)Y00
    # P3: X00(Y10-Y11)
    # P4: X11(Y01-Y00)
    # P5: (X00+X01)Y11
    # P6: (X10-X00)(Y00+Y01)
    # P7: (X01-X11)(Y10+Y11)
    
    # Block 1: A cols 0-1, B rows 0-1
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # Helper to set U, V, W for a term
    # u_coeffs: list of (i, j, coeff)
    # v_coeffs: list of (j, k, coeff)
    # w_coeffs: list of (i, k, coeff)
    
    def add_term(r, u_coeffs, v_coeffs, w_coeffs):
        for (i, j, c) in u_coeffs:
            U[i * m + j][r] += c
        for (j, k, c) in v_coeffs:
            V[j * p + k][r] += c
        for (i, k, c) in w_coeffs:
            W[k * n + i][r] += c
            
    # Block 1 (r = 0..6)
    # P1
    u = [(0,0,1), (1,1,1)]
    v = [(0,0,1), (1,1,1)]
    w = [(0,0,1), (0,1,-1), (1,0,-1), (1,1,1)] # Z00=P1, Z01=-P1? No.
    # Z00 = P1 - P7 + P5 - P4
    # Z01 = P4 + P6
    # Z10 = P2 + P7
    # Z11 = P1 - P3 + P2 - P5
    
    # We need to distribute the W coefficients correctly.
    # For P1, it contributes to Z00 (+1), Z11 (+1).
    # So w_coeffs for P1 should be [(0,0,1), (1,1,1)].
    
    # Let's define the W contributions for each P
    # P1: Z00 (+1), Z11 (+1)
    # P2: Z10 (+1), Z11 (+1)
    # P3: Z11 (-1)
    # P4: Z00 (-1), Z01 (+1)
    # P5: Z00 (+1), Z11 (-1)
    # P6: Z01 (+1)
    # P7: Z00 (-1), Z10 (+1)
    
    # Block 1
    add_term(0, [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]) # P1
    add_term(1, [(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,1)]) # P2
    add_term(2, [(0,0,1)], [(1,0,1), (1,1,-1)], [(1,1,-1)]) # P3
    add_term(3, [(1,1,1)], [(0,1,1), (0,0,-1)], [(0,0,-1), (0,1,1)]) # P4
    add_term(4, [(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,1), (1,1,-1)]) # P5
    add_term(5, [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(0,1,1)]) # P6
    add_term(6, [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,-1), (1,0,1)]) # P7
    
    # Block 2 (r = 7..13)
    # A cols 2-3, B rows 2-3
    # A indices: (0,2), (0,3), (1,2), (1,3)
    # B indices: (2,0), (2,1), (3,0), (3,1)
    # C indices: (0,2), (0,3), (1,2), (1,3)
    
    # P1
    add_term(7, [(0,2,1), (1,3,1)], [(2,0,1), (3,1,1)], [(0,2,1), (1,3,1)])
    # P2
    add_term(8, [(1,2,1), (1,3,1)], [(2,0,1)], [(1,2,1), (1,3,1)])
    # P3
    add_term(9, [(0,2,1)], [(3,0,1), (3,1,-1)], [(1,3,-1)])
    # P4
    add_term(10, [(1,3,1)], [(2,1,1), (2,0,-1)], [(0,2,-1), (0,3,1)])
    # P5
    add_term(11, [(0,2,1), (0,3,1)], [(3,1,1)], [(0,2,1), (1,3,-1)])
    # P6
    add_term(12, [(1,2,1), (0,2,-1)], [(2,0,1), (2,1,1)], [(0,3,1)])
    # P7
    add_term(13, [(0,3,1), (1,3,-1)], [(3,0,1), (3,1,1)], [(0,2,-1), (1,2,1)])
    
    # The remaining ranks 14..27 are zero?
    # No, Strassen uses 7 products per 2x2 block.
    # We have two blocks, so 14 products.
    # But the rank is 28?
    # Wait, 2x2x2 rank is 7.
    # 2x4x4 is composed of two 2x2x4 blocks?
    # No, 2x4x4 = 2x2x4 + 2x2x4.
    # Rank of 2x2x4?
    # 2x2x4 is equivalent to two 2x2x2 multiplications?
    # Yes, if we split the 4 columns into two 2-column blocks.
    # So Rank(2x2x4) = Rank(2x2x2) + Rank(2x2x2) = 7 + 7 = 14.
    # So Rank(2x4x4) = Rank(2x2x4) + Rank(2x2x4) = 14 + 14 = 28.
    
    # So we need 28 terms.
    # My construction above used 14 terms (7 for block 1, 7 for block 2).
    # But each block is 2x2x4.
    # Block 1: A(2x2) * B(2x4) -> C(2x4).
    # I treated B(2x4) as two 2x2 blocks?
    # No, in my add_term calls, I only used B(2x2) (cols 0-1).
    # I missed B cols 2-3 for Block 1?
    # No, Block 1 is A cols 0-1 * B rows 0-1.
    # B rows 0-1 is a 2x4 matrix.
    # I split B rows 0-1 into B[0-1, 0-1] and B[0-1, 2-3].
    # So I need another 7 terms for the second half of B in Block 1.
    # And another 7 terms for the second half of B in Block 2.
    
    # Total 28 terms.
    
    # Block 1, Part 2 (B cols 2-3)
    # A indices same. B indices (0,2), (0,3), (1,2), (1,3). C indices (0,2), (0,3), (1,2), (1,3).
    
    # P1
    add_term(14, [(0,0,1), (1,1,1)], [(0,2,1), (1,3,1)], [(0,2,1), (1,3,1)])
    # P2
    add_term(15, [(1,0,1), (1,1,1)], [(0,2,1)], [(1,2,1), (1,3,1)])
    # P3
    add_term(16, [(0,0,1)], [(1,2,1), (1,3,-1)], [(1,3,-1)])
    # P4
    add_term(17, [(1,1,1)], [(0,3,1), (0,2,-1)], [(0,2,-1), (0,3,1)])
    # P5
    add_term(18, [(0,0,1), (0,1,1)], [(1,3,1)], [(0,2,1), (1,3,-1)])
    # P6
    add_term(19, [(1,0,1), (0,0,-1)], [(0,2,1), (0,3,1)], [(0,3,1)])
    # P7
    add_term(20, [(0,1,1), (1,1,-1)], [(1,2,1), (1,3,1)], [(0,2,-1), (1,2,1)])
    
    # Block 2, Part 2 (B cols 2-3)
    # A indices (0,2), (0,3), (1,2), (1,3). B indices (2,2), (2,3), (3,2), (3,3). C indices (0,2), (0,3), (1,2), (1,3).
    
    # P1
    add_term(21, [(0,2,1), (1,3,1)], [(2,2,1), (3,3,1)], [(0,2,1), (1,3,1)])
    # P2
    add_term(22, [(1,2,1), (1,3,1)], [(2,2,1)], [(1,2,1), (1,3,1)])
    # P3
    add_term(23, [(0,2,1)], [(3,2,1), (3,3,-1)], [(1,3,-1)])
    # P4
    add_term(24, [(1,3,1)], [(2,3,1), (2,2,-1)], [(0,2,-1), (0,3,1)])
    # P5
    add_term(25, [(0,2,1), (0,3,1)], [(3,3,1)], [(0,2,1), (1,3,-1)])
    # P6
    add_term(26, [(1,2,1), (0,2,-1)], [(2,2,1), (2,3,1)], [(0,3,1)])
    # P7
    add_term(27, [(0,3,1), (1,3,-1)], [(3,2,1), (3,3,1)], [(0,2,-1), (1,2,1)])
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
