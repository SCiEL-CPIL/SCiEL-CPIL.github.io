import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strategy: Use Strassen's algorithm structure.
    # Split the 2x4 * 4x8 multiplication into 8 independent 2x2 * 2x2 blocks.
    # Each 2x2 * 2x2 can be done with rank 7 (Strassen).
    # Total rank = 8 * 7 = 56, vs schoolbook 64.
    
    # A is 2x4. Split columns: A = [A0 | A1] where A0 is cols 0-1 (2x2), A1 is cols 2-3 (2x2).
    # B is 4x8. Split rows: B = [B0; B1] where B0 is rows 0-1 (2x8), B1 is rows 2-3 (2x8).
    # C = A * B = A0 * B0 + A1 * B1.
    # A0 * B0: A0 is 2x2, B0 is 2x8. Split B0 into 4 blocks of 2x2: B0 = [B00 | B01 | B02 | B03].
    # A1 * B1: A1 is 2x2, B1 is 2x8. Split B1 into 4 blocks: B1 = [B10 | B11 | B12 | B13].
    # So we have 8 sub-multiplications: A0*B0b for b=0..3, and A1*B1b for b=0..3.
    
    # Strassen's 7 products for X (2x2) * Y (2x2) = Z (2x2):
    # M1 = (x11 + x22) * (y11 + y22)
    # M2 = (x21 + x22) * y11
    # M3 = x11 * (y12 - y22)
    # M4 = x22 * (y21 - y11)
    # M5 = (x11 + x12) * y22
    # M6 = (x21 - x11) * (y11 + y12)
    # M7 = (x12 - x22) * (y21 + y22)
    
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # X forms (coefficients for x11, x12, x21, x22):
    # M1: 1, 0, 0, 1
    # M2: 0, 0, 1, 1
    # M3: 1, 0, 0, 0
    # M4: 0, 0, 0, 1
    # M5: 1, 1, 0, 0
    # M6: -1, 0, 1, 0
    # M7: 0, 1, 0, -1
    X_forms = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    
    # Y forms (coefficients for y11, y12, y21, y22):
    # M1: 1, 0, 0, 1
    # M2: 1, 0, 0, 0
    # M3: 0, 1, 0, -1
    # M4: -1, 0, 1, 0
    # M5: 0, 0, 0, 1
    # M6: 1, 1, 0, 0
    # M7: 0, 0, 1, 1
    Y_forms = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    
    # Z combinations (coefficients for M1..M7):
    # Z11: 1, 0, 0, 1, -1, 0, 1  (M1 + M4 - M5 + M7)
    # Z12: 0, 0, 1, 0, 1, 0, 0   (M3 + M5)
    # Z21: 0, 1, 0, 1, 0, 0, 0   (M2 + M4)
    # Z22: 1, -1, 1, 0, 0, 1, 0  (M1 - M2 + M3 + M6)
    Z_combos = [
        [1, 0, 0, 1, -1, 0, 1],   # Z11
        [0, 0, 1, 0, 1, 0, 0],    # Z12
        [0, 1, 0, 1, 0, 0, 0],    # Z21
        [1, -1, 1, 0, 0, 1, 0],   # Z22
    ]
    
    num_blocks = 8  # 2 (A blocks) * 4 (B column blocks)
    num_strassen = 7
    rank = num_blocks * num_strassen  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    r = 0
    # Iterate over A blocks (b_a = 0, 1) and B column blocks (b_c = 0, 1, 2, 3)
    for b_a in range(2):
        for b_c in range(4):
            # The B row block is determined by b_a (since A0 multiplies B0, A1 multiplies B1)
            b_b = b_a
            
            for s in range(num_strassen):
                # U: indices for A[i, j] where j is in the b_a block
                # A is 2x4. j = b_a * 2 + jp, where jp in {0, 1}
                # i in {0, 1}
                for i in range(n):
                    for jp in range(2):
                        j = b_a * 2 + jp
                        a_idx = i * m + j
                        U[a_idx][r] = X_forms[s][i * 2 + jp]
                
                # V: indices for B[j, k] where j is in the b_b block, k is in the b_c block
                # B is 4x8. j = b_b * 2 + jp, k = b_c * 2 + kp
                # jp in {0, 1}, kp in {0, 1}
                for jp in range(2):
                    for kp in range(2):
                        j = b_b * 2 + jp
                        k = b_c * 2 + kp
                        b_idx = j * p + k
                        V[b_idx][r] = Y_forms[s][jp * 2 + kp]
                
                # W: indices for C[i, k] where i in {0, 1}, k is in the b_c block
                # C is 2x8. k = b_c * 2 + kp
                # W index is k * n + i
                for i in range(n):
                    for kp in range(2):
                        k = b_c * 2 + kp
                        c_idx = k * n + i
                        # Z index is i * 2 + kp (since Z is 2x2, i is row, kp is col)
                        W[c_idx][r] = Z_combos[i * 2 + kp][s]
                
                r += 1
    
    return {"U": U, "V": V, "W": W}
