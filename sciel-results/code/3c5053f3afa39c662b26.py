"""Matrix multiplication 2,4,8 - Rank 56 Strassen-based decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen forms for 2x2 matrix multiplication
    # X_forms[s][i*2+j'] for s=0..6, i=0,1, j'=0,1
    # Indices: 00, 01, 10, 11
    X_forms = [
        [1, 0, 0, 1],   # M0: (X00 + X11)
        [0, 0, 1, 1],   # M1: (X10 + X11)
        [1, 0, 0, 0],   # M2: X00
        [0, 0, 0, 1],   # M3: X11
        [1, 1, 0, 0],   # M4: (X00 + X01)
        [-1, 0, 1, 0],  # M5: (X10 - X00)
        [0, 1, 0, -1],  # M6: (X01 - X11)
    ]
    
    # Y_forms[s][j'*2+k'] for s=0..6, j'=0,1, k'=0,1
    # Indices: 00, 01, 10, 11
    Y_forms = [
        [1, 0, 0, 1],   # M0: (Y00 + Y11)
        [1, 0, 0, 0],   # M1: Y00
        [0, 1, 0, -1],  # M2: (Y01 - Y11)
        [-1, 0, 1, 0],  # M3: (Y10 - Y00)
        [0, 0, 0, 1],   # M4: Y11
        [1, 1, 0, 0],   # M5: (Y00 + Y01)
        [0, 0, 1, 1],   # M6: (Y10 + Y11)
    ]
    
    # Z_combos[z_idx][s] for z_idx=0..3 (z00, z01, z10, z11), s=0..6
    # z00 = M0 + M3 - M4 + M6
    # z01 = M2 + M4
    # z10 = M1 + M3
    # z11 = M0 - M1 + M2 + M5
    Z_combos = [
        [1, 0, 0, 1, -1, 0, 1],   # z00
        [0, 0, 1, 0, 1, 0, 0],    # z01
        [0, 1, 0, 1, 0, 0, 0],    # z10
        [1, -1, 1, 0, 0, 1, 0],   # z11
    ]
    
    # Structure:
    # A is 2x4. Split into two 2x2 blocks horizontally: A = [A_0 | A_1]
    # B is 4x8. Split into two 2x8 blocks vertically: B = [B_0; B_1]
    # C = A_0 B_0 + A_1 B_1
    # B_0 and B_1 are 2x8. Split each into four 2x2 blocks horizontally.
    # B_0 = [B_{0,0} | B_{0,1} | B_{0,2} | B_{0,3}]
    # B_1 = [B_{1,0} | B_{1,1} | B_{1,2} | B_{1,3}]
    # Total 8 sub-multiplications of 2x2 * 2x2.
    # Each uses rank 7. Total rank = 8 * 7 = 56.
    
    num_groups = 4  # g = 0,1,2,3 (blocks of B columns)
    num_ablocks = 2  # b = 0,1 (blocks of A columns / B rows)
    num_strassen = 7  # s = 0..6
    rank = num_groups * num_ablocks * num_strassen  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    r = 0
    for g in range(num_groups):
        for b in range(num_ablocks):
            for s in range(num_strassen):
                # U: for i in {0,1}, j' in {0,1}
                # A index: i * m + (b * 2 + j')
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + (b * 2 + jp)
                        U[a_idx][r] = X_forms[s][i * 2 + jp]
                
                # V: for j' in {0,1}, k' in {0,1}
                # B index: (b * 2 + j') * p + (2 * g + k')
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * p + (2 * g + kp)
                        V[b_idx][r] = Y_forms[s][jp * 2 + kp]
                
                # W: for i in {0,1}, k' in {0,1}
                # C index: (2 * g + k') * n + i
                for i in range(n):
                    for kp in range(2):
                        k = 2 * g + kp
                        c_idx = k * n + i
                        W[c_idx][r] = Z_combos[i * 2 + kp][s]
                
                r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
