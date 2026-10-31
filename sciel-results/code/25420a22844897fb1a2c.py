import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 inner factors (for multiplying two 2x2 matrices X, Y)
    # X, Y flattened row-major: [00, 01, 10, 11]
    # 7 multiplications M1..M7
    # u_t: coeffs for X, v_t: coeffs for Y, w_t: coeffs for Z (result)
    # Note: w_t here is the contribution of M_t to the result Z.
    # Z = XY
    # M1 = (X00+X11)(Y00+Y11) -> Z00 += 1, Z11 += 1
    # M2 = (X10+X11)Y00 -> Z10 += 1
    # M3 = X00(Y01-Y11) -> Z01 += 1
    # M4 = X11(Y10-Y00) -> Z00 += 1, Z10 += 1
    # M5 = (X00+X01)Y11 -> Z01 += 1, Z11 -= 1
    # M6 = (X10-X00)(Y00+Y01) -> Z11 += 1
    # M7 = (X01-X11)(Y10+Y11) -> Z00 += 1, Z11 += 1
    
    # Let's define u_t, v_t, w_t for t=0..6
    # u_t, v_t are vectors of length 4
    # w_t is vector of length 4 (coeffs for Z00, Z01, Z10, Z11)
    
    u = [
        [1, 0, 0, 1],  # M1: X00+X11
        [0, 0, 1, 1],  # M2: X10+X11
        [1, 0, 0, 0],  # M3: X00
        [0, 0, 0, 1],  # M4: X11
        [1, 1, 0, 0],  # M5: X00+X01
        [-1, 0, 1, 0], # M6: X10-X00
        [0, 1, 0, -1]  # M7: X01-X11
    ]
    
    v = [
        [1, 0, 0, 1],  # M1: Y00+Y11
        [1, 0, 0, 0],  # M2: Y00
        [0, 1, 0, -1], # M3: Y01-Y11
        [0, -1, 1, 0], # M4: Y10-Y00
        [0, 0, 0, 1],  # M5: Y11
        [1, 1, 0, 0],  # M6: Y00+Y01
        [0, 0, 1, 1]   # M7: Y10+Y11
    ]
    
    w = [
        [1, 0, 0, 1],  # M1 -> Z00+1, Z11+1
        [0, 0, 1, 0],  # M2 -> Z10+1
        [0, 1, 0, 0],  # M3 -> Z01+1
        [1, 0, 1, 0],  # M4 -> Z00+1, Z10+1
        [0, 1, 0, -1], # M5 -> Z01+1, Z11-1
        [0, 0, 0, 1],  # M6 -> Z11+1
        [1, 0, 0, 1]   # M7 -> Z00+1, Z11+1
    ]
    
    # Outer Strassen combinations for 4x4 blocks
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # M_s = L_s * R_s
    # L_s = sum l_{s,b} A_b
    # R_s = sum r_{s,b} B_b
    
    # s=0 (M1): (A00+A11)(B00+B11)
    # s=1 (M2): (A10+A11)B00
    # s=2 (M3): A00(B01-B11)
    # s=3 (M4): A11(B10-B00)
    # s=4 (M5): (A00+A01)B11
    # s=5 (M6): (A10-A00)(B00+B01)
    # s=6 (M7): (A01-A11)(B10+B11)
    
    # l_s, r_s are vectors of length 4 (coeffs for A00, A01, A10, A11)
    l = [
        [1, 0, 0, 1],  # s=0
        [0, 0, 1, 1],  # s=1
        [1, 0, 0, 0],  # s=2
        [0, 0, 0, 1],  # s=3
        [1, 1, 0, 0],  # s=4
        [-1, 0, 1, 0], # s=5
        [0, 1, 0, -1]  # s=6
    ]
    
    r = [
        [1, 0, 0, 1],  # s=0
        [1, 0, 0, 0],  # s=1
        [0, 1, 0, -1], # s=2
        [0, -1, 1, 0], # s=3
        [0, 0, 0, 1],  # s=4
        [1, 1, 0, 0],  # s=5
        [0, 0, 1, 1]   # s=6
    ]
    
    # w_out: contribution of M_s to C blocks
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # w_out[s][b] where b in 0..3 corresponds to C00, C01, C10, C11
    w_out = [
        [1, 0, 0, 0],  # s=0 (M1) -> C00+1, C11-1? Wait.
        # Let's re-verify C formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # s=0 (M1): C00+1, C11+1
        # s=1 (M2): C10+1, C11-1
        # s=2 (M3): C01+1, C11+1
        # s=3 (M4): C00+1, C10+1
        # s=4 (M5): C01+1, C00-1
        # s=5 (M6): C11+1
        # s=6 (M7): C00+1
        
        [1, 0, 0, 1],  # s=0
        [0, 0, 1, -1], # s=1
        [0, 1, 0, 1],  # s=2
        [1, 0, 1, 0],  # s=3
        [-1, 1, 0, 0], # s=4
        [0, 0, 0, 1],  # s=5
        [1, 0, 0, 0]   # s=6
    ]
    
    # Offsets for blocks
    # A blocks: (i_off, j_off)
    a_offsets = [(0, 0), (0, 2), (2, 0), (2, 2)]
    b_offsets = [(0, 0), (0, 2), (2, 0), (2, 2)]
    c_offsets = [(0, 0), (0, 2), (2, 0), (2, 2)]
    
    col = 0
    for s in range(7):
        for t in range(7):
            # Fill U
            for b in range(4):
                if l[s][b] == 0: continue
                i_off, j_off = a_offsets[b]
                for e in range(4):
                    if u[t][e] == 0: continue
                    coeff = l[s][b] * u[t][e]
                    i_l = e // 2
                    j_l = e % 2
                    i = i_off + i_l
                    j = j_off + j_l
                    idx_u = i * 4 + j
                    U[idx_u][col] += coeff
                    
            # Fill V
            for b in range(4):
                if r[s][b] == 0: continue
                i_off, j_off = b_offsets[b]
                for e in range(4):
                    if v[t][e] == 0: continue
                    coeff = r[s][b] * v[t][e]
                    i_l = e // 2
                    j_l = e % 2
                    i = i_off + i_l
                    j = j_off + j_l
                    idx_v = i * 4 + j
                    V[idx_v][col] += coeff
                    
            # Fill W
            for b_c in range(4):
                if w_out[s][b_c] == 0: continue
                i_off, k_off = c_offsets[b_c]
                for e in range(4):
                    if w[t][e] == 0: continue
                    coeff = w_out[s][b_c] * w[t][e]
                    i_l = e // 2
                    k_l = e % 2
                    i = i_off + i_l
                    k = k_off + k_l
                    idx_w = k * 4 + i
                    W[idx_w][col] += coeff
                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
