import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # u, v, w are lists of 7 tuples (or vectors)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1: (A00+A11)(B00+B11) -> C00+1, C11+1
    # M2: (A10+A11)B00 -> C10+1
    # M3: A00(B01-B11) -> C01+1
    # M4: A11(B10-B00) -> C00+1, C10+1
    # M5: (A00+A01)B11 -> C00-1, C01+1
    # M6: (A10-A00)(B00+B01) -> C11+1
    # M7: (A01-A11)(B10+B11) -> C00+1, C11+1
    
    # We define u, v, w for each of the 7 steps
    # u: 4 values (A00, A01, A10, A11)
    # v: 4 values (B00, B01, B10, B11)
    # w: 4 values (C00, C01, C10, C11)
    
    strassen_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    strassen_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_w = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 0], # M2
        [0, 1, 0, 0], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 1]  # M7
    ]
    
    # Check:
    # C00 = M1 + M4 - M5 + M7 = 1 + 1 - (-1) + 1 = 4? No.
    # C00 = M1 + M4 - M5 + M7.
    # M1: 1. M4: 1. M5: -1. M7: 1. Sum = 1+1-(-1)+1 = 4.
    # Wait, C00 should be A00B00 + A01B10.
    # Let's re-verify Strassen formulas.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # My w vectors:
    # M1: C00=1, C11=1.
    # M2: C10=1.
    # M3: C01=1.
    # M4: C00=1, C10=1.
    # M5: C00=-1, C01=1.
    # M6: C11=1.
    # M7: C00=1, C11=1.
    
    # C00 sum: M1(1) + M4(1) + M5(-1) + M7(1) = 2.
    # This is wrong. C00 should be 1.
    # Ah, the coefficients in the sum are 1.
    # The formula is C = sum M_c * W_c.
    # If W_c has entries, they are the coefficients.
    # For M5, the term is -M5 in C00. So W5[0] should be -1.
    # In my strassen_w, M5 is [-1, 1, 0, 0].
    # So C00 gets -1 from M5.
    # Sum: 1 (M1) + 1 (M4) - 1 (M5) + 1 (M7) = 2.
    # Why 2?
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # M4 = A11(B10-B00) = A11B10 - A11B00.
    # M5 = (A00+A01)B11 = A00B11 + A01B11.
    # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11.
    
    # C00 = M1 + M4 - M5 + M7
    # = (A00B00 + A00B11 + A11B00 + A11B11) + (A11B10 - A11B00) - (A00B11 + A01B11) + (A01B10 + A01B11 - A11B10 - A11B11)
    # A00B00: 1
    # A00B11: 1 - 1 = 0
    # A11B00: 1 - 1 = 0
    # A11B11: 1 - 1 = 0
    # A11B10: 1 - 1 = 0
    # A01B11: -1 + 1 = 0
    # A01B10: 1
    # Result: A00B00 + A01B10. Correct.
    
    # So the w vectors are correct.
    
    col = 0
    for t in range(4): # k block 0..3
        for j_half in range(2): # j block 0..1
            # A indices: i in 0..1, j in {2*j_half, 2*j_half+1}
            # B indices: j in {2*j_half, 2*j_half+1}, k in {2*t, 2*t+1}
            # C indices: i in 0..1, k in {2*t, 2*t+1}
            
            for s in range(7):
                # u
                for li in range(2):
                    for lj in range(2):
                        # local j index lj maps to global j = 2*j_half + lj
                        # local i index li maps to global i = li
                        u_val = strassen_u[s][li*2 + lj]
                        if u_val != 0:
                            global_i = li
                            global_j = 2 * j_half + lj
                            U[global_i * 4 + global_j][col] += u_val
                
                # v
                for lj in range(2):
                    for lk in range(2):
                        # local j index lj maps to global j = 2*j_half + lj
                        # local k index lk maps to global k = 2*t + lk
                        v_val = strassen_v[s][lj*2 + lk]
                        if v_val != 0:
                            global_j = 2 * j_half + lj
                            global_k = 2 * t + lk
                            V[global_j * 8 + global_k][col] += v_val
                
                # w
                for lk in range(2):
                    for li in range(2):
                        # local k index lk maps to global k = 2*t + lk
                        # local i index li maps to global i = li
                        w_val = strassen_w[s][lk*2 + li]
                        if w_val != 0:
                            global_k = 2 * t + lk
                            global_i = li
                            W[global_k * 2 + global_i][col] += w_val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
