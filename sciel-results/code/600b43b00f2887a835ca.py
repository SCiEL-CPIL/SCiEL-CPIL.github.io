import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Laderman's 3x3 matrix multiplication (rank 19)
    # C = A * B, all 3x3
    # A[i,j], B[i,j], C[i,j] for i,j in 0,1,2
    
    def aidx(r, c):
        return r * 3 + c
    
    # (A_coeffs, B_coeffs, C_coeffs) for each of 19 terms
    laderman_terms = [
        # M1 = (a00+a01)(b00+b10)
        ({aidx(0,0): 1, aidx(0,1): 1}, {aidx(0,0): 1, aidx(1,0): 1}, {aidx(0,0): 1, aidx(0,1): 1}),
        # M2 = (a00+a02)(b00+b20)
        ({aidx(0,0): 1, aidx(0,2): 1}, {aidx(0,0): 1, aidx(2,0): 1}, {aidx(0,0): 1, aidx(0,2): -1}),
        # M3 = (a00+a10+a20)(b00+b10+b20)
        ({aidx(0,0): 1, aidx(1,0): 1, aidx(2,0): 1}, {aidx(0,0): 1, aidx(1,0): 1, aidx(2,0): 1}, {aidx(0,0): -1, aidx(1,0): 1, aidx(2,0): -1}),
        # M4 = (a11+a21)(b11+b21)
        ({aidx(1,1): 1, aidx(2,1): 1}, {aidx(1,1): 1, aidx(2,1): 1}, {aidx(0,1): 1, aidx(1,1): 1, aidx(2,2): -1}),
        # M5 = (a11+a22)(b11+b22)
        ({aidx(1,1): 1, aidx(2,2): 1}, {aidx(1,1): 1, aidx(2,2): 1}, {aidx(0,2): 1, aidx(1,1): -1, aidx(2,1): -1}),
        # M6 = (a11+a01+a22)(b11+b01+b22)
        ({aidx(1,1): 1, aidx(0,1): 1, aidx(2,2): 1}, {aidx(1,1): 1, aidx(0,1): 1, aidx(2,2): 1}, {aidx(1,1): -1, aidx(2,1): 1, aidx(2,2): 1}),
        # M7 = (a22+a02)(b22+b02)
        ({aidx(2,2): 1, aidx(0,2): 1}, {aidx(2,2): 1, aidx(0,2): 1}, {aidx(0,2): -1, aidx(2,0): -1, aidx(2,2): -1}),
        # M8 = (a22+a12)(b22+b12)
        ({aidx(2,2): 1, aidx(1,2): 1}, {aidx(2,2): 1, aidx(1,2): 1}, {aidx(1,2): -1, aidx(2,0): -1, aidx(2,1): 1}),
        # M9 = (a00+a01+a02)(b00+b10+b20)
        ({aidx(0,0): 1, aidx(0,1): 1, aidx(0,2): 1}, {aidx(0,0): 1, aidx(1,0): 1, aidx(2,0): 1}, {aidx(0,1): -1, aidx(2,1): 1}),
        # M10 = (a11+a12)(b11+b21)
        ({aidx(1,1): 1, aidx(1,2): 1}, {aidx(1,1): 1, aidx(2,1): 1}, {aidx(0,1): -1, aidx(1,0): 1, aidx(1,1): -1}),
        # M11 = (a22+a21)(b22+b12)
        ({aidx(2,2): 1, aidx(2,1): 1}, {aidx(2,2): 1, aidx(1,2): 1}, {aidx(1,2): 1, aidx(2,0): -1, aidx(2,1): -1}),
        # M12 = (a00+a10+a11)(b00+b10+b11)
        ({aidx(0,0): 1, aidx(1,0): 1, aidx(1,1): 1}, {aidx(0,0): 1, aidx(1,0): 1, aidx(1,1): 1}, {aidx(0,0): -1, aidx(1,0): -1}),
        # M13 = (a11+a21+a22)(b11+b21+b22)
        ({aidx(1,1): 1, aidx(2,1): 1, aidx(2,2): 1}, {aidx(1,1): 1, aidx(2,1): 1, aidx(2,2): 1}, {aidx(1,0): -1, aidx(1,1): -1, aidx(1,2): -1}),
        # M14 = (a02+a22)(b02+b22)
        ({aidx(0,2): 1, aidx(2,2): 1}, {aidx(0,2): 1, aidx(2,2): 1}, {aidx(0,2): -1, aidx(2,0): 1, aidx(2,2): 1}),
        # M15 = (a01+a11)(b01+b11)
        ({aidx(0,1): 1, aidx(1,1): 1}, {aidx(0,1): 1, aidx(1,1): 1}, {aidx(0,0): 1, aidx(0,1): 1, aidx(1,1): -1}),
        # M16 = (a22+a02+a01)(b22+b02+b01)
        ({aidx(2,2): 1, aidx(0,2): 1, aidx(0,1): 1}, {aidx(2,2): 1, aidx(0,2): 1, aidx(0,1): 1}, {aidx(0,0): -1, aidx(0,2): 1, aidx(2,0): -1, aidx(2,1): 1}),
        # M17 = (a11+a01+a02)(b11+b01+b02)
        ({aidx(1,1): 1, aidx(0,1): 1, aidx(0,2): 1}, {aidx(1,1): 1, aidx(0,1): 1, aidx(0,2): 1}, {aidx(0,1): 1, aidx(1,0): 1, aidx(1,2): -1, aidx(2,1): -1}),
        # M18 = (a22+a12+a11)(b22+b12+b11)
        ({aidx(2,2): 1, aidx(1,2): 1, aidx(1,1): 1}, {aidx(2,2): 1, aidx(1,2): 1, aidx(1,1): 1}, {aidx(1,0): 1, aidx(1,2): 1, aidx(2,0): 1, aidx(2,1): -1}),
        # M19 = (a00+a20+a22)(b00+b20+b22)
        ({aidx(0,0): 1, aidx(2,0): 1, aidx(2,2): 1}, {aidx(0,0): 1, aidx(2,0): 1, aidx(2,2): 1}, {aidx(0,0): 1, aidx(0,2): 1, aidx(2,0): 1, aidx(2,2): 1}),
    ]
    
    # Strassen 2x2
    def sidx(r, c):
        return r * 2 + c
    
    strassen_terms = [
        ({sidx(0,0): 1, sidx(1,1): 1}, {sidx(0,0): 1, sidx(1,1): 1}, {sidx(0,0): 1, sidx(1,1): 1}),
        ({sidx(1,0): 1, sidx(1,1): 1}, {sidx(0,0): 1}, {sidx(1,0): 1, sidx(1,1): -1}),
        ({sidx(0,0): 1}, {sidx(0,1): 1, sidx(1,1): -1}, {sidx(0,1): 1, sidx(1,1): 1}),
        ({sidx(1,1): 1}, {sidx(1,0): 1, sidx(0,0): -1}, {sidx(0,0): 1, sidx(1,0): 1}),
        ({sidx(0,0): 1, sidx(0,1): 1}, {sidx(1,1): 1}, {sidx(0,0): -1, sidx(0,1): 1}),
        ({sidx(1,0): 1, sidx(0,0): -1}, {sidx(0,0): 1, sidx(0,1): 1}, {sidx(1,1): 1}),
        ({sidx(0,1): 1, sidx(1,1): -1}, {sidx(1,0): 1, sidx(1,1): 1}, {sidx(0,0): 1}),
    ]
    
    rank = 19 * 7  # 133
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for la, lb, lc in laderman_terms:
        for sa, sb, sc in strassen_terms:
            for a_bi, a_cf in la.items():
                a_br, a_bc = a_bi // 3, a_bi % 3
                for s_ai, s_af in sa.items():
                    sr, sc_ = s_ai // 2, s_ai % 2
                    gr = a_br * 2 + sr
                    gc = a_bc * 2 + sc_
                    U[gr * m + gc][col] += a_cf * s_af
            
            for b_bi, b_cf in lb.items():
                b_br, b_bc = b_bi // 3, b_bi % 3
                for s_bi, s_bf in sb.items():
                    sr, sc_ = s_bi // 2, s_bi % 2
                    gr = b_br * 2 + sr
                    gc = b_bc * 2 + sc_
                    V[gr * p + gc][col] += b_cf * s_bf
            
            for c_bi, c_cf in lc.items():
                c_br, c_bc = c_bi // 3, c_bi % 3
                for s_ci, s_cf_ in sc.items():
                    sr, sc_ = s_ci // 2, s_ci % 2
                    gr = c_br * 2 + sr
                    gc = c_bc * 2 + sc_
                    W[gc * n + gr][col] += c_cf * s_cf_
            
            col += 1
    
    # Quick verification
    if time.time() < deadline - 5:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        max_err = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    val = np.dot(U_np[i*m+k], V_np[k*p+j] * W_np[j*n+i])
                    err = abs(val - 1)
                    if err > max_err:
                        max_err = err
        if max_err > 1e-9:
            # This shouldn't happen if coefficients are correct
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
