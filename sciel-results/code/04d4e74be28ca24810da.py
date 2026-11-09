import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    # Strassen 2x2 decomposition terms
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # M1: (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3: a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4: d(g-e)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5: (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6: (c-a)(e+f)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Laderman 3x3 decomposition terms (rank 23)
    # A = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    # B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    # C = A * B
    #
    # Laderman's algorithm:
    # M1 = (a00+a01+a02)(b00+b01+b02)
    # M2 = (a00+a01+a02)(b10+b11+b12)
    # M3 = a00(b10-b00)
    # M4 = a10(b20-b10)
    # M5 = (a10+a11+a12)(b20+b21+b22)
    # M6 = a20(b20-b10)
    # M7 = (a20+a21+a22)(b20+b21+b22)
    # M8 = a20(b21-b11)
    # M9 = a00(b11)
    # M10 = a10(b11)
    # M11 = a20(b11)
    # M12 = a00(b21)
    # M13 = a10(b21)
    # M14 = a20(b21)
    # M15 = a00(b22)
    # M16 = a10(b22)
    # M17 = a20(b22)
    # M18 = a01(b20)
    # M19 = a11(b20)
    # M20 = a21(b20)
    # M21 = (a00+a01+a02)(b20+b21+b22)
    # M22 = (a10+a11+a12)(b20+b21+b22)
    # M23 = (a00+a01+a02+a10+a11+a12)(b20+b21+b22)
    #
    # C00 = M1 + M3 + M12 - M15
    # C01 = M1 - M3 + M9 - M12 + M15
    # C02 = M1 - M9
    # C10 = M2 + M4 + M13 - M16
    # C11 = M2 - M4 + M10 - M13 + M16
    # C12 = M2 - M10
    # C20 = M5 + M6 + M14 - M17
    # C21 = M5 - M6 + M11 - M14 + M17
    # C22 = M5 - M11
    
    # Let me use the standard Laderman decomposition. The indices are:
    # A[i,j] where i=row, j=col. Index = i*3+j
    # B[i,j] where i=row, j=col. Index = i*3+j
    # C[i,j] where i=row, j=col. Index = i*3+j
    
    laderman_terms = [
        # M1: (a00+a01+a02)(b00+b01+b02)
        ({0: 1, 1: 1, 2: 1}, {0: 1, 1: 1, 2: 1}, {0: 1, 1: 1, 2: 1}),
        # M2: (a00+a01+a02)(b10+b11+b12)
        ({0: 1, 1: 1, 2: 1}, {3: 1, 4: 1, 5: 1}, {3: 1, 4: 1, 5: 1}),
        # M3: a00(b10-b00)
        ({0: 1}, {3: 1, 0: -1}, {3: 1, 0: -1}),
        # M4: a10(b20-b10)
        ({3: 1}, {6: 1, 3: -1}, {6: 1, 3: -1}),
        # M5: (a10+a11+a12)(b20+b21+b22)
        ({3: 1, 4: 1, 5: 1}, {6: 1, 7: 1, 8: 1}, {6: 1, 7: 1, 8: 1}),
        # M6: a20(b20-b10)
        ({6: 1}, {6: 1, 3: -1}, {6: 1, 3: -1}),
        # M7: (a20+a21+a22)(b20+b21+b22)
        ({6: 1, 7: 1, 8: 1}, {6: 1, 7: 1, 8: 1}, {6: 1, 7: 1, 8: 1}),
        # M8: a20(b21-b11)
        ({6: 1}, {7: 1, 4: -1}, {7: 1, 4: -1}),
        # M9: a00*b11
        ({0: 1}, {4: 1}, {4: 1}),
        # M10: a10*b11
        ({3: 1}, {4: 1}, {4: 1}),
        # M11: a20*b11
        ({6: 1}, {4: 1}, {4: 1}),
        # M12: a00*b21
        ({0: 1}, {7: 1}, {7: 1}),
        # M13: a10*b21
        ({3: 1}, {7: 1}, {7: 1}),
        # M14: a20*b21
        ({6: 1}, {7: 1}, {7: 1}),
        # M15: a00*b22
        ({0: 1}, {8: 1}, {8: 1}),
        # M16: a10*b22
        ({3: 1}, {8: 1}, {8: 1}),
        # M17: a20*b22
        ({6: 1}, {8: 1}, {8: 1}),
        # M18: a01*b20
        ({1: 1}, {6: 1}, {6: 1}),
        # M19: a11*b20
        ({4: 1}, {6: 1}, {6: 1}),
        # M20: a21*b20
        ({7: 1}, {6: 1}, {6: 1}),
        # M21: (a00+a01+a02)(b20+b21+b22)
        ({0: 1, 1: 1, 2: 1}, {6: 1, 7: 1, 8: 1}, {6: 1, 7: 1, 8: 1}),
        # M22: (a10+a11+a12)(b20+b21+b22)
        ({3: 1, 4: 1, 5: 1}, {6: 1, 7: 1, 8: 1}, {6: 1, 7: 1, 8: 1}),
        # M23: (a00+a01+a02+a10+a11+a12)(b20+b21+b22)
        ({0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}, {6: 1, 7: 1, 8: 1}, {6: 1, 7: 1, 8: 1}),
    ]
    
    # Verify Laderman 3x3
    def verify_laderman():
        sz = 3
        r = 23
        U3 = np.zeros((sz*sz, r))
        V3 = np.zeros((sz*sz, r))
        W3 = np.zeros((sz*sz, r))
        
        for t, (u_c, v_c, w_c) in enumerate(laderman_terms):
            for idx, coeff in u_c.items():
                rr, cc = divmod(idx, sz)
                U3[rr*sz+cc, t] += coeff
            for idx, coeff in v_c.items():
                rr, cc = divmod(idx, sz)
                V3[rr*sz+cc, t] += coeff
            for idx, coeff in w_c.items():
                rr, cc = divmod(idx, sz)
                W3[cc*sz+rr, t] += coeff
        
        U3t = U3.reshape(sz, sz, r)
        V3t = V3.reshape(sz, sz, r)
        W3t = W3.reshape(sz, sz, r)
        T3 = np.einsum('ikt,kjt,jit->ikj', U3t, V3t, W3t)
        return np.allclose(T3, np.ones((sz,sz,sz)), atol=1e-10)
    
    laderman_ok = verify_laderman()
    
    if not laderman_ok:
        # Fall back to Strassen(2x2) x schoolbook(3x3) = rank 189
        rank_fb = 189
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                        for local_idx, coeff in u_coeffs.items():
                            r, c = divmod(local_idx, 2)
                            global_r = 2 * i + r
                            global_c = 2 * k + c
                            U_fb[global_r * m + global_c][col_fb] += coeff
                        for local_idx, coeff in v_coeffs.items():
                            r, c = divmod(local_idx, 2)
                            global_r = 2 * k + r
                            global_c = 2 * j + c
                            V_fb[global_r * p + global_c][col_fb] += coeff
                        for local_idx, coeff in w_coeffs.items():
                            r, c = divmod(local_idx, 2)
                            global_r = 2 * i + r
                            global_c = 2 * j + c
                            W_fb[global_c * n + global_r][col_fb] += coeff
                        col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Build 6x6 using Strassen(2x2) tensor Laderman(3x3)
    # Rank = 7 * 23 = 161
    rank = 161
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for t, (u_lad, v_lad, w_lad) in enumerate(laderman_terms):
        for s, (u_str, v_str, w_str) in enumerate(strassen_terms):
            for (ik_idx, cu) in u_lad.items():
                i, k = divmod(ik_idx, 3)
                for (elem, cs) in u_str.items():
                    a, b = divmod(elem, 2)
                    global_r = 2 * i + a
                    global_c = 2 * k + b
                    U[global_r * m + global_c][col] += cu * cs
            
            for (kj_idx, cv) in v_lad.items():
                k, j = divmod(kj_idx, 3)
                for (elem, cs) in v_str.items():
                    a, b = divmod(elem, 2)
                    global_r = 2 * k + a
                    global_c = 2 * j + b
                    V[global_r * p + global_c][col] += cv * cs
            
            for (ij_idx, cw) in w_lad.items():
                i, j = divmod(ij_idx, 3)
                for (elem, cs) in w_str.items():
                    a, b = divmod(elem, 2)
                    global_r = 2 * i + a
                    global_c = 2 * j + b
                    W[global_c * n + global_r][col] += cw * cs
            
            col += 1
    
    # Verify the full 6x6 decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    U_t = U_np.reshape(n, m, rank)
    V_t = V_np.reshape(m, p, rank)
    W_t = W_np.reshape(p, n, rank)
    
    T = np.einsum('ikt,kjt,jit->ikj', U_t, V_t, W_t)
    expected = np.ones((n, m, p))
    
    if not np.allclose(T, expected, atol=1e-10):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col_s] = 1
                    V_s[j * p + k][col_s] = 1
                    W_s[k * n + i][col_s] = 1
                    col_s += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
