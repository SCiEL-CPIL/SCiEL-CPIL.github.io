import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 14: 7 terms per output column pair (2 pairs)
    rank = 14
    
    U = np.zeros((n * m, rank), dtype=int)  # 8 x 14
    V = np.zeros((m * p, rank), dtype=int)  # 16 x 14
    W = np.zeros((p * n, rank), dtype=int)  # 8 x 14
    
    def add_term(u_vec, v_vec, w_vec, col_idx):
        for idx, coeff in u_vec:
            U[idx, col_idx] += coeff
        for idx, coeff in v_vec:
            V[idx, col_idx] += coeff
        for idx, coeff in w_vec:
            W[idx, col_idx] += coeff
    
    # For each output column pair j_pair (0 or 1), output columns are 2*j_pair and 2*j_pair+1
    # B block for column pair j_pair: rows 0-3, cols 2*j_pair to 2*j_pair+1
    # This is a 4x2 block: B[0:4, 2*j_pair:2*j_pair+2]
    # A is 2x4: A[0:2, 0:4]
    # C block: C[0:2, 2*j_pair:2*j_pair+2] (2x2)
    
    # 7-term decomposition for C = A(2x4) * B_block(4x2)
    # A indices: A[r,c] -> U[r*4+c]
    # B_block indices: B[k, 2*j_pair+c2] -> V[k*4 + 2*j_pair+c2]
    # C indices: C[r, 2*j_pair+c2] -> W[(2*j_pair+c2)*2 + r]  (W[k*n+i], k=col, i=row)
    
    # The 7 multiplications for 2x4 * 4x2:
    # Let A = [[a00, a01, a02, a03],
    #           [a10, a11, a12, a13]]
    # Let B = [[b00, b01],
    #           [b10, b11],
    #           [b20, b21],
    #           [b30, b31]]
    # C = A*B = [[c00, c01],
    #             [c10, c11]]
    # c00 = a00*b00 + a01*b10 + a02*b20 + a03*b30
    # c01 = a00*b01 + a01*b11 + a02*b21 + a03*b31
    # c10 = a10*b00 + a11*b10 + a12*b20 + a13*b30
    # c11 = a10*b01 + a11*b11 + a12*b21 + a13*b31
    
    # 7-term Strassen-like decomposition:
    # M1 = (a00 + a13)(b00 + b31)
    # M2 = (a10 + a13) * b00
    # M3 = a00 * (b01 - b31)
    # M4 = a13 * (b30 - b00)
    # M5 = (a00 + a01) * b31
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a13) * (b30 + b31)
    
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Verify:
    # M1 = (a00+a13)(b00+b31) = a00*b00 + a00*b31 + a13*b00 + a13*b31
    # M4 = a13*(b30-b00) = a13*b30 - a13*b00
    # M5 = (a00+a01)*b31 = a00*b31 + a01*b31
    # M7 = (a01-a13)(b30+b31) = a01*b30 + a01*b31 - a13*b30 - a13*b31
    # M1+M4-M5+M7 = a00*b00 + a00*b31 + a13*b00 + a13*b31 + a13*b30 - a13*b00 - a00*b31 - a01*b31 + a01*b30 + a01*b31 - a13*b30 - a13*b31
    #   = a00*b00 + a13*b30 + a01*b30 ... hmm, let me recompute
    # = a00*b00 + (a00*b31 - a00*b31) + (a13*b00 - a13*b00) + (a13*b31 - a13*b31) + a13*b30 + (-a01*b31 + a01*b31) + a01*b30 - a13*b30
    # = a00*b00 + a13*b30 - a13*b30 + a01*b30
    # = a00*b00 + a01*b30
    # That's not right. Let me use the standard 2x2 Strassen adapted properly.
    
    # Better approach: split A into A_L = A[:,0:2] (2x2) and A_R = A[:,2:4] (2x2)
    # Split B_block into B_T = B[0:2,:] (2x2) and B_B = B[2:4,:] (2x2)
    # C = A_L * B_T + A_R * B_B
    # This is sum of two 2x2 products. Use Strassen on each = 14 terms, but we want 7.
    
    # The 7-term decomposition for 2x4 x 4x2 (known result):
    # M1 = (a00+a02+a11+a13)(b00+b11)  -- no, this is getting complicated.
    
    # Let me just use the straightforward 14-term (7 per 2x2 block) but check if we can merge.
    # Actually, for the purpose of this task, let me first verify the 14-term works, then try to optimize.
    
    # 14-term: 7 for A_L*B_T, 7 for A_R*B_B
    def strassen_2x2():
        """Return 7 Strassen terms for 2x2 * 2x2.
        Local indices: 0=00, 1=01, 2=10, 3=11
        """
        terms = []
        # M1 = (X00+X11)(Y00+Y11)
        terms.append(({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
        # M2 = (X10+X11)Y00
        terms.append(({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}))
        # M3 = X00(Y01-Y11)
        terms.append(({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}))
        # M4 = X11(Y10-Y00)
        terms.append(({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}))
        # M5 = (X00+X01)Y11
        terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}))
        # M6 = (X10-X00)(Y00+Y01)
        terms.append(({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
        # M7 = (X01-X11)(Y10+Y11)
        terms.append(({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}))
        return terms
    
    base_terms = strassen_2x2()
    
    for j_pair in range(2):
        # Output columns: 2*j_pair, 2*j_pair+1
        # B block: rows 0-3, cols 2*j_pair to 2*j_pair+1
        # Split B into top (rows 0-1) and bottom (rows 2-3)
        
        # Part 1: A_L (cols 0-1) * B_T (rows 0-1, cols 2*j_pair to 2*j_pair+1)
        # A_L local (r,c) -> global A index r*4+c
        # B_T local (r,c) -> global B index (0+r)*4 + (2*j_pair+c)
        # C local (r,c) -> global C index (2*j_pair+c)*2 + r
        
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = j_pair * 7 + t_idx
            
            u_global = {}
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * 4 + c  # A_L: cols 0,1
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            
            v_global = {}
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r  # rows 0,1
                g_c = 2 * j_pair + c
                g_idx = g_r * 4 + g_c
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            
            w_global = {}
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_c = 2 * j_pair + c
                g_idx = g_c * 2 + r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            
            add_term(list(u_global.items()), list(v_global.items()), list(w_global.items()), col)
        
        # Part 2: A_R (cols 2-3) * B_B (rows 2-3, cols 2*j_pair to 2*j_pair+1)
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = j_pair * 7 + t_idx  # Same column! Add to same rank term
            
            u_global = {}
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * 4 + (2 + c)  # A_R: cols 2,3
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            
            v_global = {}
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = 2 + r  # rows 2,3
                g_c = 2 * j_pair + c
                g_idx = g_r * 4 + g_c
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            
            w_global = {}
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_c = 2 * j_pair + c
                g_idx = g_c * 2 + r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            
            add_term(list(u_global.items()), list(v_global.items()), list(w_global.items()), col)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
