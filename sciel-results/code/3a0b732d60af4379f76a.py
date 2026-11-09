import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    deadline = time.time() + 140
    
    # Schoolbook fallback
    best_rank = nm * p  # 120
    schoolbook_U = np.zeros((nm, best_rank))
    schoolbook_V = np.zeros((mp, best_rank))
    schoolbook_W = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                schoolbook_U[i * m + j, col] = 1
                schoolbook_V[j * p + k, col] = 1
                schoolbook_W[k * n + i, col] = 1
                col += 1
    best_result = {
        "U": schoolbook_U.tolist(),
        "V": schoolbook_V.tolist(),
        "W": schoolbook_W.tolist()
    }
    
    # Strassen-Winograd construction
    # A (4x5) = [A_core (4x4) | A_tail (4x1)]
    # B (5x6) = [B_top (4x6); B_bottom (1x6)]
    # AB = A_core * B_top + A_tail * B_bottom
    #
    # B_top (4x6) = [B1 (4x4) | B2 (4x2)]
    # A_core * B1: 4x4x4 Strassen (rank 49)
    # A_core * B2: 4x4x2 schoolbook (rank 32)
    # A_tail * B_bottom: 4x1x6 outer product (rank 4)
    # Total: 49 + 32 + 4 = 85
    
    try:
        U_list = []
        V_list = []
        W_list = []
        
        def add_term(u_vec, v_vec, w_vec):
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
        
        # Strassen 2x2x2 terms
        strassen_2x2x2 = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
            ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
            ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
            ([0,1,0,-1], [0,0,1,1], [1,0,0,1]),
        ]
        
        # 4x4x4 Strassen combinations
        # m1 = (A00+A11)(B00+B11), C00+=1, C11+=1
        # m2 = (A10+A11)(B00), C10+=1, C11-=1
        # m3 = A00(B01-B11), C01+=1, C11+=1
        # m4 = A11(B10-B00), C00+=1, C10+=1
        # m5 = (A00+A01)(B11), C00-=1, C01+=1
        # m6 = (A10-A00)(B00+B01), C11+=1
        # m7 = (A01-A11)(B10+B11), C00+=1
        
        A_combos = [
            {(0,0): 1, (1,1): 1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,1): 1},
            {(0,0): 1, (0,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,1): 1, (1,1): -1},
        ]
        
        B_combos = [
            {(0,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (0,0): -1},
            {(1,1): 1},
            {(0,0): 1, (0,1): 1},
            {(1,0): 1, (1,1): 1},
        ]
        
        C_combos = [
            {(0,0): 1, (1,1): 1},
            {(1,0): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1},
            {(0,0): 1, (1,0): 1},
            {(0,0): -1, (0,1): 1},
            {(1,1): 1},
            {(0,0): 1},
        ]
        
        # Build 4x4x4 Strassen (49 terms)
        # A_core is at global (0..3, 0..3) in A
        # B1 is at global (0..3, 0..3) in B
        # C output is at global (0..3, 0..3)
        
        for mj in range(7):
            for t in range(7):
                u_c = strassen_2x2x2[t][0]
                v_c = strassen_2x2x2[t][1]
                w_c = strassen_2x2x2[t][2]
                
                u_vec = np.zeros(nm)
                v_vec = np.zeros(mp)
                w_vec = np.zeros(pn)
                
                for (bi, bj), alpha in A_combos[mj].items():
                    for li in range(2):
                        for lj in range(2):
                            coeff = u_c[li*2+lj]
                            if coeff != 0:
                                gi = 2*bi + li
                                gj = 2*bj + lj
                                if gi < n and gj < m:
                                    u_vec[gi * m + gj] += alpha * coeff
                
                for (bk, bl), beta in B_combos[mj].items():
                    for lj in range(2):
                        for lk in range(2):
                            coeff = v_c[lj*2+lk]
                            if coeff != 0:
                                gj = 2*bk + lj
                                gk = 2*bl + lk
                                if gj < m and gk < p:
                                    v_vec[gj * p + gk] += beta * coeff
                
                for (ci, ck), gamma in C_combos[mj].items():
                    for li in range(2):
                        for lk in range(2):
                            coeff = w_c[li*2+lk]
                            if coeff != 0:
                                gi = 2*ci + li
                                gk = 2*ck + lk
                                if gi < n and gk < p:
                                    w_vec[gk * n + gi] += gamma * coeff
                
                add_term(u_vec, v_vec, w_vec)
        
        # Schoolbook for A_core (4x4) * B2 (4x2): 32 terms
        for gi in range(4):
            for gj in range(4):
                for gk in range(4, 6):
                    u_vec = np.zeros(nm)
                    v_vec = np.zeros(mp)
                    w_vec = np.zeros(pn)
                    u_vec[gi * m + gj] = 1
                    v_vec[gj * p + gk] = 1
                    w_vec[gk * n + gi] = 1
                    add_term(u_vec, v_vec, w_vec)
        
        # Outer product for A_tail (4x1) * B_bottom (1x6): 24 terms
        for gi in range(4):
            for gk in range(6):
                u_vec = np.zeros(nm)
                v_vec = np.zeros(mp)
                w_vec = np.zeros(pn)
                u_vec[gi * m + 4] = 1
                v_vec[4 * p + gk] = 1
                w_vec[gk * n + gi] = 1
                add_term(u_vec, v_vec, w_vec)
        
        # Convert to arrays
        rank = len(U_list)
        U_arr = np.array(U_list).T
        V_arr = np.array(V_list).T
        W_arr = np.array(W_list).T
        
        # Verify
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-10:
            best_rank = rank
            best_result = {
                "U": U_arr.tolist(),
                "V": V_arr.tolist(),
                "W": W_arr.tolist()
            }
        else:
            print(f"Strassen construction error: {max_err}")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
    
    return best_result
