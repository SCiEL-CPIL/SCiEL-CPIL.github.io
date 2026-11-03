import numpy as np

def construct():
    N, M, P = 4, 4, 5
    
    # 1. Strassen terms for 2x2
    # u, v, w are vectors of length 4
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) for A, B
    # Indices for C (W): k*2+i. 
    # 0:(0,0), 1:(0,1)? No.
    # C is 2x2. i is row, k is col.
    # W index = k*2 + i.
    # C00: k=0, i=0 -> 0
    # C01: k=1, i=0 -> 2
    # C10: k=0, i=1 -> 1
    # C11: k=1, i=1 -> 3
    # So W indices:
    # C00 -> 0
    # C10 -> 1
    # C01 -> 2
    # C11 -> 3
    
    # Strassen Formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    s2x2 = [
        # (u, v, w)
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # C00(0), C11(3)
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # C10(1), C11(3)
        # M3
        ([1, 0, 0, 0], [0, -1, 1, 0], [0, 0, 1, 1]), # C01(2), C11(3)
        # M4
        ([0, 0, 0, 1], [1, 1, 0, -1], [1, 1, 0, 0]), # C00(0), C10(1)
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # C00(0), C01(2)
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # C11(3)
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # C00(0)
    ]
    
    # 2. Block Strassen terms for 4x4 (composed of 2x2 blocks)
    # Blocks A_ab, B_bc, C_ac. a,b,c in {0,1}.
    # u_blocks[a][b], v_blocks[b][c], w_blocks[a][c]
    
    b2x2 = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    terms = []
    
    # Part 1: 4x4x4 using block Strassen
    for ub, vb, wb in b2x2:
        for u_loc, v_loc, w_loc in s2x2:
            gu = {}
            gv = {}
            gw = {}
            
            # Map local A indices (i,j) to global A indices
            # Global A index = (2*a + i)*4 + (2*b + j)
            # u_loc[k] corresponds to A[i,j] where k = i*2+j?
            # My s2x2 u_loc indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
            # So i = k//2, j = k%2.
            
            for a in range(2):
                for b in range(2):
                    if (a,b) in ub:
                        coeff_a = ub[(a,b)]
                        for k in range(4):
                            if u_loc[k] != 0:
                                i = k // 2
                                j = k % 2
                                g_idx = (2*a + i)*4 + (2*b + j)
                                val = coeff_a * u_loc[k]
                                gu[g_idx] = gu.get(g_idx, 0) + val
            
            # Map local B indices (j,k) to global B indices
            # Global B index = (2*b + j)*4 + (2*c + k)
            # v_loc indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
            # j = k//2, k = k%2.
            
            for b in range(2):
                for c in range(2):
                    if (b,c) in vb:
                        coeff_b = vb[(b,c)]
                        for k in range(4):
                            if v_loc[k] != 0:
                                j = k // 2
                                kk = k % 2
                                g_idx = (2*b + j)*4 + (2*c + kk)
                                val = coeff_b * v_loc[k]
                                gv[g_idx] = gv.get(g_idx, 0) + val
            
            # Map local C indices (i,k) to global C indices
            # Global C index = (2*c + k)*4 + (2*a + i)
            # w_loc indices: 0:C00, 1:C10, 2:C01, 3:C11.
            # My mapping above:
            # 0: k=0, i=0
            # 1: k=0, i=1
            # 2: k=1, i=0
            # 3: k=1, i=1
            # So i = k % 2, k = k // 2.
            
            for a in range(2):
                for c in range(2):
                    if (a,c) in wb:
                        coeff_c = wb[(a,c)]
                        for k in range(4):
                            if w_loc[k] != 0:
                                i = k % 2
                                kk = k // 2
                                g_idx = (2*c + kk)*4 + (2*a + i)
                                val = coeff_c * w_loc[k]
                                gw[g_idx] = gw.get(g_idx, 0) + val
            
            terms.append((gu, gv, gw))
            
    # Part 2: 4x4x1 for the 5th column
    # C[:, 4] = A @ B[:, 4]
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Term for each i, j
    for i in range(4):
        for j in range(4):
            gu = {(i*4 + j): 1}
            gv = {(j*5 + 4): 1}
            gw = {(4*4 + i): 1} # k=4, i=i -> 4*4 + i
            terms.append((gu, gv, gw))
            
    # Build matrices
    rank = len(terms)
    U = np.zeros((N*M, rank))
    V = np.zeros((M*P, rank))
    W = np.zeros((P*N, rank))
    
    for r, (gu, gv, gw) in enumerate(terms):
        for idx, val in gu.items():
            U[idx, r] = val
        for idx, val in gv.items():
            V[idx, r] = val
        for idx, val in gw.items():
            W[idx, r] = val
            
    return {'U': U, 'V': V, 'W': W}
