import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Total rank is 22
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen helper
    def add_strassen(u_offset, v_offset, w_offset):
        nonlocal col
        # u_offset: (i, j) base index in U
        # v_offset: (j, k) base index in V
        # w_offset: (i, k) base index in W
        
        # Strassen matrices for 2x2x2
        # U vectors (4 entries: a,b,c,d)
        sU = [
            [1, 0, 1, 0], # M1
            [0, 0, 1, 1], # M2
            [1, 1, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [1, 0, -1, 0],# M6
            [0, 1, 0, -1] # M7
        ]
        # V vectors (4 entries: e,f,g,h)
        sV = [
            [1, 1, 1, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 1, 0, -1],# M3
            [0, -1, 1, 0],# M4
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ]
        # W vectors (4 entries: p11, p12, p21, p22)
        sW = [
            [1, 0, 0, 0], # M1 -> P11
            [0, 0, 1, 0], # M2 -> P21
            [0, 1, 0, 0], # M3 -> P12
            [1, 0, 1, 0], # M4 -> P11, P21
            [-1, 1, 0, 0],# M5 -> P11, P12
            [0, 0, 0, 1], # M6 -> P22
            [1, 0, 0, 0]  # M7 -> P11
        ]
        # Wait, M7 in P11 is +1. P11 = M1 + M4 - M5 + M7.
        # My sW for M7 is [1,0,0,0]. Correct.
        # M4 is [1,0,1,0]. P11 gets +1, P21 gets +1. Correct.
        # M5 is [-1,1,0,0]. P11 gets -1, P12 gets +1. Correct.
        
        for k in range(7):
            # Fill U
            for ii in range(2):
                for jj in range(2):
                    u_idx = (u_offset[0] + ii) * m + (u_offset[1] + jj)
                    U[u_idx][col] = sU[k][ii * 2 + jj]
                    
            # Fill V
            for jj in range(2):
                for kk in range(2):
                    v_idx = (v_offset[0] + jj) * p + (v_offset[1] + kk)
                    V[v_idx][col] = sV[k][jj * 2 + kk]
                    
            # Fill W
            for ii in range(2):
                for kk in range(2):
                    w_idx = (w_offset[0] + kk) * n + (w_offset[1] + ii)
                    # Note: W index is k*n + i.
                    # w_offset is (k_base, i_base)?
                    # In my call, w_offset is (i_base, k_base) or (k_base, i_base)?
                    # Let's define w_offset as (i_base, k_base) to match U.
                    # But W index is k*n + i.
                    # So w_idx = (k_base + kk) * n + (i_base + ii).
                    W[w_idx][col] = sW[k][ii * 2 + kk]
                    
            col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    # U offset: i=0, j=0
    # V offset: j=0, k=0
    # W offset: i=0, k=0
    add_strassen((0, 0), (0, 0), (0, 0))
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    # U offset: i=0, j=0
    # V offset: j=0, k=2
    # W offset: i=0, k=2
    add_strassen((0, 0), (0, 2), (0, 2))
    
    # Block 3: A[0:2, 2:3] * B[2:3, 0:4]
    # Schoolbook
    for i in range(2):
        for k in range(4):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
