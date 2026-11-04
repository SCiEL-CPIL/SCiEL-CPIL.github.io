"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # 189 Rank Solution: Strassen 2x2 on 3x3 blocks
    rank = 189
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 7 terms
    # A = [[A0, A1], [A2, A3]] (0:00, 1:01, 2:10, 3:11)
    # B = [[B0, B1], [B2, B3]]
    
    # M0 = (A0+A3)(B0+B3) -> C00 += M0, C11 += M0
    # M1 = (A2+A3)B0 -> C10 += M1, C11 -= M1
    # M2 = A0(B1-B3) -> C01 += M2, C11 += M2
    # M3 = A3(B2-B0) -> C00 += M3, C10 += M3
    # M4 = (A0+A1)B3 -> C00 -= M4, C01 += M4
    # M5 = (A2-A0)(B0+B1) -> C11 += M5
    # M6 = (A1-A3)(B2+B3) -> C00 += M6
    
    strassen_terms = [
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]), # M0
        ([(2,1), (3,1)], [(0,1)], [(2,1), (3,-1)]),       # M1
        ([(0,1)], [(1,1), (3,-1)], [(1,1), (3,1)]),       # M2
        ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),       # M3
        ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),       # M4
        ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),       # M5
        ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)])        # M6
    ]
    
    col = 0
    for u_blocks, v_blocks, w_blocks in strassen_terms:
        for (ub, uc) in u_blocks:
            for (vb, vc) in v_blocks:
                ur = ub // 2
                uc_local = ub % 2
                vr = vb // 2
                vc_local = vb % 2
                
                for j in range(3):
                    for i in range(3):
                        for k in range(3):
                            a_idx = (3*ur + i) * m + (3*uc_local + j)
                            b_idx = (3*vr + j) * p + (3*vc_local + k)
                            
                            U[a_idx, col] += uc
                            V[b_idx, col] += vc
                            
                            for (wb, wc) in w_blocks:
                                wr = wb // 2
                                wc_local = wb % 2
                                w_idx = (3*wc_local + k) * n + (3*wr + i)
                                W[w_idx, col] += wc
                            
                            col += 1
                            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
