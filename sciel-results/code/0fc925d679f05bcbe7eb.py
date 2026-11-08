import numpy as np
import time

def construct():
    n, m, p = 4, 4, 6
    start_time = time.time()
    deadline = start_time + 140

    # Strassen decomposition for 2x2x2
    # u, v, w are lists of 7 tuples of (index, value)
    # u: (i, j, val)
    # v: (j, k, val)
    # w: (k, i, val)
    
    # M1 = (a+d)(e+h)
    u1 = [(0,0,1), (1,1,1)]
    v1 = [(0,0,1), (1,1,1)]
    w1 = [(0,0,1), (1,1,1)] # C00 += M1, C11 += M1
    
    # M2 = (c+d)e
    u2 = [(1,0,1), (1,1,1)]
    v2 = [(0,0,1)]
    w2 = [(0,1,1)] # C10 += M2 (k=0, i=1) -> index k*2+i = 0*2+1=1. Wait.
    
    # Let's re-verify W index mapping.
    # Task: W[k*n+i, r].
    # For 2x2x2, n=2. W index = k*2 + i.
    # C matrix is 2x2. C[i, k].
    # C00: i=0, k=0. Index 0*2+0=0.
    # C01: i=0, k=1. Index 1*2+0=2.
    # C10: i=1, k=0. Index 0*2+1=1.
    # C11: i=1, k=1. Index 1*2+1=3.
    
    # My previous w lists were (k, i, val).
    # Let's check M1 again.
    # C00 += M1 -> i=0, k=0. Index 0.
    # C11 += M1 -> i=1, k=1. Index 3.
    # So w1 should have entries at index 0 and 3.
    # My previous w1: [(0,0,1), (1,1,1)]. If this is (k,i,val), then:
    # (0,0) -> k=0, i=0 -> idx 0. Correct.
    # (1,1) -> k=1, i=1 -> idx 3. Correct.
    
    # M2: C10 += M2. i=1, k=0. Index 0*2+1=1.
    # w2 should have entry at index 1.
    # w2 = [(0,1,1)] -> k=0, i=1 -> idx 1. Correct.
    
    # M3: C01 += M3, C11 += M3.
    # C01: i=0, k=1. Index 1*2+0=2.
    # C11: i=1, k=1. Index 1*2+1=3.
    # w3 = [(1,0,1), (1,1,1)].
    
    # M4: C00 += M4, C10 += M4.
    # C00: idx 0.
    # C10: idx 1.
    # w4 = [(0,0,1), (0,1,1)].
    
    # M5: C00 -= M5, C01 += M5.
    # C00: idx 0, val -1.
    # C01: idx 2, val 1.
    # w5 = [(0,0,-1), (1,0,1)].
    
    # M6: C11 += M6.
    # C11: idx 3.
    # w6 = [(1,1,1)].
    
    # M7: C00 += M7.
    # C00: idx 0.
    # w7 = [(0,0,1)].

    strassen = [
        (u1, v1, w1),
        (u2, v2, w2),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], w3),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], w4),
        ([(0,0,1), (0,1,1)], [(1,1,1)], w5),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], w6),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], w7)
    ]
    
    # Re-verify u, v for M3-M7
    # M3 = A00(B01-B11). u: (0,0,1). v: (0,1,1), (1,1,-1). Correct.
    # M4 = A11(B10-B00). u: (1,1,1). v: (1,0,1), (0,0,-1). Correct.
    # M5 = (A00+A01)B11. u: (0,0,1), (0,1,1). v: (1,1,1). Correct.
    # M6 = (A10-A00)(B00+B01). u: (1,0,1), (0,0,-1). v: (0,0,1), (0,1,1). Correct.
    # M7 = (A01-A11)(B10+B11). u: (0,1,1), (1,1,-1). v: (1,0,1), (1,1,1). Correct.

    rank = 84
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for I in range(2):
        for J in range(2):
            for k in range(3):
                for r in range(7):
                    u_list, v_list, w_list = strassen[r]
                    
                    for (li, lj, val) in u_list:
                        gi = 2*I + li
                        gj = 2*J + lj
                        U[gi*m + gj, col] += val
                        
                    for (lj, lk, val) in v_list:
                        gj = 2*J + lj
                        gk = 2*k + lk
                        V[gj*p + gk, col] += val
                        
                    for (lk, li, val) in w_list:
                        gk = 2*k + lk
                        gi = 2*I + li
                        W[gk*n + gi, col] += val
                    col += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
