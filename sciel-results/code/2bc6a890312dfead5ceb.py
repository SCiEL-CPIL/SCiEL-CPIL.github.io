import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank-7 Strassen for 2x2
    # U, V, W are 4x7
    U_loc = np.zeros((4, 7), dtype=int)
    V_loc = np.zeros((4, 7), dtype=int)
    W_loc = np.zeros((4, 7), dtype=int)
    
    # r=0: (a00+a11)(b00+b11) -> c00+1, c11+1
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # r=1: (a10+a11)b00 -> c10+1, c11-1
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    # r=2: a00(b01-b11) -> c01+1, c11+1
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    # r=3: a11(b10-b00) -> c00+1, c10+1
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    # r=4: (a00+a01)b11 -> c00-1, c01+1
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    # r=5: (a10-a00)(b00+b01) -> c11+1
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    # r=6: (a01-a11)(b10+b11) -> c00+1
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1

    total_rank = 56
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)

    group = 0
    for jj in range(p // 2):
        for half in range(2):
            base_r = group * 7
            group += 1

            # A indices for 2x2 block in A (2x4)
            # A is row-major: i*m + j
            # Block half: cols 2*half, 2*half+1
            # Rows 0, 1
            a_map = [
                0*m + (2*half + 0),
                0*m + (2*half + 1),
                1*m + (2*half + 0),
                1*m + (2*half + 1),
            ]
            
            # B indices for 2x2 block in B (4x8)
            # B is row-major: i*p + j
            # Block half: rows 2*half, 2*half+1
            # Block jj: cols 2*jj, 2*jj+1
            b_map = [
                (2*half + 0)*p + (2*jj + 0),
                (2*half + 0)*p + (2*jj + 1),
                (2*half + 1)*p + (2*jj + 0),
                (2*half + 1)*p + (2*jj + 1),
            ]
            
            # C indices for 2x2 block in C (2x8)
            # C is column-major? "W[k*n+i,r]"
            # k is col, i is row.
            # Block jj: cols 2*jj, 2*jj+1
            # Rows 0, 1
            c_map = [
                (2*jj + 0)*n + 0,
                (2*jj + 1)*n + 0,
                (2*jj + 0)*n + 1,
                (2*jj + 1)*n + 1,
            ]

            # Add contributions
            for r in range(7):
                rr = base_r + r
                for la in range(4):
                    val = U_loc[la, r]
                    if val:
                        U[a_map[la], rr] += val
                for lb in range(4):
                    val = V_loc[lb, r]
                    if val:
                        V[b_map[lb], rr] += val
                for lc in range(4):
                    val = W_loc[lc, r]
                    if val:
                        W[c_map[lc], rr] += val

    if time.time() - start_time > 115:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
