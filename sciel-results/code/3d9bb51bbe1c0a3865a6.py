import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Rank 49 for 4x4x4 core, plus 36 standard terms
    rank = 49 + 36
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)
    
    # Strassen coefficients for 2x2x2
    # u_s[i,j], v_s[j,k], w_s[k,i]
    # s in 0..6
    u_2x2 = [
        [[1, 0], [1, 1]], # M1: (A00+A11)
        [[0, 0], [1, 1]], # M2: (A10+A11)
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: (A00+A01)
        [[-1, 0], [1, 0]],# M6: (A10-A00)
        [[0, 1], [0, -1]] # M7: (A01-A11)
    ]
    v_2x2 = [
        [[1, 0], [0, 1]], # M1: (B00+B11)
        [[1, 0], [0, 0]], # M2: B00
        [[0, 1], [0, -1]],# M3: (B01-B11)
        [[0, -1], [1, 0]],# M4: (B10-B00)
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: (B00+B01)
        [[0, 1], [1, 1]]  # M7: (B10+B11)
    ]
    w_2x2 = [
        [[1, 0], [0, 1]], # C00 += M1, C11 += M1
        [[0, 0], [0, -1]],# C11 -= M2
        [[0, 1], [0, 1]], # C01 += M3, C11 += M3
        [[1, 0], [0, 0]], # C00 += M4
        [[-1, 0], [0, 1]],# C00 -= M5, C11 += M5
        [[0, 0], [1, 1]], # C10 += M6, C11 += M6
        [[1, 0], [0, 0]]  # C00 += M7
    ]
    # Note: w_2x2 indices are [k, i] in my previous thought, but standard is C[i,k].
    # Let's stick to C[i,k] = sum u[i,j] v[j,k] w[k,i]?
    # Task says W[k*n+i, r]. So W is indexed by k, i.
    # My w_2x2 above: w[k, i].
    # M1: C00 += M1, C11 += M1.
    # w[0,0]=1, w[1,1]=1.
    # M2: C11 -= M2. w[1,1]=-1.
    # M3: C01 += M3, C11 += M3. w[0,1]=1, w[1,1]=1.
    # M4: C00 += M4. w[0,0]=1.
    # M5: C00 -= M5, C11 += M5. w[0,0]=-1, w[1,1]=1.
    # M6: C10 += M6, C11 += M6. w[1,0]=1, w[1,1]=1.
    # M7: C00 += M7. w[0,0]=1.
    
    # Correct w_2x2 (indexed [k, i]):
    w_2x2 = [
        [[1, 0], [0, 1]], # M1: C00, C11
        [[0, 0], [0, -1]],# M2: C11
        [[0, 1], [0, 1]], # M3: C01, C11 (k=0,i=1 and k=1,i=1) -> C[1,0]? No C[i,k].
        # C[i,k]. i=0,k=1 is C01. i=1,k=1 is C11.
        # W[k, i].
        # C01: i=0, k=1. W[1, 0] += 1.
        # C11: i=1, k=1. W[1, 1] += 1.
        # So w[1,0]=1, w[1,1]=1.
        [[1, 0], [0, 0]], # M4: C00. i=0,k=0. W[0,0]=1.
        [[-1, 0], [0, 1]],# M5: C00 -=, C11 +=. W[0,0]=-1, W[1,1]=1.
        [[0, 1], [1, 1]], # M6: C10 +=, C11 +=. i=1,k=0 -> W[0,1]=1. i=1,k=1 -> W[1,1]=1.
        [[1, 0], [0, 0]]  # M7: C00 +=. W[0,0]=1.
    ]
    # Re-verify M3: C01 (i=0,k=1) and C11 (i=1,k=1).
    # W[1,0] and W[1,1].
    # My list: [[0, 1], [0, 1]] -> w[0,0]=0, w[0,1]=1, w[1,0]=0, w[1,1]=1.
    # This corresponds to W[0,1]=1 and W[1,1]=1.
    # W[k, i].
    # C01: i=0, k=1. Index in W is k*n+i = 1*2+0 = 2? No, local.
    # Local W[k, i].
    # C01: i=0, k=1. W[1, 0].
    # C11: i=1, k=1. W[1, 1].
    # So w[1,0]=1, w[1,1]=1.
    # My list w_2x2[2] is [[0, 1], [0, 1]].
    # w[0,0]=0, w[0,1]=1. w[1,0]=0, w[1,1]=1.
    # This is W[0,1]=1 and W[1,1]=1.
    # W[0,1] is k=0, i=1. C10.
    # W[1,1] is k=1, i=1. C11.
    # So M3 adds to C10 and C11?
    # Strassen M3 = A00(B01-B11).
    # C01 = A00B01 + A01B11.
    # C11 = A10B01 + A11B11.
    # M3 contributes to C01 and C11?
    # C01 = M3 + M5.
    # C11 = M1 - M2 + M3 + M6.
    # So M3 adds to C01 and C11.
    # C01: i=0, k=1. W[1,0].
    # C11: i=1, k=1. W[1,1].
    # So w[1,0]=1, w[1,1]=1.
    # My list has w[0,1]=1. This is wrong.
    
    # Let's hardcode w_2x2 correctly.
    w_2x2 = [
        [[1, 0], [0, 1]], # M1: C00(0,0), C11(1,1)
        [[0, 0], [0, -1]],# M2: C11(1,1)
        [[0, 0], [1, 1]], # M3: C01(1,0), C11(1,1) -> W[1,0]=1, W[1,1]=1
        [[1, 0], [0, 0]], # M4: C00(0,0)
        [[-1, 0], [0, 1]],# M5: C00(0,0), C11(1,1)
        [[0, 1], [1, 1]], # M6: C10(0,1), C11(1,1) -> W[0,1]=1, W[1,1]=1
        [[1, 0], [0, 0]]  # M7: C00(0,0)
    ]
    
    # Strassen coefficients for 4x4x4 (Outer)
    # alpha[r, i_a, j_a]
    alpha = [
        [[1, 0], [0, 1]], # M1: A00+A11
        [[0, 0], [1, 1]], # M2: A10+A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: A00+A01
        [[-1, 0], [1, 0]],# M6: A10-A00
        [[0, 1], [0, -1]] # M7: A01-A11
    ]
    beta = [
        [[1, 0], [0, 1]], # M1: B00+B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 1], [0, -1]],# M3: B01-B11
        [[0, -1], [1, 0]],# M4: B10-B00
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: B00+B01
        [[0, 1], [1, 1]]  # M7: B10+B11
    ]
    gamma = [
        [[1, 0], [0, 1]], # C00, C11
        [[0, 0], [0, -1]],# C11
        [[0, 0], [1, 1]], # C01, C11
        [[1, 0], [0, 0]], # C00
        [[-1, 0], [0, 1]],# C00, C11
        [[0, 1], [1, 1]], # C10, C11
        [[1, 0], [0, 0]]  # C00
    ]
    
    col = 0
    for r in range(7):
        for s in range(7):
            # u_{r,s}[i,j]
            for i in range(4):
                for j in range(4):
                    i_a, i_b = i // 2, i % 2
                    j_a, j_b = j // 2, j % 2
                    U[i * m + j, col] = alpha[r][i_a][j_a] * u_2x2[s][i_b][j_b]
            
            # v_{r,s}[j,k]
            for j in range(4):
                for k in range(4):
                    j_a, j_b = j // 2, j % 2
                    k_a, k_b = k // 2, k % 2
                    V[j * p + k, col] = beta[r][j_a][k_a] * v_2x2[s][j_b][k_b]
            
            # w_{r,s}[k,i]
            for k in range(4):
                for i in range(4):
                    k_a, k_b = k // 2, k % 2
                    i_a, i_b = i // 2, i % 2
                    W[k * n + i, col] = gamma[r][i_a][k_a] * w_2x2[s][k_b][i_b]
            
            col += 1
            
    # Standard terms for j=4 or k=4
    for i in range(4):
        for j in range(5):
            for k in range(5):
                if j < 4 and k < 4:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
