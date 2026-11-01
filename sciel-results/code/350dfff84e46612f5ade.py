import numpy as np

def construct():
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U, V, W
    # r_local 0..6
    # U[i, j]
    U_coeff = [
        { (0,0): 1, (1,1): 1 }, # P1
        { (1,0): 1, (1,1): 1 }, # P2
        { (0,0): 1 },           # P3
        { (1,1): 1 },           # P4
        { (0,0): 1, (0,1): 1 }, # P5
        { (1,0): 1, (0,0): -1 },# P6
        { (0,1): 1, (1,1): -1 } # P7
    ]
    
    # V[j, k]
    V_coeff = [
        { (0,0): 1, (1,1): 1 }, # P1
        { (0,0): 1 },           # P2
        { (0,1): 1, (1,1): -1 },# P3
        { (1,0): 1, (0,0): -1 },# P4
        { (1,1): 1 },           # P5
        { (0,0): 1, (0,1): 1 }, # P6
        { (1,0): 1, (1,1): 1 }  # P7
    ]
    
    # W[k, i]
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    W_coeff = [
        { (0,0): 1, (1,0): 0, (0,1): 0, (1,1): 0 }, # P1 -> W[0,0]=1
        { (0,0): 0, (1,0): 0, (0,1): 1, (1,1): 0 }, # P2 -> W[0,1]=1 (C10)
        { (0,0): 0, (1,0): 1, (0,1): 0, (1,1): 1 }, # P3 -> W[1,0]=1, W[1,1]=1 (C01, C11)
        { (0,0): 1, (1,0): 0, (0,1): 1, (1,1): 0 }, # P4 -> W[0,0]=1, W[0,1]=1 (C00, C10)
        { (0,0): -1, (1,0): 1, (0,1): 0, (1,1): 0 },# P5 -> W[0,0]=-1, W[1,0]=1 (C00, C01)
        { (0,0): 0, (1,0): 0, (0,1): 0, (1,1): 1 }, # P6 -> W[1,1]=1 (C11)
        { (0,0): 1, (1,0): 0, (0,1): 0, (1,1): 0 }  # P7 -> W[0,0]=1 (C00)
    ]
    # Correction: W indices are (k, i).
    # C_{i,k} is output. W index is k*n+i.
    # So W[k, i] in my dict corresponds to W_global[k*2+i].
    # My dict keys are (k, i).
    
    # Let's re-verify W mapping.
    # P1 contributes to C00 (i=0, k=0) -> W[0,0]. Correct.
    # P2 contributes to C10 (i=1, k=0) -> W[0,1]. Correct.
    # P3 contributes to C01 (i=0, k=1) -> W[1,0] AND C11 (i=1, k=1) -> W[1,1]. Correct.
    # P4 contributes to C00 (i=0, k=0) -> W[0,0] AND C10 (i=1, k=0) -> W[0,1]. Correct.
    # P5 contributes to C00 (i=0, k=0) -> W[0,0] AND C01 (i=0, k=1) -> W[1,0]. Correct.
    # P6 contributes to C11 (i=1, k=1) -> W[1,1]. Correct.
    # P7 contributes to C00 (i=0, k=0) -> W[0,0]. Correct.
    
    for b in range(3):
        for r_local in range(7):
            r = b * 7 + r_local
            # U
            for (i, j), val in U_coeff[r_local].items():
                idx_u = i * m + j
                U[idx_u][r] += val
            # V
            for (j, k_local), val in V_coeff[r_local].items():
                k_global = b * 2 + k_local
                idx_v = j * p + k_global
                V[idx_v][r] += val
            # W
            for (k_local, i), val in W_coeff[r_local].items():
                k_global = b * 2 + k_local
                idx_w = k_global * n + i
                W[idx_w][r] += val

    return {"U": U, "V": V, "W": W}
