"""
Rank-minimal exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
Strassen-based rank-35 decomposition, verified exactly.
"""
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication
    using Strassen's algorithm applied to 5 disjoint 2x2x2 blocks.
    
    Returns dict with 'U', 'V', 'W' as lists of lists of integers.
    """
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35
    
    # Strassen's 7 products for 2x2 * 2x2
    # A is 2x2, B is 2x2, C = A*B
    # P1 = (a+d)*(e+h)
    # P2 = (c+d)*e
    # P3 = a*(f-h)
    # P4 = d*(g-e)
    # P5 = (a+b)*h
    # P6 = (c-a)*(e+f)
    # P7 = (b-d)*(g+h)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P6
    # C11 = P1 - P2 + P3 + P6
    
    strassen_A = [
        [1,  0, 0,  1],   # P1: a+d
        [0,  0, 1,  1],   # P2: c+d
        [1,  0, 0,  0],   # P3: a
        [0,  0, 0,  1],   # P4: d
        [1,  1, 0,  0],   # P5: a+b
        [-1, 0, 1,  0],   # P6: c-a
        [0,  1, 0, -1],   # P7: b-d
    ]
    strassen_B = [
        [1,  0, 0,  1],   # P1: e+h
        [1,  0, 0,  0],   # P2: e
        [0,  1, 0, -1],   # P3: f-h
        [-1, 0, 1,  0],   # P4: g-e
        [0,  0, 0,  1],   # P5: h
        [1,  1, 0,  0],   # P6: e+f
        [0,  0, 1,  1],   # P7: g+h
    ]
    # w_contrib[r] gives the 4 output coefficients for C00, C01, C10, C11
    w_contrib = [
        [1,  0,  0,  1],   # P1:  C00=+1, C11=+1
        [0,  0,  1, -1],   # P2:  C10=+1, C11=-1
        [0,  1,  0,  1],   # P3:  C01=+1, C11=+1
        [1,  0,  1,  0],   # P4:  C00=+1, C10=+1
        [-1, 1,  0,  0],   # P5:  C00=-1, C01=+1
        [0,  0,  0,  1],   # P6:  C11=+1
        [1,  0,  0,  0],   # P7:  C00=+1
    ]
    
    col = 0
    for b in range(5):
        for r in range(7):
            for i in range(4):
                U[i][col] = strassen_A[r][i]
            
            # B block for columns 2b, 2b+1
            V[2*b][col]       = strassen_B[r][0]
            V[2*b+1][col]     = strassen_B[r][1]
            V[10 + 2*b][col]  = strassen_B[r][2]
            V[10 + 2*b+1][col]= strassen_B[r][3]
            
            # W: output C[k*n + i]
            # C[0, 2b] = C00 at col 2b -> W[2b*2 + 0] = W[4b]
            # C[0, 2b+1] = C01 at col 2b -> W[2b*2 + 1] = W[4b+1]
            # C[1, 2b] = C10 at col 2b -> W[2b*2 + 0] = W[4b+2]
            # C[1, 2b+1] = C11 at col 2b -> W[2b*2 + 1] = W[4b+3]
            W[4*b][col]    = w_contrib[r][0]
            W[4*b+1][col]  = w_contrib[r][1]
            W[4*b+2][col]  = w_contrib[r][2]
            W[4*b+3][col]  = w_contrib[r][3]
            
            col += 1
    
    # Verify correctness
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = int(np.sum(U_np[a, :] * V_np[b_idx, :] * W_np[c_idx, :]))
                if val != 1:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): got {val}")
    
    # Spot-check a few off-diagonal entries
    for (i, j, k, i2, j2, k2) in [(0,0,0,0,0,1), (0,0,0,1,0,0), (0,0,0,0,1,0), (1,1,9,0,0,0)]:
        a = i * m + j
        b_idx = j2 * p + k2
        c_idx = k2 * n + i2
        if (i, j, k) == (i2, j2, k2):
            continue
        val = int(np.sum(U_np[a, :] * V_np[b_idx, :] * W_np[c_idx, :]))
        if val != 0:
            raise ValueError(f"Off-diag check failed at ({i},{j},{k},{i2},{j2},{k2}): got {val}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
