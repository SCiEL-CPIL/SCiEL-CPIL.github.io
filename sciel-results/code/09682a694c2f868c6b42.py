"""Matrix multiplication 2x4 * 4x8 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Rank-56 decomposition using Strassen (rank 7) for each of 8 sub-problems.
    """
    deadline = time.time() + 100
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Strassen 2x2*2x2 rank-7 decomposition
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00+B01)
    # M3 = (A00-A10)(B10+B11)
    # M4 = (A01-A11)(B00+B10)
    # M5 = (A00+A01)(B01+B11)
    # M6 = (A10+A00)(B11-B01)
    # M7 = (A11+A01)(B10-B00)
    
    # U factors: coefficients of (a00, a01, a10, a11) for each M
    U_s = [
        [1, 0, 0, 1],   # M1: a00+a11
        [0, 0, 1, 1],   # M2: a10+a11
        [1, 0, -1, 0],  # M3: a00-a10
        [0, 1, 0, -1],  # M4: a01-a11
        [1, 1, 0, 0],   # M5: a00+a01
        [1, 0, 1, 0],   # M6: a10+a00
        [0, 1, 0, 1],   # M7: a11+a01
    ]
    
    # V factors: coefficients of (b00, b01, b10, b11) for each M
    V_s = [
        [1, 0, 0, 1],   # M1: b00+b11
        [1, 1, 0, 0],   # M2: b00+b01
        [0, 0, 1, 1],   # M3: b10+b11
        [1, 0, 1, 0],   # M4: b00+b10
        [0, 1, 0, 1],   # M5: b01+b11
        [0, -1, 0, 1],  # M6: b11-b01
        [0, 0, 1, -1],  # M7: b10-b00
    ]
    
    # W factors: for each output c_ij, coefficients of (M1,...,M7)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M5 - M1 - M3
    # C10 = M3 + M6 - M7 - M2
    # C11 = M1 + M3 - M4 + M6
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [-1, 1, -1, 0, 1, 0, 0],  # C01
        [0, -1, 1, 0, 0, 1, -1],  # C10
        [1, 0, 1, -1, 0, 1, 0],   # C11
    ]
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    for jg in range(2):   # j-group: {0,1} or {2,3}
        for kg in range(4):  # k-group: {0,1}, {2,3}, {4,5}, {6,7}
            for r in range(7):
                # U: A[i, j] for i in {0,1}, j in {jg*2, jg*2+1}
                for i in range(2):
                    for lj in range(2):
                        j = jg * 2 + lj
                        U[i * m + j][col] += U_s[r][i * 2 + lj]
                
                # V: B[j, k] for j in {jg*2, jg*2+1}, k in {kg*2, kg*2+1}
                for lj in range(2):
                    for lk in range(2):
                        j = jg * 2 + lj
                        k = kg * 2 + lk
                        V[j * p + k][col] += V_s[r][lj * 2 + lk]
                
                # W: C[i, k] for i in {0,1}, k in {kg*2, kg*2+1}
                for i in range(2):
                    for lk in range(2):
                        k = kg * 2 + lk
                        W[k * n + i][col] += W_s[i * 2 + lk][r]
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
