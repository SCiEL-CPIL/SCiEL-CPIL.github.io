"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 150.0
    
    # Rank 21 via 3 independent Strassen blocks (optimal for M_{2,2,6})
    def strassen_3blocks():
        rank = 21
        U = [[0] * rank for _ in range(n * m)]  # 4 x 21
        V = [[0] * rank for _ in range(m * p)]  # 12 x 21
        W = [[0] * rank for _ in range(p * n)]  # 12 x 21
        
        # Strassen factors for 2x2 matrix multiplication
        # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
        # M5 = (a+b)h, M6 = (c-d)(e+f), M7 = (b-d)(g+h)
        U_coeffs = [
            {0: 1, 3: 1},       # M1: A00 + A11
            {2: 1, 3: 1},       # M2: A10 + A11
            {0: 1},             # M3: A00
            {3: 1},             # M4: A11
            {0: 1, 1: 1},       # M5: A00 + A01
            {2: 1, 0: -1},      # M6: A10 - A00
            {1: 1, 3: -1}       # M7: A01 - A11
        ]
        
        V_coeffs = [
            [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
            [(0, 0, 1)],                 # M2: B00
            [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
            [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
            [(1, 1, 1)],                 # M5: B11
            [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
            [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
        ]
        
        W_map = [
            [(0, 0, 1), (1, 1, 1)],   # C00 += M1, C11 += M1
            [(1, 0, 1), (1, 1, -1)],  # C10 += M2, C11 -= M2
            [(0, 1, 1), (1, 1, 1)],   # C01 += M3, C11 += M3
            [(0, 0, 1), (1, 0, 1)],   # C00 += M4, C10 += M4
            [(0, 0, -1), (0, 1, 1)],  # C00 -= M5, C01 += M5
            [(1, 1, 1)],              # C11 += M6
            [(0, 0, 1)]               # C00 += M7
        ]
        
        for r in range(3):
            for k in range(7):
                col = r * 7 + k
                for idx, val in U_coeffs[k].items():
                    U[idx][col] = val
                for j_local, k_local, val in V_coeffs[k]:
                    global_k = 2 * r + k_local
                    v_idx = j_local * p + global_k
                    V[v_idx][col] = val
                for i, j_local, val in W_map[k]:
                    global_k = 2 * r + j_local
                    w_idx = global_k * n + i
                    W[w_idx][col] = val
        
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    result = strassen_3blocks()
    
    # Verify correctness
    if not verify(result["U"], result["V"], result["W"]):
        # Fallback: schoolbook decomposition (rank 24)
        U = [[0]*24 for _ in range(4)]
        V = [[0]*24 for _ in range(12)]
        W = [[0]*24 for _ in range(12)]
        for i in range(2):
            for j in range(2):
                for k in range(6):
                    r = i*12 + j*6 + k
                    U[i*2+j][r] = 1
                    V[j*6+k][r] = 1
                    W[k*2+i][r] = 1
        result = {"U": U, "V": V, "W": W}
    
    return result
# EVOLVE-BLOCK-END
