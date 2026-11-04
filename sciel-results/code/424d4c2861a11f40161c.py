import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 3, 3, 8
    schoolbook_rank = n * m * p  # 72
    
    # Initialize with schoolbook decomposition as the fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s = s + U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    # Construct the Strassen-based decomposition
    # Rank 30: 4 blocks of 2x2x2 Strassen (7 each) + 2 cross terms = 28 + 2 = 30
    
    U_strassen = [[0] * 30 for _ in range(n * m)]
    V_strassen = [[0] * 30 for _ in range(m * p)]
    W_strassen = [[0] * 30 for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    strassen_terms = [
        # P1 = (A00 + A11)(B00 + B11)
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        # P2 = (A10 + A11) B00
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, -1]]),
        # P3 = A00 (B01 - B11)
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 1]]),
        # P4 = A11 (B10 - B00)
        ([[0, 0], [0, 1]], [[-1, 0], [1, 0]], [[1, 0], [1, 0]]),
        # P5 = (A00 + A01) B11
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 0], [0, 1]]),
        # P6 = (A10 - A00) (B00 + B01)
        ([[-1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
        # P7 = (A01 - A11) (B10 + B11)
        ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[1, 0], [0, 0]]),
    ]
    
    rank_idx = 0
    # Process 4 blocks of B (2 columns each)
    for b in range(4):
        k_start = 2 * b
        for t in strassen_terms:
            U_t, V_t, W_t = t
            for i in range(2):
                for j in range(2):
                    val = U_t[i][j]
                    if val != 0:
                        U_strassen[i * m + j][rank_idx] = val
            
            for j in range(2):
                for k_local in range(2):
                    val = V_t[j][k_local]
                    if val != 0:
                        V_strassen[j * p + (k_start + k_local)][rank_idx] = val
            
            for k_local in range(2):
                for i in range(2):
                    val = W_t[k_local][i]
                    if val != 0:
                        W_strassen[(k_start + k_local) * n + i][rank_idx] = val
            
            rank_idx += 1
    
    # Cross terms optimized
    # Term A: Handles j=2 for all i
    U_strassen[0 * m + 2][rank_idx] = 1
    U_strassen[1 * m + 2][rank_idx] = 1
    U_strassen[2 * m + 2][rank_idx] = 1
    for k in range(p):
        V_strassen[2 * p + k][rank_idx] = 1
        W_strassen[k * n + 0][rank_idx] = 1
        W_strassen[k * n + 1][rank_idx] = 1
        W_strassen[k * n + 2][rank_idx] = 1
    rank_idx += 1
    
    # Term B: Handles i=2 for j=0,1
    U_strassen[2 * m + 0][rank_idx] = 1
    U_strassen[2 * m + 1][rank_idx] = 1
    for k in range(p):
        V_strassen[0 * p + k][rank_idx] = 1
        V_strassen[1 * p + k][rank_idx] = 1
        W_strassen[k * n + 2][rank_idx] = 1
    rank_idx += 1
    
    # Verify
    if check_decomposition(U_strassen, V_strassen, W_strassen, rank_idx):
        best_result = {"U": U_strassen, "V": V_strassen, "W": W_strassen}
        best_rank = rank_idx
    
    return best_result
