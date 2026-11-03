import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 150s limit, keep 10s margin
    
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6
    
    # Use Strassen's algorithm for 2x2 blocks
    # Partition 6x6 into 3x3 grid of 2x2 blocks
    # Each 2x2 multiplication uses 7 multiplications (Strassen)
    # Total rank = 3 * 3 * 3 * 7 = 189
    
    rank = 3 * 3 * 3 * 7
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2 coefficients
    # For 2x2 matrix multiplication:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    strassen_U = [
        {0: 1, 3: 1},      # M1: A11+A22 (local indices 0,3)
        {2: 1, 3: 1},      # M2: A21+A22
        {0: 1},            # M3: A11
        {3: 1},            # M4: A22
        {0: 1, 1: 1},      # M5: A11+A12
        {2: 1, 0: -1},     # M6: A21-A11
        {1: 1, 3: -1},     # M7: A12-A22
    ]
    
    strassen_V = [
        {0: 1, 3: 1},      # M1: B11+B22
        {0: 1},            # M2: B11
        {2: 1, 3: -1},     # M3: B21-B22
        {1: 1, 0: -1},     # M4: B12-B11
        {3: 1},            # M5: B22
        {0: 1, 1: 1},      # M6: B11+B12
        {2: 1, 3: 1},      # M7: B21+B22
    ]
    
    strassen_W = [
        {0: 1, 2: 1, 1: -1, 3: 1},   # C11 = M1+M4-M5+M7
        {0: 1, 1: -1},                # C12 = M3+M5
        {0: 1, 1: 1},                 # C21 = M2+M4
        {0: 1, 1: -1, 2: 1, 3: 1},    # C22 = M1-M2+M3+M6
    ]
    
    def get_rank_idx(bi, bj, bk, si):
        return ((bi * 3 + bj) * 3 + bk) * 7 + si
    
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for si in range(7):
                    rank_idx = get_rank_idx(bi, bj, bk, si)
                    
                    # U: from A block (bi, bj)
                    # A is 6x6, block (bi, bj) has rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
                    # U index: row * m + col = row * 6 + col
                    for (local_idx, coeff) in strassen_U[si].items():
                        local_row = local_idx // 2
                        local_col = local_idx % 2
                        global_row = 2 * bi + local_row
                        global_col = 2 * bj + local_col
                        U[global_row * 6 + global_col, rank_idx] += coeff
                    
                    # V: from B block (bj, bk)
                    # B is 6x6, block (bj, bk) has rows 2*bj..2*bj+1, cols 2*bk..2*bk+1
                    # V index: row * p + col = row * 6 + col
                    for (local_idx, coeff) in strassen_V[si].items():
                        local_row = local_idx // 2
                        local_col = local_idx % 2
                        global_row = 2 * bj + local_row
                        global_col = 2 * bk + local_col
                        V[global_row * 6 + global_col, rank_idx] += coeff
                    
                    # W: contributes to C block (bi, bk)
                    # C is 6x6, block (bi, bk) has rows 2*bi..2*bi+1, cols 2*bk..2*bk+1
                    # W index: k*n + i where k=col, i=row, n=6
                    # So index = col * 6 + row
                    for (local_idx, coeff) in strassen_W[si].items():
                        local_row = local_idx // 2
                        local_col = local_idx % 2
                        global_row = 2 * bi + local_row
                        global_col = 2 * bk + local_col
                        W[global_col * 6 + global_row, rank_idx] += coeff
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
