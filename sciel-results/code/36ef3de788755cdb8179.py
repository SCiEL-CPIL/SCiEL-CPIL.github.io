import time
import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Build the rank-49 Strassen-based decomposition
    # Split 14 k-values into 7 pairs: (0,1), (2,3), ..., (12,13)
    # For each pair, use 2x2x2 Strassen (rank 7)
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]   # 4 x 49
    V = [[0] * rank for _ in range(m * p)]   # 28 x 49
    W = [[0] * rank for _ in range(p * n)]   # 28 x 49
    
    # Strassen 2x2x2 decomposition:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M2 + M3 + M4
    # C01 = M5 - M6 - M7
    # C10 = M2 + M6 - M4
    # C11 = M1 + M3 - M5 + M7
    
    # In our tensor: A is 2x2 (indices i,j), B is 2x2 per pair (indices j,k_pair)
    # For a pair (k1, k2), B_pair is 2x2 with B[j,k1] and B[j,k2]
    # C_pair is 2x2 with C[i,k1] and C[i,k2]
    
    # Strassen terms for 2x2x2 (using 0-indexed):
    # Term 0: U: a00,a11; V: b00,b11; W: c00,c11
    # Term 1: U: a10,a11; V: b00; W: c10
    # Term 2: U: a00; V: b01,b11; W: c01
    # Term 3: U: a11; V: b10,b00; W: c10
    # Term 4: U: a00,a01; V: b11; W: c01
    # Term 5: U: a10,a00; V: b00,b01; W: c10
    # Term 6: U: a01,a11; V: b10,b11; W: c00
    
    strassen_terms = [
        # (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
        # Term 0: M1 = (a00+a11)(b00+b11)
        ({0: 1, 3: 1}, {0: 1, 2: 1, 1: 1, 3: 1}, {0: 1, 1: 1, 2: 1, 3: 1}),
        # Term 1: M2 = (a10+a11)b00
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        # Term 2: M3 = a00(b01-b11)
        ({0: 1}, {2: 1, 3: -1}, {1: 1}),
        # Term 3: M4 = a11(b10-b00)
        ({3: 1}, {1: 1, 0: -1}, {2: 1}),
        # Term 4: M5 = (a00+a01)b11
        ({0: 1, 1: 1}, {3: 1}, {1: 1}),
        # Term 5: M6 = (a10-a00)(b00+b01)
        ({2: 1, 0: -1}, {0: 1, 2: 1}, {2: 1}),
        # Term 6: M7 = (a01-a11)(b10+b11)
        ({1: 1, 3: -1}, {1: 1, 3: 1}, {0: 1}),
    ]
    
    # For each pair of k-values, apply the Strassen decomposition
    for pair_idx in range(7):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        base_col = pair_idx * 7
        
        for term_idx, (u_dict, v_dict, w_dict) in enumerate(strassen_terms):
            col = base_col + term_idx
            
            # U: index is i*m + j = 2*i + j
            for idx, val in u_dict.items():
                U[idx][col] += val
            
            # V: index is j*p + k
            # v_dict keys: 0=b(0,k1), 1=b(1,k1), 2=b(0,k2), 3=b(1,k2)
            for idx, val in v_dict.items():
                if idx == 0:  # b(0,k1)
                    V[0 * p + k1][col] += val
                elif idx == 1:  # b(1,k1)
                    V[1 * p + k1][col] += val
                elif idx == 2:  # b(0,k2)
                    V[0 * p + k2][col] += val
                elif idx == 3:  # b(1,k2)
                    V[1 * p + k2][col] += val
            
            # W: index is k*n + i
            # w_dict keys: 0=c(k1,0), 1=c(k2,0), 2=c(k1,1), 3=c(k2,1)
            for idx, val in w_dict.items():
                if idx == 0:  # c(k1, 0)
                    W[k1 * n + 0][col] += val
                elif idx == 1:  # c(k2, 0)
                    W[k2 * n + 0][col] += val
                elif idx == 2:  # c(k1, 1)
                    W[k1 * n + 1][col] += val
                elif idx == 3:  # c(k2, 1)
                    W[k2 * n + 1][col] += val
    
    # Verify the decomposition
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True
