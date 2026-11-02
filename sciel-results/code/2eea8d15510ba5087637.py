import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = 145
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p # 52
    
    # We will use Strassen for pairs of columns and schoolbook for the last one
    # 6 pairs (12 cols) + 1 single col (1 col)
    # Rank = 6 * 7 + 4 = 46
    
    total_rank = 46
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen terms for 2x2x2 block
    # U_vecs are length 4: [a, b, c, d] corresponding to A[0,0], A[0,1], A[1,0], A[1,1]
    # V_dict keys are j*p + k
    # W_dict keys are k*n + i
    
    strassen_U = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    
    for pair_idx in range(6):
        k0 = 2 * pair_idx
        k1 = 2 * pair_idx + 1
        
        # V and W dictionaries for each term
        # Term 1: M1
        V_dict_1 = {0 * p + k0: 1, 1 * p + k1: 1}
        W_dict_1 = {k0 * n + 0: 1, k1 * n + 1: 1}
        
        # Term 2: M2
        V_dict_2 = {0 * p + k0: 1}
        W_dict_2 = {k0 * n + 1: 1, k1 * n + 1: -1}
        
        # Term 3: M3
        V_dict_3 = {0 * p + k1: 1, 1 * p + k1: -1}
        W_dict_3 = {k1 * n + 0: 1, k1 * n + 1: 1}
        
        # Term 4: M4
        V_dict_4 = {1 * p + k0: 1, 0 * p + k1: -1}
        W_dict_4 = {k0 * n + 0: 1, k0 * n + 1: 1}
        
        # Term 5: M5
        V_dict_5 = {1 * p + k1: 1}
        W_dict_5 = {k0 * n + 0: -1, k1 * n + 0: 1}
        
        # Term 6: M6
        V_dict_6 = {0 * p + k0: 1, 1 * p + k0: 1}
        W_dict_6 = {k1 * n + 1: 1}
        
        # Term 7: M7
        V_dict_7 = {1 * p + k0: 1, 1 * p + k1: 1}
        W_dict_7 = {k0 * n + 0: 1}
        
        terms = [
            (strassen_U[0], V_dict_1, W_dict_1),
            (strassen_U[1], V_dict_2, W_dict_2),
            (strassen_U[2], V_dict_3, W_dict_3),
            (strassen_U[3], V_dict_4, W_dict_4),
            (strassen_U[4], V_dict_5, W_dict_5),
            (strassen_U[5], V_dict_6, W_dict_6),
            (strassen_U[6], V_dict_7, W_dict_7)
        ]
        
        for U_vec, V_dict, W_dict in terms:
            for i in range(4):
                if U_vec[i] != 0:
                    U[i][col] = U_vec[i]
            for idx, val in V_dict.items():
                V[idx][col] = val
            for idx, val in W_dict.items():
                W[idx][col] = val
            col += 1
        
    # Last column k = 12
    k2 = 12
    schoolbook_terms = [
        ([1, 0, 0, 0], {0 * p + k2: 1}, {k2 * n + 0: 1}),
        ([0, 1, 0, 0], {1 * p + k2: 1}, {k2 * n + 0: 1}),
        ([0, 0, 1, 0], {0 * p + k2: 1}, {k2 * n + 1: 1}),
        ([0, 0, 0, 1], {1 * p + k2: 1}, {k2 * n + 1: 1})
    ]
    
    for U_vec, V_dict, W_dict in schoolbook_terms:
        for i in range(4):
            if U_vec[i] != 0:
                U[i][col] = U_vec[i]
        for idx, val in V_dict.items():
            V[idx][col] = val
        for idx, val in W_dict.items():
            W[idx][col] = val
        col += 1
        
    # Verify
    def verify(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for t in range(rank):
                        C_decomp[i, k] += U_np[i*m+j, t] * V_np[j*p+k, t] * W_np[k*n+i, t]
        
        return np.allclose(C_decomp, C_expected, atol=1e-10)
    
    if verify(U, V, W, total_rank):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
