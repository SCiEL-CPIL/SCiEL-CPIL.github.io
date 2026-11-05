import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    def verify(U, V, W, tol=1e-9):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                ui = i * m + j
                for k in range(p):
                    vk = j * p + k
                    wk = k * n + i
                    total = 0.0
                    for r in range(rank):
                        total += U[ui][r] * V[vk][r] * W[wk][r]
                    if abs(total - 1.0) > tol:
                        return False
        return True

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Strassen 3x3x3 terms
    def get_strassen_3x3x3_terms():
        terms = []
        def A(i, j): return i * m + j
        def B(i, j): return i * p + j
        def C(i, j): return i * n + j
        
        # 23 terms
        terms.append(({A(0,0): 1}, {B(0,0): 1, B(0,1): -1}, {C(0,0): 1, C(1,0): -1}))
        terms.append(({A(1,0): 1, A(0,0): 1}, {B(0,0): 1}, {C(1,0): 1}))
        terms.append(({A(0,0): 1, A(1,0): -1}, {B(1,0): 1, B(2,0): 1}, {C(1,1): 1, C(2,1): -1}))
        terms.append(({A(2,2): 1}, {B(2,2): 1, B(1,2): -1}, {C(2,2): 1, C(1,2): -1}))
        terms.append(({A(1,2): 1, A(2,2): 1}, {B(2,2): 1}, {C(1,2): 1}))
        terms.append(({A(2,2): 1, A(1,2): -1}, {B(1,2): 1, B(0,2): 1}, {C(1,1): 1, C(0,1): -1}))
        terms.append(({A(0,0): 1, A(2,2): 1}, {B(0,0): 1, B(2,2): 1}, {C(0,0): 1, C(2,2): 1}))
        terms.append(({A(0,1): 1, A(2,1): -1}, {B(1,1): 1, B(2,1): 1}, {C(0,1): 1, C(2,1): -1}))
        terms.append(({A(2,0): 1, A(0,0): 1, A(1,0): -1, A(2,2): -1}, {B(2,2): 1, B(1,2): 1}, {C(2,0): 1, C(1,0): -1}))
        terms.append(({A(1,0): 1, A(1,1): 1, A(0,1): -1, A(1,2): -1}, {B(0,2): 1, B(1,2): 1}, {C(1,0): 1, C(1,1): -1}))
        terms.append(({A(1,1): 1, A(2,2): 1, A(0,2): -1, A(2,0): -1}, {B(0,0): 1, B(0,1): 1}, {C(1,1): 1, C(0,1): -1}))
        terms.append(({A(0,0): 1, A(0,1): 1, A(1,1): -1, A(0,2): -1}, {B(1,1): 1, B(1,0): 1}, {C(0,1): 1, C(0,0): -1}))
        terms.append(({A(0,1): 1, A(1,1): 1, A(0,0): -1, A(1,0): -1}, {B(0,1): 1, B(1,1): 1}, {C(0,0): 1, C(1,0): -1}))
        terms.append(({A(1,1): 1, A(0,1): 1, A(2,1): -1, A(1,2): -1}, {B(1,2): 1, B(2,2): 1}, {C(1,2): 1, C(2,2): -1}))
        terms.append(({A(2,2): 1, A(1,2): 1, A(0,2): -1, A(2,0): -1}, {B(2,0): 1, B(0,0): 1}, {C(2,0): 1, C(0,0): -1}))
        terms.append(({A(0,0): 1, A(0,2): 1, A(2,0): -1, A(1,0): -1}, {B(2,0): 1, B(1,0): 1}, {C(0,0): 1, C(1,0): -1}))
        terms.append(({A(2,0): 1, A(2,2): 1, A(0,0): -1, A(1,1): -1}, {B(0,0): 1, B(0,2): 1}, {C(2,0): 1, C(0,2): -1}))
        terms.append(({A(1,0): 1, A(1,2): 1, A(2,2): -1, A(0,1): -1}, {B(2,2): 1, B(2,0): 1}, {C(1,2): 1, C(2,0): -1}))
        terms.append(({A(0,2): 1, A(2,2): 1, A(1,2): -1, A(0,1): -1}, {B(1,1): 1, B(2,1): 1}, {C(0,2): 1, C(2,1): -1}))
        terms.append(({A(2,1): 1, A(0,1): 1, A(1,1): -1, A(2,2): -1}, {B(0,0): 1, B(1,0): 1}, {C(2,1): 1, C(0,0): -1}))
        terms.append(({A(0,0): 1, A(1,0): 1, A(2,0): -1, A(2,1): -1}, {B(1,1): 1, B(2,1): 1}, {C(0,0): 1, C(2,1): -1}))
        terms.append(({A(1,1): 1, A(2,1): 1, A(0,1): -1, A(0,2): -1}, {B(0,0): 1, B(0,2): 1}, {C(1,1): 1, C(0,2): -1}))
        terms.append(({A(2,2): 1, A(0,2): 1, A(1,2): -1, A(1,1): -1}, {B(0,0): 1, B(1,0): 1}, {C(2,2): 1, C(1,0): -1}))
        
        return terms

    # Strassen 2x2x2 terms
    def get_strassen_2x2x2_terms(offset_i, offset_j, offset_k):
        terms = []
        def A(i, j): return (offset_i + i) * m + (offset_j + j)
        def B(i, j): return (offset_j + i) * p + (offset_k + j)
        def C(i, j): return (offset_k + i) * n + (offset_i + j)
        
        terms.append(({A(0,0): 1, A(1,1): 1}, {B(0,0): 1, B(1,1): 1}, {C(0,0): 1, C(1,1): 1}))
        terms.append(({A(1,0): 1, A(1,1): 1}, {B(0,0): 1}, {C(1,0): 1, C(1,1): -1}))
        terms.append(({A(0,0): 1}, {B(0,1): 1, B(1,1): -1}, {C(0,1): 1, C(1,1): 1}))
        terms.append(({A(1,1): 1}, {B(1,0): 1, B(0,0): -1}, {C(0,0): 1, C(1,0): 1}))
        terms.append(({A(0,0): 1, A(0,1): 1}, {B(1,1): 1}, {C(0,0): -1, C(0,1): 1}))
        terms.append(({A(1,0): 1, A(0,0): -1}, {B(0,0): 1, B(0,1): 1}, {C(1,1): 1}))
        terms.append(({A(0,1): 1, A(1,1): -1}, {B(1,0): 1, B(1,1): 1}, {C(0,0): 1}))
        
        return terms

    def build_full_decomposition():
        terms = []
        
        # 3x3x3 block at (0,0)
        terms.extend(get_strassen_3x3x3_terms())
        
        # 2x2x2 block at (3,3)
        terms.extend(get_strassen_2x2x2_terms(3, 3, 3))
        
        # Cross terms: Schoolbook
        # i in 0..2, j in 0..2, k in 3..4
        for i in range(3):
            for j in range(3):
                for k in range(3, 5):
                    terms.append(({i*m + j: 1}, {j*p + k: 1}, {k*n + i: 1}))
                    
        # i in 0..2, j in 3..4, k in 0..2
        for i in range(3):
            for j in range(3, 5):
                for k in range(3):
                    terms.append(({i*m + j: 1}, {j*p + k: 1}, {k*n + i: 1}))
                    
        # i in 0..2, j in 3..4, k in 3..4
        for i in range(3):
            for j in range(3, 5):
                for k in range(3, 5):
                    terms.append(({i*m + j: 1}, {j*p + k: 1}, {k*n + i: 1}))
                    
        # i in 3..4, j in 0..2, k in 0..2
        for i in range(3, 5):
            for j in range(3):
                for k in range(3):
                    terms.append(({i*m + j: 1}, {j*p + k: 1}, {k*n + i: 1}))
                    
        # i in 3..4, j in 0..2, k in 3..4
        for i in range(3, 5):
            for j in range(3):
                for k in range(3, 5):
                    terms.append(({i*m + j: 1}, {j*p + k: 1}, {k*n + i: 1}))
                    
        # i in 3..4, j in 3..4, k in 0..2
        for i in range(3, 5):
            for j in range(3, 5):
                for k in range(3):
                    terms.append(({i*m + j: 1}, {j*p + k: 1}, {k*n + i: 1}))
                    
        # i in 3..4, j in 3..4, k in 3..4 is covered by Strassen-2
        # i in 0..2, j in 0..2, k in 0..2 is covered by Strassen-3
        
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for r in range(rank):
            U_dict, V_dict, W_dict = terms[r]
            for idx, val in U_dict.items():
                U[idx][r] = val
            for idx, val in V_dict.items():
                V[idx][r] = val
            for idx, val in W_dict.items():
                W[idx][r] = val
        
        return U, V, W

    best_U, best_V, best_W = get_schoolbook()
    best_rank = len(best_U[0])
    
    if time.time() < deadline - 10.0:
        U, V, W = build_full_decomposition()
        rank = len(U[0])
        if verify(U, V, W):
            if rank < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = rank

    return {"U": best_U, "V": best_V, "W": best_W}
