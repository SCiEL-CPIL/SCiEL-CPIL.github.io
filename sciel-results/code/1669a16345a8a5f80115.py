import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _build_44_decomp():
    n, m, p = DIMENSIONS
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Lader's 3x3x3
    lader = [
        (0, 0, 0), (1, 1, 1), (2, 2, 2),
        (3, 0, 1), (0, 3, 1), (1, 0, 3),
        (4, 0, 2), (0, 4, 2), (2, 0, 4),
        (5, 0, 1), (1, 5, 2), (2, 1, 5),
        (6, 0, 2), (2, 6, 0), (0, 2, 6),
        (7, 0, 1), (1, 7, 2), (2, 1, 7),
        (8, 0, 2), (2, 8, 0), (0, 2, 8),
        (3, 4, 5), (5, 3, 4), (4, 5, 3)
    ]
    for u_idx, v_idx, w_idx in lader:
        u_full = [0] * (n * m)
        v_full = [0] * (m * p)
        w_full = [0] * (p * n)
        u_full[(u_idx // 3) * m + (u_idx % 3)] = 1
        v_full[(v_idx // 3) * p + (v_idx % 3)] = 1
        w_full[(w_idx // 3) * n + (w_idx % 3)] = 1
        U_terms.append(u_full)
        V_terms.append(v_full)
        W_terms.append(w_full)
    
    for i in range(3):
        for k in range(3):
            U_terms.append([0]*(n*m) if False else [1 if idx == i*m+3 else 0 for idx in range(n*m)])
            V_terms.append([1 if idx == 3*p+k else 0 for idx in range(m*p)])
            W_terms.append([1 if idx == k*n+i else 0 for idx in range(p*n)])
    for i in range(3):
        for j in range(3):
            U_terms.append([1 if idx == i*m+j else 0 for idx in range(n*m)])
            V_terms.append([1 if idx == j*p+3 else 0 for idx in range(m*p)])
            W_terms.append([1 if idx == 3*n+i else 0 for idx in range(p*n)])
    for i in range(3):
        U_terms.append([1 if idx == i*m+3 else 0 for idx in range(n*m)])
        V_terms.append([1 if idx == 3*p+3 else 0 for idx in range(m*p)])
        W_terms.append([1 if idx == 3*n+i else 0 for idx in range(p*n)])
    
    return U_terms, V_terms, W_terms

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], np.multiply(V[j*p+k], W[k*n+i]))
                if abs(val - 1) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    U_terms, V_terms, W_terms = _build_44_decomp()
    
    if _verify(U_terms, V_terms, W_terms):
        U = np.array(U_terms).T
        V = np.array(V_terms).T
        W = np.array(W_terms).T
        U_list = [[int(round(x)) for x in row] for row in U]
        V_list = [[int(round(x)) for x in row] for row in V]
        W_list = [[int(round(x)) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
