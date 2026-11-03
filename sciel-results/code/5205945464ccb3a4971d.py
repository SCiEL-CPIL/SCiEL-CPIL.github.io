import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    rank = 26
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen(i_off, j_off, k_off, c):
        # m1
        U[i_off*m + j_off][c] = 1
        U[(i_off+1)*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + i_off][c] = 1
        W[(k_off+1)*n + (i_off+1)][c] = 1
        c += 1
        # m2
        U[(i_off+1)*m + j_off][c] = 1
        U[(i_off+1)*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1
        W[k_off*n + (i_off+1)][c] = 1
        W[(k_off+1)*n + (i_off+1)][c] = -1
        c += 1
        # m3
        U[i_off*m + j_off][c] = 1
        V[j_off*p + (k_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + i_off][c] = 1
        W[(k_off+1)*n + (i_off+1)][c] = 1
        c += 1
        # m4
        U[(i_off+1)*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + k_off][c] = 1
        V[j_off*p + k_off][c] = -1
        W[k_off*n + i_off][c] = 1
        W[k_off*n + (i_off+1)][c] = 1
        c += 1
        # m5
        U[i_off*m + j_off][c] = 1
        U[i_off*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + i_off][c] = -1
        W[(k_off+1)*n + i_off][c] = 1
        c += 1
        # m6
        U[(i_off+1)*m + j_off][c] = 1
        U[i_off*m + j_off][c] = -1
        V[j_off*p + k_off][c] = 1
        V[j_off*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + (i_off+1)][c] = 1
        c += 1
        # m7
        U[i_off*m + (j_off+1)][c] = 1
        U[(i_off+1)*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + k_off][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + i_off][c] = 1
        c += 1
        return c

    def add_school(i, j, k, c):
        U[i*m + j][c] = 1
        V[j*p + k][c] = 1
        W[k*n + i][c] = 1
        return c + 1

    c = 0
    # Part 1: k in {0, 1, 2}
    c = add_strassen(0, 0, 0, c)
    for i in range(n):
        for j in range(2):
            c = add_school(i, j, 2, c)
    for i in range(n):
        for k in range(3):
            c = add_school(i, 2, k, c)
            
    # Part 2: k in {3, 4}
    c = add_strassen(0, 0, 3, c)
    for i in range(n):
        for k in range(3, 5):
            c = add_school(i, 2, k, c)
            
    assert c == rank, f"Rank mismatch: {c}"
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if _verify_decomposition(result['U'], result['V'], result['W']):
        return result
        
    # Fallback
    rank_school = n * m * p
    U_s = np.zeros((n*m, rank_school))
    V_s = np.zeros((m*p, rank_school))
    W_s = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
