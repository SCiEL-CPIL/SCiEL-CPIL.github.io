"""Matrix multiplication 2,3,5 — rank-27 decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = 2, 3, 5
    rank = 27
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c = 0
    
    def add_strassen_block(j_off, k_off):
        nonlocal c
        U[0*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[(j_off)*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off)*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        U[1*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[(j_off)*p + k_off][c] = 1
        W[(k_off)*n + 1][c] = 1; W[(k_off+1)*n + 1][c] = -1
        c += 1
        U[0*m + j_off][c] = 1
        V[(j_off)*p + (k_off+1)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        U[1*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + k_off][c] = 1; V[(j_off)*p + k_off][c] = -1
        W[(k_off)*n + 0][c] = 1; W[(k_off)*n + 1][c] = 1
        c += 1
        U[0*m + j_off][c] = 1; U[0*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off)*n + 0][c] = -1; W[(k_off+1)*n + 0][c] = 1
        c += 1
        U[1*m + j_off][c] = 1; U[0*m + j_off][c] = -1
        V[(j_off)*p + k_off][c] = 1; V[(j_off)*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        U[0*m + (j_off+1)][c] = 1; U[1*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off)*n + 0][c] = 1
        c += 1

    add_strassen_block(0, 0)
    add_strassen_block(0, 2)
    
    # Merged term: covers (0,2,0) and (1,2,0)
    U[2][c] = 1; U[5][c] = 1
    V[10][c] = 1
    W[0][c] = 1; W[1][c] = 1
    c += 1
    
    # j in {0,1}, k=4
    U[0][c] = 1; V[4][c] = 1; W[8][c] = 1; c += 1
    U[3][c] = 1; V[9][c] = 1; W[8][c] = 1; c += 1
    U[1][c] = 1; V[4][c] = 1; W[9][c] = 1; c += 1
    U[4][c] = 1; V[9][c] = 1; W[9][c] = 1; c += 1
    
    # j=2, k in {1,2,3,4}
    U[2][c] = 1; V[11][c] = 1; W[2][c] = 1; c += 1
    U[5][c] = 1; V[11][c] = 1; W[3][c] = 1; c += 1
    U[2][c] = 1; V[12][c] = 1; W[4][c] = 1; c += 1
    U[5][c] = 1; V[12][c] = 1; W[5][c] = 1; c += 1
    U[2][c] = 1; V[13][c] = 1; W[6][c] = 1; c += 1
    U[5][c] = 1; V[13][c] = 1; W[7][c] = 1; c += 1
    U[2][c] = 1; V[14][c] = 1; W[8][c] = 1; c += 1
    U[5][c] = 1; V[14][c] = 1; W[9][c] = 1; c += 1
    
    assert c == 27, f"Expected 27, got {c}"
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                assert abs(total - 1.0) < 1e-10, f"Failed at ({i},{j},{k}): got {total}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
