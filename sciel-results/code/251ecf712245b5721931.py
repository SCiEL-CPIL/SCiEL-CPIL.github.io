import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        U[(i_off+0)*m+(j_off+0)][col] = 1
        U[(i_off+1)*m+(j_off+1)][col] = 1
        V[(j_off+0)*p+(k_off+0)][col] = 1
        V[(j_off+1)*p+(k_off+1)][col] = 1
        W[(k_off+0)*n+(i_off+0)][col] = 1
        W[(k_off+1)*n+(i_off+1)][col] = 1
        col += 1
        U[(i_off+1)*m+(j_off+0)][col] = 1
        U[(i_off+1)*m+(j_off+1)][col] = 1
        V[(j_off+0)*p+(k_off+0)][col] = 1
        W[(k_off+0)*n+(i_off+1)][col] = 1
        W[(k_off+1)*n+(i_off+1)][col] = -1
        col += 1
        U[(i_off+0)*m+(j_off+0)][col] = 1
        V[(j_off+0)*p+(k_off+1)][col] = 1
        V[(j_off+1)*p+(k_off+1)][col] = -1
        W[(k_off+1)*n+(i_off+0)][col] = 1
        W[(k_off+1)*n+(i_off+1)][col] = 1
        col += 1
        U[(i_off+1)*m+(j_off+1)][col] = 1
        V[(j_off+1)*p+(k_off+0)][col] = 1
        V[(j_off+0)*p+(k_off+0)][col] = -1
        W[(k_off+0)*n+(i_off+0)][col] = 1
        W[(k_off+0)*n+(i_off+1)][col] = 1
        col += 1
        U[(i_off+0)*m+(j_off+0)][col] = 1
        U[(i_off+0)*m+(j_off+1)][col] = 1
        V[(j_off+1)*p+(k_off+1)][col] = 1
        W[(k_off+0)*n+(i_off+0)][col] = -1
        W[(k_off+1)*n+(i_off+0)][col] = 1
        col += 1
        U[(i_off+1)*m+(j_off+0)][col] = 1
        U[(i_off+0)*m+(j_off+0)][col] = -1
        V[(j_off+0)*p+(k_off+0)][col] = 1
        V[(j_off+0)*p+(k_off+1)][col] = 1
        W[(k_off+1)*n+(i_off+1)][col] = 1
        col += 1
        U[(i_off+0)*m+(j_off+1)][col] = 1
        U[(i_off+1)*m+(j_off+1)][col] = -1
        V[(j_off+1)*p+(k_off+0)][col] = 1
        V[(j_off+1)*p+(k_off+1)][col] = 1
        W[(k_off+0)*n+(i_off+0)][col] = 1
        col += 1
    
    # Block 1: j=0 (Rank 10)
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2: j=1,2 (Rank 18)
    add_strassen(0, 1, 0)
    add_strassen(0, 1, 2)
    
    # k=4: Schoolbook (Rank 4)
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 120:
        pass  # Already done, just return
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
