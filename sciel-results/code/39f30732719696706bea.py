import time
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = 150.0  # 150s limit, leave margin
    
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(i_off, j_off, k_off, c):
        nonlocal col
        # M1
        U[i_off*m + j_off][c] += 1
        U[(i_off+1)*m + (j_off+1)][c] += 1
        V[j_off*p + k_off][c] += 1
        V[(j_off+1)*p + (k_off+1)][c] += 1
        W[k_off*n + i_off][c] += 1
        W[(k_off+1)*n + i_off][c] += 1
        W[k_off*n + (i_off+1)][c] += 1
        W[(k_off+1)*n + (i_off+1)][c] += 1
        col += 1
        
        if time.time() > deadline: return
        
        # M2
        U[(i_off+1)*m + j_off][c] += 1
        U[(i_off+1)*m + (j_off+1)][c] += 1
        V[j_off*p + k_off][c] += 1
        W[k_off*n + (i_off+1)][c] += 1
        W[(k_off+1)*n + (i_off+1)][c] -= 1
        col += 1
        
        if time.time() > deadline: return
        
        # M3
        U[i_off*m + j_off][c] += 1
        V[(j_off+1)*p + k_off][c] += 1
        V[(j_off+1)*p + (k_off+1)][c] -= 1
        W[(k_off+1)*n + i_off][c] += 1
        W[(k_off+1)*n + (i_off+1)][c] += 1
        col += 1
        
        if time.time() > deadline: return
        
        # M4
        U[(i_off+1)*m + (j_off+1)][c] += 1
        V[(j_off+1)*p + k_off][c] += 1
        V[j_off*p + k_off][c] -= 1
        W[k_off*n + i_off][c] += 1
        W[k_off*n + (i_off+1)][c] += 1
        col += 1
        
        if time.time() > deadline: return
        
        # M5
        U[i_off*m + j_off][c] += 1
        U[i_off*m + (j_off+1)][c] += 1
        V[(j_off+1)*p + (k_off+1)][c] += 1
        W[k_off*n + i_off][c] -= 1
        W[(k_off+1)*n + i_off][c] += 1
        col += 1
        
        if time.time() > deadline: return
        
        # M6
        U[(i_off+1)*m + j_off][c] -= 1
        U[i_off*m + j_off][c] -= 1
        V[j_off*p + k_off][c] += 1
        V[j_off*p + (k_off+1)][c] += 1
        W[(k_off+1)*n + (i_off+1)][c] += 1
        col += 1
        
        if time.time() > deadline: return
        
        # M7
        U[i_off*m + (j_off+1)][c] -= 1
        U[(i_off+1)*m + (j_off+1)][c] -= 1
        V[(j_off+1)*p + k_off][c] += 1
        V[(j_off+1)*p + (k_off+1)][c] += 1
        W[k_off*n + i_off][c] += 1
        col += 1

    # Block 1: i in 0..1, j in 0..1, k in 0..1
    add_strassen(0, 0, 0, col)
    
    # Block 2: i in 0..1, j in 0..1, k in 2..3
    add_strassen(0, 0, 2, col)
    
    # Block 3: i in 0..1, j in 0..1, k in 4
    # Schoolbook 2x2x1
    # c_04 = a_00 b_04 + a_01 b_14
    # c_14 = a_10 b_04 + a_11 b_14
    
    # Term 1: a_00 b_04 -> c_04
    U[0*m + 0][col] = 1
    V[0*p + 4][col] = 1
    W[4*n + 0][col] = 1
    col += 1
    
    # Term 2: a_01 b_14 -> c_04
    U[0*m + 1][col] = 1
    V[1*p + 4][col] = 1
    W[4*n + 0][col] = 1
    col += 1
    
    # Term 3: a_10 b_04 -> c_14
    U[1*m + 0][col] = 1
    V[0*p + 4][col] = 1
    W[4*n + 1][col] = 1
    col += 1
    
    # Term 4: a_11 b_14 -> c_14
    U[1*m + 1][col] = 1
    V[1*p + 4][col] = 1
    W[4*n + 1][col] = 1
    col += 1
    
    # Block 4: i in 0..1, j in 2, k in 0..4
    # Schoolbook 2x1x5
    # c_ik = a_i2 b_2k
    
    for i in range(2):
        for k in range(5):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
