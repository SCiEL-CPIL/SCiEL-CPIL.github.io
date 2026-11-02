import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen block
    # local i, j, k -> global i, j, k
    # j_offset, k_offset
    def add_strassen(j_off, k_off, col_start):
        # M1
        U[0*m + 0 + j_off][col_start+0] += 1
        U[1*m + 1 + j_off][col_start+0] += 1
        V[0*p + 0 + k_off][col_start+0] += 1
        V[1*p + 1 + k_off][col_start+0] += 1
        W[0*n + 0][col_start+0] += 1
        W[0*n + 1][col_start+0] += 1
        W[1*n + 0][col_start+0] += 1
        W[1*n + 1][col_start+0] += 1
        
        # M2
        U[1*m + 0 + j_off][col_start+1] += 1
        U[1*m + 1 + j_off][col_start+1] += 1
        V[0*p + 0 + k_off][col_start+1] += 1
        W[1*n + 0][col_start+1] += 1
        W[1*n + 1][col_start+1] += 1
        
        # M3
        U[0*m + 0 + j_off][col_start+2] += 1
        V[1*p + 1 + k_off][col_start+2] += 1
        V[0*p + 1 + k_off][col_start+2] -= 1
        W[1*n + 0][col_start+2] += 1
        W[1*n + 1][col_start+2] += 1
        
        # M4
        U[1*m + 1 + j_off][col_start+3] += 1
        V[1*p + 0 + k_off][col_start+3] += 1
        V[0*p + 0 + k_off][col_start+3] -= 1
        W[1*n + 0][col_start+3] += 1
        W[1*n + 1][col_start+3] += 1
        
        # M5
        U[0*m + 0 + j_off][col_start+4] += 1
        U[0*m + 1 + j_off][col_start+4] += 1
        V[1*p + 1 + k_off][col_start+4] += 1
        W[0*n + 0][col_start+4] += 1
        W[0*n + 1][col_start+4] += 1
        W[1*n + 0][col_start+4] += 1
        W[1*n + 1][col_start+4] += 1
        
        # M6
        U[0*m + 1 + j_off][col_start+5] += 1
        U[1*m + 1 + j_off][col_start+5] -= 1
        V[1*p + 0 + k_off][col_start+5] += 1
        V[1*p + 1 + k_off][col_start+5] += 1
        W[1*n + 0][col_start+5] += 1
        W[1*n + 1][col_start+5] += 1
        
        # M7
        U[1*m + 0 + j_off][col_start+6] += 1
        U[0*m + 0 + j_off][col_start+6] -= 1
        V[0*p + 0 + k_off][col_start+6] += 1
        V[0*p + 1 + k_off][col_start+6] += 1
        W[0*n + 0][col_start+6] += 1
        W[1*n + 0][col_start+6] += 1

    # Block 1: j=0,1 k=0,1
    add_strassen(0, 0, 0)
    
    # Block 2: j=0,1 k=2,3
    add_strassen(0, 2, 7)
    
    # Block 3: j=0,1 k=4 (Schoolbook 4)
    # C04 = A00 B04 + A01 B14
    # C14 = A10 B04 + A11 B14
    U[0*3+0][14] = 1
    V[0*5+4][14] = 1
    W[4*2+0][14] = 1
    
    U[0*3+1][15] = 1
    V[1*5+4][15] = 1
    W[4*2+0][15] = 1
    
    U[1*3+0][16] = 1
    V[0*5+4][16] = 1
    W[4*2+1][16] = 1
    
    U[1*3+1][17] = 1
    V[1*5+4][17] = 1
    W[4*2+1][17] = 1
    
    # Block 4: j=2 k=0..4 (Schoolbook 10)
    col = 18
    for i in range(2):
        for k in range(5):
            U[i*3+2][col] = 1
            V[2*5+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
