"""Matrix multiplication 2x3x6 rank-minimal decomposition."""
import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = 2, 3, 6
    rank = 33
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # Strassen for three 2x2x2 blocks: j in {0,1}, k_offset in {0,2,4}
    for k_off in [0, 2, 4]:
        if time.time() > deadline:
            break
        # M1 = (A11+A22)(B11+B22)
        U[0][col] = 1; U[4][col] = 1
        V[k_off][col] = 1; V[6+k_off+1][col] = 1
        W[k_off*2+0][col] = 1
        W[(k_off+1)*2+0][col] = 1
        W[k_off*2+1][col] = 1
        W[(k_off+1)*2+1][col] = 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[3][col] = 1; U[4][col] = 1
        V[k_off][col] = 1
        W[k_off*2+1][col] = 1
        W[(k_off+1)*2+1][col] = 1
        col += 1
        
        # M3 = A11(B21-B22)
        U[0][col] = 1
        V[6+k_off][col] = 1; V[6+k_off+1][col] = -1
        W[k_off*2+0][col] = 1
        W[(k_off+1)*2+0][col] = 1
        W[(k_off+1)*2+1][col] = -1
        col += 1
        
        # M4 = A22(B12-B11)
        U[4][col] = 1
        V[k_off+1][col] = 1; V[k_off][col] = -1
        W[k_off*2+0][col] = -1
        W[k_off*2+1][col] = 1
        W[(k_off+1)*2+1][col] = 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[0][col] = 1; U[1][col] = 1
        V[6+k_off+1][col] = 1
        W[k_off*2+0][col] = 1
        W[(k_off+1)*2+0][col] = 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[3][col] = 1; U[0][col] = -1
        V[k_off][col] = 1; V[k_off+1][col] = 1
        W[(k_off+1)*2+1][col] = 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1][col] = 1; U[4][col] = -1
        V[6+k_off][col] = 1; V[6+k_off+1][col] = 1
        W[k_off*2+0][col] = 1
        col += 1
    
    # Schoolbook for 2x1x6: i in {0,1}, j=2, k in {0..5}
    for i in range(2):
        for k in range(6):
            if time.time() > deadline:
                break
            U[i*3+2][col] = 1
            V[12+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
