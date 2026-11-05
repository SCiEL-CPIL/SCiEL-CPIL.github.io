import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 54
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(i_start, j_start, jb_start, k_start):
        nonlocal col
        # Local indices:
        # A: (0,0), (0,1), (1,0), (1,1) -> Global (i_start+0, j_start+0) etc
        # B: (0,0), (0,1), (1,0), (1,1) -> Global (jb_start+0, k_start+0) etc
        # C: (0,0), (0,1), (1,0), (1,1) -> Global (i_start+0, k_start+0) etc
        
        # m1 = (a+d)(e+h)
        U[i_start*m + (j_start+0)][col] = 1
        U[i_start*m + (j_start+1)][col] = 0 # a is (0,0), d is (1,1)
        U[(i_start+1)*m + (j_start+1)][col] = 1
        V[jb_start*p + (k_start+0)][col] = 1
        V[(jb_start+1)*p + (k_start+1)][col] = 1
        W[(k_start+0)*n + i_start][col] = 1
        W[(k_start+1)*n + (i_start+1)][col] = -1
        col += 1
        
        # m2 = (c+d)e
        U[(i_start+1)*m + (j_start+0)][col] = 1
        U[(i_start+1)*m + (j_start+1)][col] = 1
        V[jb_start*p + (k_start+0)][col] = 1
        W[(k_start+0)*n + (i_start+1)][col] = 1
        W[(k_start+1)*n + (i_start+1)][col] = -1
        col += 1
        
        # m3 = a(f-h)
        U[i_start*m + (j_start+0)][col] = 1
        V[jb_start*p + (k_start+1)][col] = 1
        V[(jb_start+1)*p + (k_start+1)][col] = -1
        W[(k_start+1)*n + i_start][col] = 1
        W[(k_start+1)*n + (i_start+1)][col] = 1
        col += 1
        
        # m4 = d(g-e)
        U[(i_start+1)*m + (j_start+1)][col] = 1
        V[(jb_start+1)*p + (k_start+0)][col] = 1
        V[jb_start*p + (k_start+0)][col] = -1
        W[(k_start+0)*n + i_start][col] = 1
        W[(k_start+0)*n + (i_start+1)][col] = 1
        col += 1
        
        # m5 = (a+b)h
        U[i_start*m + (j_start+0)][col] = 1
        U[i_start*m + (j_start+1)][col] = 1
        V[(jb_start+1)*p + (k_start+1)][col] = 1
        W[(k_start+0)*n + i_start][col] = -1
        W[(k_start+1)*n + i_start][col] = 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[(i_start+1)*m + (j_start+0)][col] = 1
        U[i_start*m + (j_start+0)][col] = -1
        V[jb_start*p + (k_start+0)][col] = 1
        V[jb_start*p + (k_start+1)][col] = 1
        W[(k_start+1)*n + (i_start+1)][col] = 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[i_start*m + (j_start+1)][col] = 1
        U[(i_start+1)*m + (j_start+1)][col] = -1
        V[(jb_start+1)*p + (k_start+0)][col] = 1
        V[(jb_start+1)*p + (k_start+1)][col] = 1
        W[(k_start+0)*n + i_start][col] = 1
        W[(k_start+1)*n + (i_start+1)][col] = 1
        col += 1

    # Part 1: A_R B_R
    # A_R: i in {0,1}, j in {3,4}
    # B_R: j in {3,4}, k in {0..5}
    # Blocks:
    add_strassen(0, 3, 3, 0) # k 0,1
    add_strassen(0, 3, 3, 2) # k 2,3
    add_strassen(0, 3, 3, 4) # k 4,5
    
    # Part 2a: A_L,01 B_L,01
    # A_L,01: i in {0,1}, j in {0,1}
    # B_L,01: j in {0,1}, k in {0..5}
    # Blocks:
    add_strassen(0, 0, 0, 0) # k 0,1
    add_strassen(0, 0, 0, 2) # k 2,3
    add_strassen(0, 0, 0, 4) # k 4,5
    
    # Part 2b: A_L,2 B_L,2
    # A_L,2: i in {0,1}, j=2
    # B_L,2: j=2, k in {0..5}
    for i in range(2):
        for k in range(6):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
