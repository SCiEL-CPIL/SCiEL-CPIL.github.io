"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is 90. We aim for 84.
    # Blocks:
    # I1=[0,1], I2=[2]
    # J1=[0,1], J2=[2,3], J3=[4]
    # K1=[0,1], K2=[2,3], K3=[4,5]
    
    U = [[0] * 84 for _ in range(n * m)]
    V = [[0] * 84 for _ in range(m * p)]
    W = [[0] * 84 for _ in range(p * n)]
    
    col = 0
    
    # Helper to add schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
        
    # Helper to add Strassen block
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # i_off, j_off, k_off are top-left indices of the 2x2x2 block
        # Local indices 0,1 map to global i_off, i_off+1 etc.
        
        # M1
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # M2
        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = -1
        col += 1
        
        # M3
        U[i_off * m + j_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = -1
        W[(k_off+1) * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # M4
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        V[j_off * p + k_off][col] = -1
        W[k_off * n + i_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        col += 1
        
        # M5
        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = -1
        W[(k_off+1) * n + i_off][col] = 1
        col += 1
        
        # M6
        U[(i_off+1) * m + j_off][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # M7
        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        col += 1

    # Iterate through blocks
    # I1=[0,1], I2=[2]
    # J1=[0,1], J2=[2,3], J3=[4]
    # K1=[0,1], K2=[2,3], K3=[4,5]
    
    # Block I1, K1
    add_strassen(0, 0, 0) # J1
    add_strassen(0, 2, 0) # J2
    for i in [0, 1]:
        for k in [0, 1]:
            add_schoolbook(i, 4, k) # J3
            
    # Block I1, K2
    add_strassen(0, 0, 2) # J1
    add_strassen(0, 2, 2) # J2
    for i in [0, 1]:
        for k in [2, 3]:
            add_schoolbook(i, 4, k) # J3
            
    # Block I1, K3
    add_strassen(0, 0, 4) # J1
    add_strassen(0, 2, 4) # J2
    for i in [0, 1]:
        for k in [4, 5]:
            add_schoolbook(i, 4, k) # J3
            
    # Block I2, K1
    for i in [2]:
        for j in [0, 1]:
            for k in [0, 1]:
                add_schoolbook(i, j, k) # J1
        for j in [2, 3]:
            for k in [0, 1]:
                add_schoolbook(i, j, k) # J2
        for k in [0, 1]:
            add_schoolbook(i, 4, k) # J3
            
    # Block I2, K2
    for i in [2]:
        for j in [0, 1]:
            for k in [2, 3]:
                add_schoolbook(i, j, k) # J1
        for j in [2, 3]:
            for k in [2, 3]:
                add_schoolbook(i, j, k) # J2
        for k in [2, 3]:
            add_schoolbook(i, 4, k) # J3
            
    # Block I2, K3
    for i in [2]:
        for j in [0, 1]:
            for k in [4, 5]:
                add_schoolbook(i, j, k) # J1
        for j in [2, 3]:
            for k in [4, 5]:
                add_schoolbook(i, j, k) # J2
        for k in [4, 5]:
            add_schoolbook(i, 4, k) # J3
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
