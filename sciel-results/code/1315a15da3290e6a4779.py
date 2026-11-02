"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 1 Strassen 2x2x2 (j=0,1; k=0,1) -> 7
    # 14 Schoolbook (j=2; all k) -> 14
    # 1 Strassen 2x2x2 (j=0,1; k=2,3) -> 7
    # 1 Strassen 2x2x2 (j=0,1; k=4,5) -> 7
    # 4 Schoolbook (j=0,1; k=6) -> 4
    # Total: 39
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    # i_offset, j_offset, k_offset
    # Local indices i,j,k in 0..1
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        U[i_off*m + 0][col] = 1
        U[i_off*m + 1][col] = 1
        V[j_off*p + 0][col] = 1
        V[j_off*p + 1][col] = 1
        W[(k_off+0)*n + i_off] = 1
        W[(k_off+1)*n + (i_off+1)] = 1
        col += 1
        
        # M2 = (A10+A11)B00 -> C10
        U[i_off*m + 1][col] = 1
        U[(i_off+1)*m + 1][col] = 1
        V[j_off*p + 0][col] = 1
        W[(k_off+0)*n + (i_off+1)] = 1
        col += 1
        
        # M3 = A00B10 -> C01
        U[i_off*m + 0][col] = 1
        V[j_off*p + 1][col] = 1
        W[(k_off+1)*n + i_off] = 1
        col += 1
        
        # M4 = A11B11 -> C11, C10
        U[(i_off+1)*m + 1][col] = 1
        V[j_off*p + 1][col] = 1
        W[(k_off+1)*n + (i_off+1)] = 1
        W[(k_off+0)*n + (i_off+1)] = 1
        col += 1
        
        # M5 = (A00+A01)(B00+B01) -> C00, C01
        U[i_off*m + 0][col] = 1
        U[i_off*m + 1][col] = 1
        V[j_off*p + 0][col] = 1
        V[j_off*p + 1][col] = 1
        W[(k_off+0)*n + i_off] = 1
        W[(k_off+1)*n + i_off] = 1
        col += 1
        
        # M6 = (A10-A00)(B01+B11) -> C11
        U[i_off*m + 1][col] = 1
        U[i_off*m + 0][col] = -1
        V[j_off*p + 1][col] = 1
        V[j_off*p + 1][col] = 1 # Wait, B01 and B11?
        # M6 = (a10 - a00)(b01 + b11)
        # V index for b01 is j=0, k=1. V[j_off*p + 1]
        # V index for b11 is j=1, k=1. V[(j_off+1)*p + 1]
        # My previous line overwrote V[j_off*p+1].
        # Let's fix M6.
        V[j_off*p + 1][col] = 1
        V[(j_off+1)*p + 1][col] = 1
        W[(k_off+1)*n + (i_off+1)] = 1
        col += 1
        
        # M7 = (A01+A11)(B10-B00) -> C00, C11
        U[i_off*m + 1][col] = 1
        U[(i_off+1)*m + 1][col] = 1
        V[j_off*p + 1][col] = 1
        V[j_off*p + 0][col] = -1
        W[(k_off+0)*n + i_off] = 1
        W[(k_off+1)*n + (i_off+1)] = 1
        col += 1

    # Block 1: j=0,1; k=0,1
    add_strassen(0, 0, 0)
    
    # Block 2: j=0,1; k=2,3
    add_strassen(0, 0, 2)
    
    # Block 3: j=0,1; k=4,5
    add_strassen(0, 0, 4)
    
    # Schoolbook: j=2; all k
    for i in range(n):
        for k in range(p):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # Schoolbook: j=0,1; k=6
    for i in range(n):
        for j in range(2):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
