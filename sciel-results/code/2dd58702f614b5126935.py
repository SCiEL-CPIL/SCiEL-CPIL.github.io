import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U[(i_base+0)*m + (j_base+0)][col] += 1
        U[(i_base+1)*m + (j_base+1)][col] += 1
        V[(j_base+0)*p + (k_base+0)][col] += 1
        V[(j_base+1)*p + (k_base+1)][col] += 1
        W[(k_base+0)*n + (i_base+0)][col] += 1
        W[(k_base+1)*n + (i_base+1)][col] += 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[(i_base+1)*m + (j_base+0)][col] += 1
        U[(i_base+1)*m + (j_base+1)][col] += 1
        V[(j_base+0)*p + (k_base+0)][col] += 1
        W[(k_base+0)*n + (i_base+1)][col] += 1
        W[(k_base+1)*n + (i_base+1)][col] += 1
        col += 1
        
        # M3 = A00(B10 - B11)
        U[(i_base+0)*m + (j_base+0)][col] += 1
        V[(j_base+1)*p + (k_base+0)][col] += 1
        V[(j_base+1)*p + (k_base+1)][col] -= 1
        W[(k_base+0)*n + (i_base+0)][col] += 1
        W[(k_base+1)*n + (i_base+0)][col] += 1
        col += 1
        
        # M4 = A11 B11
        U[(i_base+1)*m + (j_base+1)][col] += 1
        V[(j_base+1)*p + (k_base+1)][col] += 1
        W[(k_base+1)*n + (i_base+1)][col] += 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[(i_base+0)*m + (j_base+0)][col] += 1
        U[(i_base+0)*m + (j_base+1)][col] += 1
        V[(j_base+1)*p + (k_base+1)][col] += 1
        W[(k_base+0)*n + (i_base+0)][col] -= 1
        W[(k_base+1)*n + (i_base+0)][col] += 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B10)
        U[(i_base+1)*m + (j_base+0)][col] += 1
        U[(i_base+0)*m + (j_base+0)][col] -= 1
        V[(j_base+0)*p + (k_base+0)][col] += 1
        V[(j_base+0)*p + (k_base+1)][col] += 1
        W[(k_base+1)*n + (i_base+1)][col] += 1
        col += 1
        
        # M7 = (A01 - A11)(B11 + B01)
        U[(i_base+0)*m + (j_base+1)][col] += 1
        U[(i_base+1)*m + (j_base+1)][col] -= 1
        V[(j_base+1)*p + (k_base+1)][col] += 1
        V[(j_base+0)*p + (k_base+1)][col] += 1
        W[(k_base+0)*n + (i_base+0)][col] += 1
        W[(k_base+1)*n + (i_base+1)][col] -= 1
        col += 1

    # Part 1a: i in {0,1}, j in {0,1}, k in {0..7}
    for k_block in range(4):
        k_base = k_block * 2
        add_strassen(0, 0, k_base)
        
    # Part 1b: i in {0,1}, j in {2,3}, k in {0..7}
    for k_block in range(4):
        k_base = k_block * 2
        add_strassen(0, 2, k_base)
        
    # Part 2: i in {2}, j in {0,1,2,3}, k in {0..7}
    # 1 x 4 x 8 schoolbook
    for j in range(4):
        for k in range(8):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
