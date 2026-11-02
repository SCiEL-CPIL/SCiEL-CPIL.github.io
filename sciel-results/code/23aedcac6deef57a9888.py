import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1
        
    # Helper for 2x2x2 Strassen
    def add_strassen(i_off, j_off, k_off):
        # i_off: offset in i (0..n-1)
        # j_off: offset in j (0..m-1)
        # k_off: offset in k (0..p-1)
        # Local indices: i in 0..1, j in 0..1, k in 0..1
        # U index: (i_off+i)*m + (j_off+j)
        # V index: (j_off+j)*p + (k_off+k)
        # W index: (k_off+k)*n + (i_off+i)
        
        # M1
        # (A11+A22)(B11+B22)
        add_term((i_off+0)*m + (j_off+0), 1, (j_off+0)*p + (k_off+0), 1, (k_off+0)*n + (i_off+0), 1)
        add_term((i_off+0)*m + (j_off+0), 1, (j_off+0)*p + (k_off+1), 1, (k_off+1)*n + (i_off+0), 1)
        add_term((i_off+1)*m + (j_off+1), 1, (j_off+0)*p + (k_off+0), 1, (k_off+0)*n + (i_off+1), 1)
        add_term((i_off+1)*m + (j_off+1), 1, (j_off+0)*p + (k_off+1), 1, (k_off+1)*n + (i_off+1), 1)
        
        # M2
        # (A21+A22)B11
        add_term((i_off+1)*m + (j_off+0), 1, (j_off+0)*p + (k_off+0), 1, (k_off+0)*n + (i_off+1), 1)
        add_term((i_off+1)*m + (j_off+1), 1, (j_off+0)*p + (k_off+0), 1, (k_off+0)*n + (i_off+1), 1)
        
        # M3
        # A11(B12-B22)
        add_term((i_off+0)*m + (j_off+0), 1, (j_off+0)*p + (k_off+1), 1, (k_off+1)*n + (i_off+0), 1)
        add_term((i_off+0)*m + (j_off+0), 1, (j_off+1)*p + (k_off+1), -1, (k_off+1)*n + (i_off+0), 1)
        
        # M4
        # A22(B21-B11)
        add_term((i_off+1)*m + (j_off+1), 1, (j_off+1)*p + (k_off+0), 1, (k_off+0)*n + (i_off+1), 1)
        add_term((i_off+1)*m + (j_off+1), 1, (j_off+0)*p + (k_off+0), -1, (k_off+0)*n + (i_off+1), 1)
        
        # M5
        # (A11+A12)B22
        add_term((i_off+0)*m + (j_off+0), 1, (j_off+1)*p + (k_off+1), 1, (k_off+1)*n + (i_off+0), 1)
        add_term((i_off+0)*m + (j_off+1), 1, (j_off+1)*p + (k_off+1), 1, (k_off+1)*n + (i_off+0), 1)
        
        # M6
        # (A21-A11)(B11+B12)
        add_term((i_off+1)*m + (j_off+0), 1, (j_off+0)*p + (k_off+0), 1, (k_off+0)*n + (i_off+1), 1)
        add_term((i_off+1)*m + (j_off+0), 1, (j_off+0)*p + (k_off+1), 1, (k_off+1)*n + (i_off+1), 1)
        add_term((i_off+0)*m + (j_off+0), -1, (j_off+0)*p + (k_off+0), 1, (k_off+0)*n + (i_off+0), 1)
        add_term((i_off+0)*m + (j_off+0), -1, (j_off+0)*p + (k_off+1), 1, (k_off+1)*n + (i_off+0), 1)
        
        # M7
        # (A12-A22)(B21+B22)
        add_term((i_off+0)*m + (j_off+1), 1, (j_off+1)*p + (k_off+0), 1, (k_off+0)*n + (i_off+0), 1)
        add_term((i_off+0)*m + (j_off+1), 1, (j_off+1)*p + (k_off+1), 1, (k_off+1)*n + (i_off+0), 1)
        add_term((i_off+1)*m + (j_off+1), -1, (j_off+1)*p + (k_off+0), 1, (k_off+0)*n + (i_off+1), 1)
        add_term((i_off+1)*m + (j_off+1), -1, (j_off+1)*p + (k_off+1), 1, (k_off+1)*n + (i_off+1), 1)

    # Helper for 2x3x2
    def add_232(i_off, j_off, k_off):
        # i_off, j_off, k_off
        # A is 2x3, B is 3x2
        # Split j into 1+2.
        # Part 1: A1 (2x1) B1 (1x2). Rank 4.
        # A1 indices: (i_off, j_off), (i_off+1, j_off)
        # B1 indices: (j_off, k_off), (j_off, k_off+1)
        
        # C00 = A00 B00
        add_term((i_off+0)*m + (j_off+0), 1, (j_off+0)*p + (k_off+0), 1, (k_off+0)*n + (i_off+0), 1)
        # C01 = A00 B01
        add_term((i_off+0)*m + (j_off+0), 1, (j_off+0)*p + (k_off+1), 1, (k_off+1)*n + (i_off+0), 1)
        # C10 = A10 B00
        add_term((i_off+1)*m + (j_off+0), 1, (j_off+0)*p + (k_off+0), 1, (k_off+0)*n + (i_off+1), 1)
        # C11 = A10 B01
        add_term((i_off+1)*m + (j_off+0), 1, (j_off+0)*p + (k_off+1), 1, (k_off+1)*n + (i_off+1), 1)
        
        # Part 2: A2 (2x2) B2 (2x2). Rank 7.
        # A2 indices: j_off+1, j_off+2
        # B2 indices: j_off+1, j_off+2
        add_strassen(i_off, j_off+1, k_off)

    # Block 1: A1 (2x2) B1 (2x2) at i=0, j=0, k=0
    add_strassen(0, 0, 0)
    
    # Block 2: A1 (2x2) B2 (2x4) at i=0, j=0, k=2..5
    # B2 is 2x4. Split into two 2x2.
    add_strassen(0, 0, 2)
    add_strassen(0, 0, 4)
    
    # Block 3: A2 (2x3) B3 (3x2) at i=0, j=2, k=0
    add_232(0, 2, 0)
    
    # Block 4: A2 (2x3) B4 (3x4) at i=0, j=2, k=2..5
    # B4 is 3x4. Split into two 3x2.
    add_232(0, 2, 2)
    add_232(0, 2, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
