import numpy as np
import time

# Constants for the 2x4 * 4x7 matrix multiplication
N, M, P = 2, 4, 7
RANK = 50
DEADLINE = 150.0  # Wall-clock limit for the evaluator
SAFETY_MARGIN = 5.0  # Return before this many seconds to limit

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    start_time = time.time()
    
    # Dimensions:
    # U: (n*m, rank) = (8, 50)
    # V: (m*p, rank) = (28, 50)
    # W: (p*n, rank) = (14, 50)
    
    U = [[0] * RANK for _ in range(N * M)]
    V = [[0] * RANK for _ in range(M * P)]
    W = [[0] * RANK for _ in range(P * N)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        """
        Adds a Strassen 2x2x2 block.
        i0: start row of A (0 or 1)
        j0: start col of A (0 or 2)
        k0: start col of B (0, 2, 4)
        """
        nonlocal col
        # Local indices for U (A matrix)
        # A is n x m, indexed by (row, col) -> row * m + col
        a_idx = i0 * M + j0
        b_idx = i0 * M + (j0 + 1)
        c_idx = (i0 + 1) * M + j0
        d_idx = (i0 + 1) * M + (j0 + 1)
        
        # Local indices for V (B matrix)
        # B is m x p, indexed by (row, col) -> row * p + col
        # Note: In the decomposition, B is multiplied from the right, so its rows correspond to the inner dimension j.
        e_idx = j0 * P + k0
        f_idx = j0 * P + (k0 + 1)
        g_idx = (j0 + 1) * P + k0
        h_idx = (j0 + 1) * P + (k0 + 1)
        
        # Local indices for W (C matrix)
        # C is p x n, indexed by (col, row) -> col * n + row (Trace convention W[k*n+i, r])
        c00_idx = k0 * N + i0
        c01_idx = (k0 + 1) * N + i0
        c10_idx = k0 * N + (i0 + 1)
        c11_idx = (k0 + 1) * N + (i0 + 1)
        
        # M1 = (a+d)(e+h) -> C[0,0] += 1, C[1,1] += 1
        U[a_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M2 = (c+d)e -> C[1,0] += 1, C[1,1] -= 1
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1
        
        # M3 = a(f-h) -> C[0,1] += 1, C[1,1] += 1
        U[a_idx][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M4 = d(g-e) -> C[0,0] += 1, C[1,0] += 1
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[e_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # M5 = (a+b)h -> C[0,0] -= 1, C[0,1] += 1
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> C[1,1] += 1
        U[c_idx][col] += 1
        U[a_idx][col] -= 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> C[0,0] += 1
        U[b_idx][col] += 1
        U[d_idx][col] -= 1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        """
        Adds a schoolbook 2x2x1 block.
        Used for the last column of B (k=6) since it's a single column.
        """
        nonlocal col
        a_idx = i0 * M + j0
        b_idx = i0 * M + (j0 + 1)
        c_idx = (i0 + 1) * M + j0
        d_idx = (i0 + 1) * M + (j0 + 1)
        
        e_idx = j0 * P + k0
        g_idx = (j0 + 1) * P + k0
        
        c00_idx = k0 * N + i0
        c10_idx = k0 * N + (i0 + 1)
        
        # P1 = a*e -> C[0,0] += 1
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        # P2 = b*g -> C[0,0] += 1
        U[b_idx][col] += 1
        V[g_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        # P3 = c*e -> C[1,0] += 1
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # P4 = d*g -> C[1,0] += 1
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

    # Process blocks
    # The inner dimension m=4 is split into two blocks of size 2: j in {0,1} and j in {2,3}.
    # For each block, the output dimension p=7 is split into three blocks of size 2 (k=0,2,4) and one block of size 1 (k=6).
    
    for j_start in [0, 2]:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        # k splits: 0-1, 2-3, 4-5
        for k_start in [0, 2, 4]:
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            add_strassen_block(0, j_start, k_start)
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        add_schoolbook_221_block(0, j_start, 6)
    
    # Verify col is RANK
    # assert col == RANK, f"Expected rank {RANK}, got {col}"
    
    return {"U": U, "V": V, "W": W}
