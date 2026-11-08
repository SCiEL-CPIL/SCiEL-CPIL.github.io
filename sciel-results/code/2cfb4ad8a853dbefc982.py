import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2 (rows of A), m=4 (inner), p=7 (cols of B)
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin out of 150s
    
    # Target Rank: 50
    # Decomposition:
    # Split inner dimension m=4 into two blocks of size 2: J1={0,1}, J2={2,3}
    # Each block is a 2x2x7 multiplication.
    # For 2x2x7, split p=7 into 2+2+2+1.
    # Three 2x2x2 blocks (Strassen, rank 7) and one 2x2x1 block (Schoolbook, rank 4).
    # Total per 2x2x7 block: 3*7 + 4 = 25.
    # Total for 2x4x7: 2 * 25 = 50.
    
    rank = 50
    # U: shape (n*m, rank) = (8, 50)
    # V: shape (m*p, rank) = (28, 50)
    # W: shape (p*n, rank) = (14, 50)
    # Using lists for speed and simplicity, initialized to 0
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        """
        Adds a 2x2x2 Strassen block.
        A block: rows i0, i0+1; cols j0, j0+1.
        B block: rows j0, j0+1; cols k0, k0+1.
        C block: rows i0, i0+1; cols k0, k0+1.
        """
        nonlocal col
        if col >= rank:
            return
            
        # Indices in U (A): i*m + j
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # Indices in V (B): j*p + k
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        # Indices in W (C): k*n + i
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h) -> C00 += M1, C11 += M1
        U[a_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1; V[h_idx][col] += 1
        W[c00_idx][col] += 1; W[c11_idx][col] += 1
        col += 1
        
        # M2 = (c+d)e -> C10 += M2, C11 -= M2
        U[c_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1; W[c11_idx][col] -= 1
        col += 1
        
        # M3 = a(f-h) -> C01 += M3, C11 += M3
        U[a_idx][col] += 1
        V[f_idx][col] += 1; V[h_idx][col] -= 1
        W[c01_idx][col] += 1; W[c11_idx][col] += 1
        col += 1
        
        # M4 = d(g-e) -> C00 += M4, C10 += M4
        U[d_idx][col] += 1
        V[g_idx][col] += 1; V[e_idx][col] -= 1
        W[c00_idx][col] += 1; W[c10_idx][col] += 1
        col += 1
        
        # M5 = (a+b)h -> C00 -= M5, C01 += M5
        U[a_idx][col] += 1; U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1; W[c01_idx][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> C11 += M6
        U[c_idx][col] += 1; U[a_idx][col] -= 1
        V[e_idx][col] += 1; V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> C00 += M7
        U[b_idx][col] += 1; U[d_idx][col] -= 1
        V[g_idx][col] += 1; V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    def add_schoolbook_block(i0, j0, k0):
        """
        Adds a 2x2x1 schoolbook block.
        A block: rows i0, i0+1; cols j0, j0+1.
        B block: rows j0, j0+1; col k0.
        C block: rows i0, i0+1; col k0.
        """
        nonlocal col
        if col >= rank:
            return

        # Indices in U (A): i*m + j
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # Indices in V (B): j*p + k
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        # Indices in W (C): k*n + i
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        # P1 = a*e -> C00 += P1
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        # P2 = b*g -> C00 += P2
        U[b_idx][col] += 1
        V[g_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        # P3 = c*e -> C10 += P3
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # P4 = d*g -> C10 += P4
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

    # We have two j-blocks: j_start in {0, 2}
    # For each j-block, we have 7 k-columns.
    # We group them as: (0,1), (2,3), (4,5) -> Strassen, (6) -> Schoolbook
    
    for j_start in [0, 2]:
        if time.time() > deadline:
            break
        # Strassen blocks for k pairs
        add_strassen_block(0, j_start, 0)
        add_strassen_block(0, j_start, 2)
        add_strassen_block(0, j_start, 4)
        
        if time.time() > deadline:
            break
        # Schoolbook block for k=6
        add_schoolbook_block(0, j_start, 6)

    return {"U": U, "V": V, "W": W}
