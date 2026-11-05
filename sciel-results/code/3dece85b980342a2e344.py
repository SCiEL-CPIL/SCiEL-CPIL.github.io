import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Partition sizes
    n_sizes = [2, 1]
    m_sizes = [2, 2, 1]
    p_sizes = [2, 2, 2]
    
    # Calculate total rank
    total_rank = 0
    for ni in n_sizes:
        for mj in m_sizes:
            for pk in p_sizes:
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
                    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a scalar term to a specific rank slot
    def add_term(u_idx, v_idx, w_idx, val):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    # Helper to add a Strassen 2x2x2 block
    def add_strassen_block(i_start, j_start, k_start):
        nonlocal col
        # Local indices
        # A is 2x2, B is 2x2, C is 2x2
        # A indices: (i_start, j_start) to (i_start+1, j_start+1)
        # B indices: (j_start, k_start) to (j_start+1, k_start+1)
        # C indices: (i_start, k_start) to (i_start+1, k_start+1)
        
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        A00_u = (i_start) * m + (j_start)
        A01_u = (i_start) * m + (j_start + 1)
        A10_u = (i_start + 1) * m + (j_start)
        A11_u = (i_start + 1) * m + (j_start + 1)
        
        B00_v = (j_start) * p + (k_start)
        B01_v = (j_start) * p + (k_start + 1)
        B10_v = (j_start + 1) * p + (k_start)
        B11_v = (j_start + 1) * p + (k_start + 1)
        
        C00_w = (k_start) * n + (i_start)
        C01_w = (k_start + 1) * n + (i_start)
        C10_w = (k_start) * n + (i_start + 1)
        C11_w = (k_start + 1) * n + (i_start + 1)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[A00_u][col] = 1
        U[A11_u][col] = 1
        V[B00_v][col] = 1
        V[B11_v][col] = 1
        W[C00_w][col] = 1
        W[C11_w][col] = -1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[A10_u][col] = 1
        U[A11_u][col] = 1
        V[B00_v][col] = 1
        W[C10_w][col] = 1
        W[C11_w][col] = -1
        col += 1
        
        # M3 = A00(B10 - B11)
        U[A00_u][col] = 1
        V[B10_v][col] = 1
        V[B11_v][col] = -1
        W[C01_w][col] = 1
        W[C10_w][col] = 1
        col += 1
        
        # M4 = A11(B01 - B00)
        U[A11_u][col] = 1
        V[B01_v][col] = 1
        V[B00_v][col] = -1
        W[C00_w][col] = 1
        W[C10_w][col] = 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[A00_u][col] = 1
        U[A01_u][col] = 1
        V[B11_v][col] = 1
        W[C00_w][col] = -1
        W[C01_w][col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[A10_u][col] = 1
        U[A00_u][col] = -1
        V[B00_v][col] = 1
        V[B01_v][col] = 1
        W[C11_w][col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[A01_u][col] = 1
        U[A11_u][col] = -1
        V[B10_v][col] = 1
        V[B11_v][col] = 1
        W[C00_w][col] = 1
        col += 1
        
    # Iterate through blocks
    i_start = 0
    for ni in n_sizes:
        j_start = 0
        for mj in m_sizes:
            k_start = 0
            for pk in p_sizes:
                if ni == 2 and mj == 2 and pk == 2:
                    add_strassen_block(i_start, j_start, k_start)
                else:
                    # Schoolbook for this block
                    for ii in range(ni):
                        for jj in range(mj):
                            for kk in range(pk):
                                u_idx = (i_start + ii) * m + (j_start + jj)
                                v_idx = (j_start + jj) * p + (k_start + kk)
                                w_idx = (k_start + kk) * n + (i_start + ii)
                                add_term(u_idx, v_idx, w_idx, 1)
                                col += 1
                k_start += pk
            j_start += mj
        i_start += ni
        
    return {"U": U, "V": V, "W": W}
