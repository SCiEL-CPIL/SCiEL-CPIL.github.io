import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by blocks.
    # Target Rank: 66
    
    # 1. Split m=4 into 2+2
    # A is 3x4 -> A1 (3x2), A2 (3x2)
    # B is 4x6 -> B1 (2x6), B2 (2x6)
    # C = A1 B1 + A2 B2
    
    # Each 3x2x6 block has rank 33.
    # Total rank 66.
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Define helper to add a 1xmxp block decomposition
    # U indices: i*m + j (i fixed to 0 for this sub-block? No, U is global)
    # We need to map local indices to global indices.
    
    def add_1xmxp(U, V, W, n_off, m_off, p_off, m_size, p_size, col_start):
        # n_off, m_off, p_off are offsets in the full matrices
        # m_size, p_size are the dimensions of this sub-block
        # This sub-block computes C_sub (1 x p_size) = A_sub (1 x m_size) * B_sub (m_size x p_size)
        # Rank = m_size * p_size
        r = m_size * p_size
        for j in range(m_size):
            for k in range(p_size):
                c = col_start + j * p_size + k
                
                # A index: (n_off + 0) * m + (m_off + j)
                # Wait, U shape is (n*m, rank).
                # The 'i' index for A is n_off.
                # The 'j' index for A is m_off + j.
                u_idx = (n_off + 0) * m + (m_off + j)
                
                # B index: (m_off + j) * p + (p_off + k)
                v_idx = (m_off + j) * p + (p_off + k)
                
                # C index: (p_off + k) * n + (n_off + 0)
                w_idx = (p_off + k) * n + (n_off + 0)
                
                U[u_idx][c] = 1
                V[v_idx][c] = 1
                W[w_idx][c] = 1
        return r

    def add_2x2x2(U, V, W, n_off, m_off, p_off, col_start):
        # Strassen for 2x2x2
        # n_off, m_off, p_off are offsets
        r = 7
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Indices:
        # A[i, j] -> U[(n_off+i)*m + (m_off+j), c]
        # B[j, k] -> V[(m_off+j)*p + (p_off+k), c]
        # C[i, k] -> W[(p_off+k)*n + (n_off+i), c]
        
        # c=0
        U[(n_off+0)*m + (m_off+0)][col_start+0] = 1
        U[(n_off+1)*m + (m_off+1)][col_start+0] = 1
        V[(m_off+0)*p + (p_off+0)][col_start+0] = 1
        V[(m_off+1)*p + (p_off+1)][col_start+0] = 1
        W[(p_off+0)*n + (n_off+0)][col_start+0] = 1
        W[(p_off+1)*n + (n_off+1)][col_start+0] = 1
        
        # c=1
        U[(n_off+1)*m + (m_off+0)][col_start+1] = 1
        U[(n_off+1)*m + (m_off+1)][col_start+1] = 1
        V[(m_off+0)*p + (p_off+0)][col_start+1] = 1
        W[(p_off+0)*n + (n_off+1)][col_start+1] = 1
        
        # c=2
        U[(n_off+0)*m + (m_off+0)][col_start+2] = 1
        V[(m_off+0)*p + (p_off+1)][col_start+2] = 1
        V[(m_off+1)*p + (p_off+1)][col_start+2] = -1
        W[(p_off+1)*n + (n_off+0)][col_start+2] = 1
        
        # c=3
        U[(n_off+1)*m + (m_off+1)][col_start+3] = 1
        V[(m_off+1)*p + (p_off+0)][col_start+3] = 1
        V[(m_off+0)*p + (p_off+0)][col_start+3] = -1
        W[(p_off+1)*n + (n_off+1)][col_start+3] = 1
        
        # c=4
        U[(n_off+0)*m + (m_off+0)][col_start+4] = 1
        U[(n_off+0)*m + (m_off+1)][col_start+4] = 1
        V[(m_off+1)*p + (p_off+1)][col_start+4] = 1
        W[(p_off+1)*n + (n_off+0)][col_start+4] = 1
        
        # c=5
        U[(n_off+1)*m + (m_off+0)][col_start+5] = 1
        U[(n_off+0)*m + (m_off+0)][col_start+5] = -1
        V[(m_off+0)*p + (p_off+0)][col_start+5] = 1
        V[(m_off+0)*p + (p_off+1)][col_start+5] = 1
        W[(p_off+0)*n + (n_off+1)][col_start+5] = 1
        
        # c=6
        U[(n_off+0)*m + (m_off+1)][col_start+6] = 1
        U[(n_off+1)*m + (m_off+1)][col_start+6] = -1
        V[(m_off+1)*p + (p_off+0)][col_start+6] = 1
        V[(m_off+1)*p + (p_off+1)][col_start+6] = 1
        W[(p_off+1)*n + (n_off+1)][col_start+6] = 1
        
        return r

    # Block 1: n=3, m=2 (cols 0-1 of A, rows 0-1 of B), p=6
    # n_off=0, m_off=0, p_off=0
    # Split n=3 into 1 and 2.
    
    # Sub-block 1a: n=1, m=2, p=6
    # n_off=0, m_off=0, p_off=0
    r1a = add_1xmxp(U, V, W, 0, 0, 0, 2, 6, col_offset)
    col_offset += r1a
    
    # Sub-block 1b: n=2, m=2, p=6
    # n_off=1, m_off=0, p_off=0
    # Split p=6 into 2+2+2
    # p_offs: 0, 2, 4
    r1b = 0
    for k_block in range(3):
        p_off_sub = k_block * 2
        r_temp = add_2x2x2(U, V, W, 1, 0, p_off_sub, col_offset)
        col_offset += r_temp
        r1b += r_temp
    
    # Block 2: n=3, m=2 (cols 2-3 of A, rows 2-3 of B), p=6
    # n_off=0, m_off=2, p_off=0
    
    # Sub-block 2a: n=1, m=2, p=6
    # n_off=0, m_off=2, p_off=0
    r2a = add_1xmxp(U, V, W, 0, 2, 0, 2, 6, col_offset)
    col_offset += r2a
    
    # Sub-block 2b: n=2, m=2, p=6
    # n_off=1, m_off=2, p_off=0
    r2b = 0
    for k_block in range(3):
        p_off_sub = k_block * 2
        r_temp = add_2x2x2(U, V, W, 1, 2, p_off_sub, col_offset)
        col_offset += r_temp
        r2b += r_temp
        
    return {"U": U, "V": V, "W": W}
