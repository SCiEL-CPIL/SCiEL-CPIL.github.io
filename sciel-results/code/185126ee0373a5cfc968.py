"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 84
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_simple(i, j, k, col):
        """Add a simple tensor A[i,j] * B[j,k] -> C[k,i]"""
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] += 1
        V[v_idx][col] += 1
        W[w_idx][col] += 1

    def add_strassen(i_off, j_off, k_off, col_start):
        """
        Add Strassen decomposition for 2x2x2 block.
        Local indices: i in [0,2), j in [0,2), k in [0,2).
        Global indices: i+i_off, j+j_off, k+k_off.
        Uses columns col_start to col_start+6.
        """
        # M1 = (A11+A22)(B11+B22) -> C11(+1), C22(+1)
        # A parts: (1,1), (2,2) with coeff 1
        # B parts: (1,1), (2,2) with coeff 1
        add_strassen_term(col_start, 
                          [(i_off+0, j_off+0, 1), (i_off+1, j_off+1, 1)], 
                          [(j_off+0, k_off+0, 1), (j_off+1, k_off+1, 1)], 
                          [(k_off+0, i_off+0, 1), (k_off+1, i_off+1, 1)])
        
        # M2 = (A21+A22)B11 -> C21(+1)
        add_strassen_term(col_start+1,
                          [(i_off+1, j_off+0, 1), (i_off+1, j_off+1, 1)],
                          [(j_off+0, k_off+0, 1)],
                          [(k_off+0, i_off+1, 1)])
        
        # M3 = A11(B12-B22) -> C12(+1)
        add_strassen_term(col_start+2,
                          [(i_off+0, j_off+0, 1)],
                          [(j_off+0, k_off+1, 1), (j_off+1, k_off+1, -1)],
                          [(k_off+1, i_off+0, 1)])
        
        # M4 = A22(B21-B11) -> C11(-1), C21(+1)
        add_strassen_term(col_start+3,
                          [(i_off+1, j_off+1, 1)],
                          [(j_off+1, k_off+0, 1), (j_off+0, k_off+0, -1)],
                          [(k_off+0, i_off+0, -1), (k_off+0, i_off+1, 1)])
        
        # M5 = (A11+A12)B22 -> C11(-1), C12(+1)
        add_strassen_term(col_start+4,
                          [(i_off+0, j_off+0, 1), (i_off+0, j_off+1, 1)],
                          [(j_off+1, k_off+1, 1)],
                          [(k_off+1, i_off+0, -1), (k_off+1, i_off+1, 1)])
        
        # M6 = (A21-A11)(B11+B12) -> C22(+1)
        add_strassen_term(col_start+5,
                          [(i_off+1, j_off+0, 1), (i_off+0, j_off+0, -1)],
                          [(j_off+0, k_off+0, 1), (j_off+0, k_off+1, 1)],
                          [(k_off+1, i_off+1, 1)])
        
        # M7 = (A12-A22)(B21+B22) -> C11(+1), C22(-1)
        add_strassen_term(col_start+6,
                          [(i_off+0, j_off+1, 1), (i_off+1, j_off+1, -1)],
                          [(j_off+1, k_off+0, 1), (j_off+1, k_off+1, 1)],
                          [(k_off+0, i_off+0, 1), (k_off+1, i_off+1, -1)])
        
        return col_start + 7

    def add_strassen_term(col, u_terms, v_terms, w_terms):
        """
        Adds a bilinear form to U, V, W at column col.
        u_terms: list of (i, j, coeff)
        v_terms: list of (j, k, coeff)
        w_terms: list of (k, i, coeff)
        """
        for (i, j, coeff) in u_terms:
            U[i * m + j][col] += coeff
        for (j, k, coeff) in v_terms:
            V[j * p + k][col] += coeff
        for (k, i, coeff) in w_terms:
            W[k * n + i][col] += coeff

    def add_schoolbook(i_off, j_off, k_off, n_b, m_b, p_b, col_start):
        """
        Adds schoolbook decomposition for n_b x m_b x p_b block.
        """
        col = col_start
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    add_simple(i_off + i, j_off + j, k_off + k, col)
                    col += 1
        return col

    col = 0
    
    # Block 1: 2x2x2 (Top-Left)
    # i: 0-1, j: 0-1, k: 0-1
    col = add_strassen(0, 0, 0, col)
    
    # Block 2: 2x3x2 (Top-Middle)
    # i: 0-1, j: 0-2, k: 0-1
    # Split j into 2+1
    # Part 2a: 2x2x2 (j:0-1)
    col = add_strassen(0, 0, 0, col)
    # Part 2b: 2x1x2 (j:2)
    # Schoolbook for 2x1x2
    col = add_schoolbook(0, 2, 0, 2, 1, 2, col)
    
    # Block 3: 2x2x4 (Top-Right)
    # i: 0-1, j: 0-1, k: 0-3
    # Split k into 2+2
    # Part 3a: 2x2x2 (k:0-1)
    col = add_strassen(0, 0, 0, col)
    # Part 3b: 2x2x2 (k:2-3)
    col = add_strassen(0, 0, 2, col)
    
    # Block 4: 2x3x4 (Top-Right-ish)
    # i: 0-1, j: 0-2, k: 0-3
    # Split j into 2+1, k into 2+2
    # Part 4a: 2x2x2 (j:0-1, k:0-1)
    col = add_strassen(0, 0, 0, col)
    # Part 4b: 2x1x2 (j:2, k:0-1)
    col = add_schoolbook(0, 2, 0, 2, 1, 2, col)
    # Part 4c: 2x2x2 (j:0-1, k:2-3)
    col = add_strassen(0, 0, 2, col)
    # Part 4d: 2x1x2 (j:2, k:2-3)
    col = add_schoolbook(0, 2, 2, 2, 1, 2, col)
    
    # Block 5: 1x2x2 (Bottom-Left)
    # i: 2, j: 0-1, k: 0-1
    col = add_schoolbook(2, 0, 0, 1, 2, 2, col)
    
    # Block 6: 1x3x2 (Bottom-Middle)
    # i: 2, j: 0-2, k: 0-1
    col = add_schoolbook(2, 0, 0, 1, 3, 2, col)
    
    # Block 7: 1x2x4 (Bottom-Right-ish)
    # i: 2, j: 0-1, k: 0-3
    col = add_schoolbook(2, 0, 0, 1, 2, 4, col)
    
    # Block 8: 1x3x4 (Bottom-Right)
    # i: 2, j: 0-2, k: 0-3
    col = add_schoolbook(2, 0, 0, 1, 3, 4, col)
    
    # Verify rank used
    if col != rank:
        # If col < rank, we have unused columns (zeros).
        # If col > rank, we have a bug.
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
