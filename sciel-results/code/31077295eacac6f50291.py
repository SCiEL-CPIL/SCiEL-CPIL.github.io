"""Tensor decomposition search for 3x5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Dimensions: A (3x5), B (5x5), C (3x5)
    # Split B into B1 (5x3) and B2 (5x2)
    # Split A into A1 (3x3) and A2 (3x2)
    # C1 = A1 B11 + A2 B21 (3x3)
    # C2 = A1 B12 + A2 B22 (3x2)
    
    # We construct the decomposition for each block and concatenate.
    # Block 1: A1 (3x3) * B11 (3x3) -> C1_part (3x3)
    # Block 2: A2 (3x2) * B21 (2x3) -> C1_part (3x3)
    # Block 3: A1 (3x3) * B12 (3x2) -> C2_part (3x2)
    # Block 4: A2 (3x2) * B22 (2x2) -> C2_part (3x2)
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Helper to add a schoolbook decomposition for n_i x m_i x p_i
    def add_schoolbook(n_i, m_i, p_i, i_offset, j_offset, k_offset):
        """
        Adds schoolbook decomposition for a sub-multiplication.
        i: row of A (global offset i_offset)
        j: col of A / row of B (global offset j_offset)
        k: col of B / row of C (global offset k_offset)
        """
        U_sub = np.zeros((n_i * m_i, n_i * m_i * p_i))
        V_sub = np.zeros((m_i * p_i, n_i * m_i * p_i))
        W_sub = np.zeros((p_i * n_i, n_i * m_i * p_i))
        
        col = 0
        for i in range(n_i):
            for j in range(m_i):
                for k in range(p_i):
                    U_sub[i * m_i + j, col] = 1
                    V_sub[j * p_i + k, col] = 1
                    W_sub[k * n_i + i, col] = 1
                    col += 1
        
        # Map to global indices
        # U global index: (i_offset + i) * m + (j_offset + j)
        # V global index: (j_offset + j) * p + (k_offset + k)
        # W global index: (k_offset + k) * n + (i_offset + i)
        
        U_global = np.zeros((n * m, n_i * m_i * p_i))
        V_global = np.zeros((m * p, n_i * m_i * p_i))
        W_global = np.zeros((p * n, n_i * m_i * p_i))
        
        col = 0
        for i in range(n_i):
            for j in range(m_i):
                for k in range(p_i):
                    U_global[(i_offset + i) * m + (j_offset + j), col] = 1
                    V_global[(j_offset + j) * p + (k_offset + k), col] = 1
                    W_global[(k_offset + k) * n + (i_offset + i), col] = 1
                    col += 1
        
        U_blocks.append(U_global)
        V_blocks.append(V_global)
        W_blocks.append(W_global)

    # Block 1: A1 (3x3) * B11 (3x3)
    # A1 cols 0-2, B11 rows 0-2, C1 cols 0-2
    add_schoolbook(3, 3, 3, i_offset=0, j_offset=0, k_offset=0)
    
    # Block 2: A2 (3x2) * B21 (2x3)
    # A2 cols 3-4, B21 rows 3-4, C1 cols 0-2
    add_schoolbook(3, 2, 3, i_offset=0, j_offset=3, k_offset=0)
    
    # Block 3: A1 (3x3) * B12 (3x2)
    # A1 cols 0-2, B12 rows 0-2, C2 cols 3-4
    add_schoolbook(3, 3, 2, i_offset=0, j_offset=0, k_offset=3)
    
    # Block 4: A2 (3x2) * B22 (2x2)
    # A2 cols 3-4, B22 rows 3-4, C2 cols 3-4
    add_schoolbook(3, 2, 2, i_offset=0, j_offset=3, k_offset=3)
    
    # Concatenate
    U = np.hstack(U_blocks)
    V = np.hstack(V_blocks)
    W = np.hstack(W_blocks)
    
    rank = U.shape[1]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
