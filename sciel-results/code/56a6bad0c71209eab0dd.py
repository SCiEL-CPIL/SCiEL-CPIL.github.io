"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x6 matrix.
    Uses recursive decomposition based on Strassen and simple block splits.
    """
    n, m, p = DIMENSIONS
    
    # Total rank calculation
    # M_2,2,1: rank 2
    # M_2,2,2: rank 7
    # M_2,2,3: rank 9
    # M_2,2,6: rank 18
    # M_2,1,3: rank 1
    # M_2,3,3: rank 10
    # M_2,3,6: rank 20
    # Total: 38
    
    total_rank = 38
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    def add_block(n_b, m_b, p_b, i_off, j_off, k_off, rank_off):
        """
        Recursively add decomposition for M_{n_b, m_b, p_b} to global U, V, W.
        Returns new rank offset.
        """
        if (n_b, m_b, p_b) == (2, 2, 2):
            # Strassen
            # U indices: 0,1,2,3 -> A00, A01, A10, A11
            # V indices: 0,1,2,3 -> B00, B01, B10, B11
            # W indices: 0,1,2,3 -> C00, C01, C10, C11
            
            # M1
            U[(i_off+0)*m + (j_off+0), rank_off] = 1
            U[(i_off+1)*m + (j_off+1), rank_off] = 1
            V[(j_off+0)*p + (k_off+0), rank_off] = 1
            V[(j_off+1)*p + (k_off+1), rank_off] = 1
            W[(k_off+0)*n + (i_off+0), rank_off] = 1
            W[(k_off+1)*n + (i_off+1), rank_off] = 1
            rank_off += 1
            
            # M2
            U[(i_off+1)*m + (j_off+0), rank_off] = 1
            U[(i_off+1)*m + (j_off+1), rank_off] = 1
            V[(j_off+0)*p + (k_off+0), rank_off] = 1
            W[(k_off+0)*n + (i_off+1), rank_off] = 1
            rank_off += 1
            
            # M3
            U[(i_off+0)*m + (j_off+0), rank_off] = 1
            V[(j_off+0)*p + (k_off+1), rank_off] = 1
            V[(j_off+1)*p + (k_off+1), rank_off] = -1
            W[(k_off+1)*n + (i_off+0), rank_off] = 1
            rank_off += 1
            
            # M4
            U[(i_off+1)*m + (j_off+1), rank_off] = 1
            V[(j_off+1)*p + (k_off+0), rank_off] = 1
            V[(j_off+0)*p + (k_off+0), rank_off] = -1
            W[(k_off+0)*n + (i_off+1), rank_off] = 1
            rank_off += 1
            
            # M5
            U[(i_off+0)*m + (j_off+0), rank_off] = 1
            U[(i_off+0)*m + (j_off+1), rank_off] = 1
            V[(j_off+1)*p + (k_off+1), rank_off] = 1
            W[(k_off+0)*n + (i_off+0), rank_off] = -1
            W[(k_off+1)*n + (i_off+1), rank_off] = 1
            rank_off += 1
            
            # M6
            U[(i_off+1)*m + (j_off+0), rank_off] = 1
            U[(i_off+0)*m + (j_off+0), rank_off] = -1
            V[(j_off+0)*p + (k_off+0), rank_off] = 1
            V[(j_off+0)*p + (k_off+1), rank_off] = 1
            W[(k_off+1)*n + (i_off+1), rank_off] = 1
            rank_off += 1
            
            # M7
            U[(i_off+0)*m + (j_off+1), rank_off] = 1
            U[(i_off+1)*m + (j_off+1), rank_off] = -1
            V[(j_off+1)*p + (k_off+0), rank_off] = 1
            V[(j_off+1)*p + (k_off+1), rank_off] = 1
            W[(k_off+0)*n + (i_off+0), rank_off] = 1
            rank_off += 1
            
            return rank_off
        
        elif (n_b, m_b, p_b) == (2, 2, 1):
            # Rank 2
            # t=0
            U[(i_off+0)*m + (j_off+0), rank_off] = 1
            V[(j_off+0)*p + (k_off+0), rank_off] = 1
            W[(k_off+0)*n + (i_off+0), rank_off] = 1
            W[(k_off+0)*n + (i_off+1), rank_off] = 1
            rank_off += 1
            # t=1
            U[(i_off+0)*m + (j_off+1), rank_off] = 1
            V[(j_off+1)*p + (k_off+0), rank_off] = 1
            W[(k_off+0)*n + (i_off+0), rank_off] = 1
            W[(k_off+0)*n + (i_off+1), rank_off] = 1
            rank_off += 1
            return rank_off
            
        elif (n_b, m_b, p_b) == (2, 1, 3):
            # Rank 1
            for i_l in range(2):
                U[(i_off+i_l)*m + (j_off+0), rank_off] = 1
            for k_l in range(3):
                V[(j_off+0)*p + (k_off+k_l), rank_off] = 1
            for k_l in range(3):
                for i_l in range(2):
                    W[(k_off+k_l)*n + (i_off+i_l), rank_off] = 1
            rank_off += 1
            return rank_off
            
        elif (n_b, m_b, p_b) == (2, 2, 3):
            # Split p=3 into 1+2
            rank_off = add_block(2, 2, 1, i_off, j_off, k_off, rank_off)
            rank_off = add_block(2, 2, 2, i_off, j_off, k_off+1, rank_off)
            return rank_off
            
        elif (n_b, m_b, p_b) == (2, 2, 6):
            # Split p=6 into 3+3
            rank_off = add_block(2, 2, 3, i_off, j_off, k_off, rank_off)
            rank_off = add_block(2, 2, 3, i_off, j_off, k_off+3, rank_off)
            return rank_off
            
        elif (n_b, m_b, p_b) == (2, 3, 3):
            # Split m=3 into 1+2
            rank_off = add_block(2, 1, 3, i_off, j_off, k_off, rank_off)
            rank_off = add_block(2, 2, 3, i_off, j_off+1, k_off, rank_off)
            return rank_off
            
        elif (n_b, m_b, p_b) == (2, 3, 6):
            # Split p=6 into 3+3
            rank_off = add_block(2, 3, 3, i_off, j_off, k_off, rank_off)
            rank_off = add_block(2, 3, 3, i_off, j_off, k_off+3, rank_off)
            return rank_off
            
        else:
            raise ValueError(f"Unknown block {n_b}x{m_b}x{p_b}")

    # Main call
    # Split m=5 into 2+3
    rank_off = add_block(2, 2, 6, 0, 0, 0, 0)
    rank_off = add_block(2, 3, 6, 0, 2, 0, rank_off)
    
    if rank_off != total_rank:
        print(f"Warning: Rank mismatch. Expected {total_rank}, got {rank_off}")
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
