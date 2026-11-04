"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(a_j_start, b_j_start, c_k_start):
        nonlocal col
        a11_idx = 0 * m + a_j_start
        a12_idx = 0 * m + a_j_start + 1
        a21_idx = 1 * m + a_j_start
        a22_idx = 1 * m + a_j_start + 1
        
        b11_idx = b_j_start * p + c_k_start
        b12_idx = b_j_start * p + c_k_start + 1
        b21_idx = (b_j_start + 1) * p + c_k_start
        b22_idx = (b_j_start + 1) * p + c_k_start + 1
        
        c11_idx = c_k_start * n + 0
        c12_idx = (c_k_start + 1) * n + 0
        c21_idx = c_k_start * n + 1
        c22_idx = (c_k_start + 1) * n + 1
        
        # M1
        U[col, a11_idx] += 1
        U[col, a22_idx] += 1
        V[col, b11_idx] += 1
        V[col, b22_idx] += 1
        W[col, c11_idx] += 1
        W[col, c22_idx] += 1
        col += 1
        
        # M2
        U[col, a21_idx] += 1
        U[col, a22_idx] += 1
        V[col, b11_idx] += 1
        W[col, c21_idx] += 1
        W[col, c22_idx] -= 1
        col += 1
        
        # M3
        U[col, a11_idx] += 1
        V[col, b12_idx] += 1
        V[col, b22_idx] -= 1
        W[col, c12_idx] += 1
        W[col, c22_idx] += 1
        col += 1
        
        # M4
        U[col, a22_idx] += 1
        V[col, b21_idx] += 1
        V[col, b11_idx] -= 1
        W[col, c11_idx] += 1
        W[col, c21_idx] += 1
        col += 1
        
        # M5
        U[col, a11_idx] += 1
        U[col, a12_idx] += 1
        V[col, b22_idx] += 1
        W[col, c11_idx] -= 1
        W[col, c12_idx] += 1
        col += 1
        
        # M6
        U[col, a21_idx] += 1
        U[col, a11_idx] -= 1
        V[col, b11_idx] += 1
        V[col, b12_idx] += 1
        W[col, c22_idx] += 1
        col += 1
        
        # M7
        U[col, a12_idx] += 1
        U[col, a22_idx] -= 1
        V[col, b21_idx] += 1
        V[col, b22_idx] += 1
        W[col, c11_idx] += 1
        col += 1

    def add_schoolbook_2x2x1(a_j_start, b_j_start, c_k_start):
        nonlocal col
        a11_idx = 0 * m + a_j_start
        a12_idx = 0 * m + a_j_start + 1
        a21_idx = 1 * m + a_j_start
        a22_idx = 1 * m + a_j_start + 1
        
        b1_idx = b_j_start * p + c_k_start
        b2_idx = (b_j_start + 1) * p + c_k_start
        
        c1_idx = c_k_start * n + 0
        c2_idx = c_k_start * n + 1
        
        U[col, a11_idx] += 1
        V[col, b1_idx] += 1
        W[col, c1_idx] += 1
        col += 1
        
        U[col, a12_idx] += 1
        V[col, b2_idx] += 1
        W[col, c1_idx] += 1
        col += 1
        
        U[col, a21_idx] += 1
        V[col, b1_idx] += 1
        W[col, c2_idx] += 1
        col += 1
        
        U[col, a22_idx] += 1
        V[col, b2_idx] += 1
        W[col, c2_idx] += 1
        col += 1

    add_strassen(0, 0, 0)
    add_strassen(2, 2, 0)
    
    add_strassen(0, 0, 2)
    add_strassen(2, 2, 2)
    
    add_strassen(0, 0, 4)
    add_strassen(2, 2, 4)
    
    add_schoolbook_2x2x1(0, 0, 6)
    add_schoolbook_2x2x1(2, 2, 6)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
