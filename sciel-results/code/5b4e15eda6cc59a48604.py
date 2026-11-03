"""
Program to find a low-rank tensor decomposition for 4x5x5 matrix multiplication.
Uses block decomposition with Strassen's algorithm for 2x2x2 sub-blocks.
"""
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Split dimensions
    n1, n2 = 2, 2
    m1, m2 = 2, 3
    p1, p2 = 2, 3
    
    # Total rank is sum of ranks of blocks
    # We use Strassen (rank 7) for 2x2x2 blocks
    # Schoolbook for others
    
    # Blocks are (n_i, m_j, p_k)
    # A is 4x5, B is 5x5
    # A split: rows 0-1 (n1), rows 2-3 (n2). Cols 0-1 (m1), cols 2-4 (m2)
    # B split: rows 0-1 (m1), rows 2-4 (m2). Cols 0-1 (p1), cols 2-4 (p2)
    
    # Block (n1, m1, p1): 2x2x2 -> Strassen
    # Block (n1, m2, p1): 2x3x2 -> Schoolbook
    # Block (n2, m1, p1): 2x2x2 -> Strassen
    # Block (n2, m2, p1): 2x3x2 -> Schoolbook
    # Block (n1, m1, p2): 2x2x3 -> Schoolbook
    # Block (n1, m2, p2): 2x3x3 -> Schoolbook
    # Block (n2, m1, p2): 2x2x3 -> Schoolbook
    # Block (n2, m2, p2): 2x3x3 -> Schoolbook
    
    # Ranks:
    # S(2,2,2) = 7
    # S(2,3,2) = 12
    # S(2,2,2) = 7
    # S(2,3,2) = 12
    # S(2,2,3) = 12
    # S(2,3,3) = 18
    # S(2,2,3) = 12
    # S(2,3,3) = 18
    # Total = 7+12+7+12+12+18+12+18 = 98
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add schoolbook terms for a block
    def add_schoolbook(n_off, m_off, p_off, n_sz, m_sz, p_sz):
        nonlocal col
        for i in range(n_sz):
            for j in range(m_sz):
                for k in range(p_sz):
                    U[(n_off + i) * m + (m_off + j)][col] = 1
                    V[(m_off + j) * p + (p_off + k)][col] = 1
                    W[(p_off + k) * n + (n_off + i)][col] = 1
                    col += 1
                    
    # Helper to add Strassen terms for 2x2x2 block
    def add_strassen(n_off, m_off, p_off):
        nonlocal col
        # Strassen 2x2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # M1
        U[n_off*m + m_off][col] = 1; V[m_off*p + p_off][col] = 1; W[p_off*n + n_off][col] = 1
        U[(n_off+1)*m + (m_off+1)][col] = 1; V[(m_off+1)*p + (p_off+1)][col] = 1; W[(p_off+1)*n + (n_off+1)][col] = 1
        U[n_off*m + (m_off+1)][col] = 1; V[(m_off+1)*p + p_off][col] = 1; W[p_off*n + (n_off+1)][col] = 1
        U[(n_off+1)*m + m_off][col] = 1; V[m_off*p + (p_off+1)][col] = 1; W[(p_off+1)*n + n_off][col] = 1
        col += 1
        
        # M2
        U[(n_off+1)*m + m_off][col] = 1; V[m_off*p + p_off][col] = 1; W[p_off*n + (n_off+1)][col] = 1
        U[(n_off+1)*m + (m_off+1)][col] = 1; V[(m_off+1)*p + p_off][col] = 1; W[p_off*n + (n_off+1)][col] = 1
        col += 1
        
        # M3
        U[n_off*m + m_off][col] = 1; V[m_off*p + (p_off+1)][col] = 1; W[(p_off+1)*n + n_off][col] = 1
        U[n_off*m + m_off][col] = -1; V[m_off*p + p_off][col] = -1; W[p_off*n + n_off][col] = -1
        col += 1
        
        # M4
        U[(n_off+1)*m + (m_off+1)][col] = 1; V[(m_off+1)*p + p_off][col] = 1; W[p_off*n + (n_off+1)][col] = 1
        U[(n_off+1)*m + (m_off+1)][col] = -1; V[(m_off+1)*p + (p_off+1)][col] = -1; W[(p_off+1)*n + (n_off+1)][col] = -1
        col += 1
        
        # M5
        U[n_off*m + m_off][col] = 1; V[(m_off+1)*p + (p_off+1)][col] = 1; W[(p_off+1)*n + n_off][col] = 1
        U[n_off*m + (m_off+1)][col] = 1; V[(m_off+1)*p + (p_off+1)][col] = 1; W[(p_off+1)*n + n_off][col] = 1
        col += 1
        
        # M6
        U[(n_off+1)*m + m_off][col] = 1; V[m_off*p + p_off][col] = 1; W[p_off*n + (n_off+1)][col] = 1
        U[(n_off+1)*m + m_off][col] = 1; V[m_off*p + (p_off+1)][col] = 1; W[(p_off+1)*n + (n_off+1)][col] = 1
        U[n_off*m + m_off][col] = -1; V[m_off*p + p_off][col] = -1; W[p_off*n + n_off][col] = -1
        U[n_off*m + m_off][col] = -1; V[m_off*p + (p_off+1)][col] = -1; W[(p_off+1)*n + n_off][col] = -1
        col += 1
        
        # M7
        U[n_off*m + (m_off+1)][col] = 1; V[(m_off+1)*p + p_off][col] = 1; W[p_off*n + n_off][col] = 1
        U[n_off*m + (m_off+1)][col] = 1; V[(m_off+1)*p + (p_off+1)][col] = 1; W[(p_off+1)*n + n_off][col] = 1
        U[(n_off+1)*m + (m_off+1)][col] = -1; V[(m_off+1)*p + p_off][col] = -1; W[p_off*n + (n_off+1)][col] = -1
        U[(n_off+1)*m + (m_off+1)][col] = -1; V[(m_off+1)*p + (p_off+1)][col] = -1; W[(p_off+1)*n + (n_off+1)][col] = -1
        col += 1

    # Block 1: n1, m1, p1 (2,2,2) -> Strassen
    add_strassen(0, 0, 0)
    
    # Block 2: n1, m2, p1 (2,3,2) -> Schoolbook
    add_schoolbook(0, 2, 0, 2, 3, 2)
    
    # Block 3: n2, m1, p1 (2,2,2) -> Strassen
    add_strassen(2, 0, 0)
    
    # Block 4: n2, m2, p1 (2,3,2) -> Schoolbook
    add_schoolbook(2, 2, 0, 2, 3, 2)
    
    # Block 5: n1, m1, p2 (2,2,3) -> Schoolbook
    add_schoolbook(0, 0, 2, 2, 2, 3)
    
    # Block 6: n1, m2, p2 (2,3,3) -> Schoolbook
    add_schoolbook(0, 2, 2, 2, 3, 3)
    
    # Block 7: n2, m1, p2 (2,2,3) -> Schoolbook
    add_schoolbook(2, 0, 2, 2, 2, 3)
    
    # Block 8: n2, m2, p2 (2,3,3) -> Schoolbook
    add_schoolbook(2, 2, 2, 2, 3, 3)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
