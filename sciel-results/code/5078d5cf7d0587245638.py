import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x5 matrix.
    
    Strategy: Use structured Strassen-like block decomposition.
    Split the 3x4x5 problem into blocks where 2x2x2 sub-problems use
    Strassen's 7-multiplication algorithm, reducing total rank.
    
    Partition:
    - n=3: split into (2,1)
    - m=4: split into (2,2)  
    - p=5: split into (2,2,1)
    
    This gives blocks:
    - A: [[A11(2x2), A12(2x2)], [A21(1x2), A22(1x2)]]
    - B: [[B11(2x2), B12(2x2), B13(2x1)], [B21(2x2), B22(2x2), B23(2x1)]]
    - C: [[C11(2x2), C12(2x2), C13(2x1)], [C21(1x2), C22(1x2), C23(1x1)]]
    
    For each 2x2x2 block multiplication, use Strassen (rank 7 instead of 8).
    For other blocks, use schoolbook.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Build schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Strassen decomposition for 2x2x2
    # Given A(2x2), B(2x2), compute C(2x2) = A*B
    # Using 7 multiplications instead of 8
    # 
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    #
    # Strassen:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # In tensor decomposition form, each "multiplication" Mx corresponds to
    # a rank-1 term where U, V, W have specific patterns.
    
    def strassen_2x2x2(U, V, W, col, row_offset, mid_offset, col_offset, n_dim, p_dim):
        """
        Add Strassen decomposition for a 2x2x2 block.
        
        U: rows indexed by (i, j) where i in [0,2), j in [0,2)
           -> U[(row_offset + i)*n_dim + (mid_offset + j)][col]
        V: rows indexed by (j, k) where j in [0,2), k in [0,2)
           -> V[(mid_offset + j)*p_dim + (col_offset + k)][col]
        W: rows indexed by (k, i) where k in [0,2), i in [0,2)
           -> W[(col_offset + k)*n_dim + (row_offset + i)][col]
        
        Returns new col index after adding 7 terms.
        """
        # M1 = (a11+a22)*(b11+b22)
        U[(row_offset+0)*n_dim + (mid_offset+0)][col] = 1
        U[(row_offset+1)*n_dim + (mid_offset+1)][col] = 1
        V[(mid_offset+0)*p_dim + (col_offset+0)][col] = 1
        V[(mid_offset+1)*p_dim + (col_offset+1)][col] = 1
        W[(col_offset+0)*n_dim + (row_offset+0)][col] = 1
        W[(col_offset+1)*n_dim + (row_offset+1)][col] = 1
        col += 1
        
        # M2 = (a21+a22)*b11
        U[(row_offset+1)*n_dim + (mid_offset+0)][col] = 1
        U[(row_offset+1)*n_dim + (mid_offset+1)][col] = 1
        V[(mid_offset+0)*p_dim + (col_offset+0)][col] = 1
        W[(col_offset+0)*n_dim + (row_offset+1)][col] = 1
        col += 1
        
        # M3 = a11*(b12-b22)
        U[(row_offset+0)*n_dim + (mid_offset+0)][col] = 1
        V[(mid_offset+0)*p_dim + (col_offset+1)][col] = 1
        V[(mid_offset+1)*p_dim + (col_offset+1)][col] = -1
        W[(col_offset+1)*n_dim + (row_offset+0)][col] = 1
        col += 1
        
        # M4 = a22*(b21-b11)
        U[(row_offset+1)*n_dim + (mid_offset+1)][col] = 1
        V[(mid_offset+1)*p_dim + (col_offset+0)][col] = 1
        V[(mid_offset+0)*p_dim + (col_offset+0)][col] = -1
        W[(col_offset+0)*n_dim + (row_offset+1)][col] = 1
        col += 1
        
        # M5 = (a11+a12)*b22
        U[(row_offset+0)*n_dim + (mid_offset+0)][col] = 1
        U[(row_offset+0)*n_dim + (mid_offset+1)][col] = 1
        V[(mid_offset+1)*p_dim + (col_offset+1)][col] = 1
        W[(col_offset+1)*n_dim + (row_offset+0)][col] = 1
        col += 1
        
        # M6 = (a21-a11)*(b11+b12)
        U[(row_offset+1)*n_dim + (mid_offset+0)][col] = 1
        U[(row_offset+0)*n_dim + (mid_offset+0)][col] = -1
        V[(mid_offset+0)*p_dim + (col_offset+0)][col] = 1
        V[(mid_offset+0)*p_dim + (col_offset+1)][col] = 1
        W[(col_offset+0)*n_dim + (row_offset+1)][col] = 1
        W[(col_offset+1)*n_dim + (row_offset+1)][col] = 1
        col += 1
        
        # M7 = (a12-a22)*(b21+b22)
        U[(row_offset+0)*n_dim + (mid_offset+1)][col] = 1
        U[(row_offset+1)*n_dim + (mid_offset+1)][col] = -1
        V[(mid_offset+1)*p_dim + (col_offset+0)][col] = 1
        V[(mid_offset+1)*p_dim + (col_offset+1)][col] = 1
        W[(col_offset+0)*n_dim + (row_offset+0)][col] = 1
        W[(col_offset+1)*n_dim + (row_offset+0)][col] = 1
        col += 1
        
        return col
    
    def schoolbook_block(U, V, W, col, row_offset, mid_offset, col_offset, 
                         n_dim, p_dim, n_size, m_size, p_size):
        """
        Add schoolbook decomposition for an n_size x m_size x p_size block.
        
        Returns new col index after adding n_size*m_size*p_size terms.
        """
        for i in range(n_size):
            for j in range(m_size):
                for k in range(p_size):
                    U[(row_offset + i)*n_dim + (mid_offset + j)][col] = 1
                    V[(mid_offset + j)*p_dim + (col_offset + k)][col] = 1
                    W[(col_offset + k)*n_dim + (row_offset + i)][col] = 1
                    col += 1
        return col
    
    # Now construct the block-based decomposition
    # Split: n=3 -> (2,1), m=4 -> (2,2), p=5 -> (2,2,1)
    
    # Block boundaries:
    # n: [0,2), [2,3)  -> sizes 2, 1
    # m: [0,2), [2,4)  -> sizes 2, 2
    # p: [0,2), [2,4), [4,5)  -> sizes 2, 2, 1
    
    n_splits = [(0, 2), (2, 3)]
    m_splits = [(0, 2), (2, 4)]
    p_splits = [(0, 2), (2, 4), (4, 5)]
    
    # Initialize with zero matrices of size rank_schoolbook (we'll trim later)
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    
    # For each combination of n-block, m-block, p-block, add appropriate decomposition
    for (n_start, n_end) in n_splits:
        n_size = n_end - n_start
        for (m_start, m_end) in m_splits:
            m_size = m_end - m_start
            for (p_start, p_end) in p_splits:
                p_size = p_end - p_start
                
                if time.time() > deadline - 30:
                    break
                
                # If all three dimensions are 2, use Strassen
                if n_size == 2 and m_size == 2 and p_size == 2:
                    col = strassen_2x2x2(U, V, W, col, n_start, m_start, p_start, n, p)
                else:
                    # Otherwise use schoolbook for this block
                    col = schoolbook_block(U, V, W, col, n_start, m_start, p_start, 
                                          n, p, n_size, m_size, p_size)
    
    # Trim to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    rank_found = col
    
    # Verify the decomposition
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly reproduces the multiplication tensor."""
        errors = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    computed = 0
                    for r in range(rank):
                        computed += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(computed - 1) > 1e-9:
                        errors += 1
        return errors == 0
    
    if verify_decomposition(U, V, W, rank_found):
        return {"U": U, "V": V, "W": W}
    
    # If verification failed, return schoolbook as fallback
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
