"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _strassen_2x2(U_rows, V_cols, W_rows, U, V, W, base_rank):
    """
    Add Strassen 2x2 block multiplication to the decomposition.
    U: list of lists, shape (2, rank_total) - but we work with row indices
    We'll build up the full U, V, W matrices.
    """
    pass

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Use Strassen's algorithm with padding to enable 2x2 block decomposition
    # Pad m and p to even numbers: m=5->6, p=5->6
    # n=4 is already even
    m_pad = 6
    p_pad = 6
    
    # We'll decompose using Strassen recursively
    # For a 2x2 block: (A11 A12; A21 A22) * (B11 B12; B21 B22)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x6 * 6x6:
    # Split into 2x2 blocks of size 2x3 and 3x3
    # But Strassen needs square blocks. Let me pad to 8x8.
    
    # Alternative: use the known optimal or near-optimal decompositions
    # For 4x5 * 5x5, let's use a direct Strassen-based approach
    
    # Pad to 8x8 for clean Strassen: 8=2^3
    # 4x5 -> 8x8 (pad rows 4-7 and cols 5-7)
    # 5x5 -> 8x8 (pad rows 5-7 and cols 5-7)
    # Result 8x8, but we only need 4x5
    
    n_pad = 8
    m_pad = 8
    p_pad = 8
    
    # Strassen decomposition of 8x8 * 8x8
    # Level 1: 8x8 = 2x2 blocks of 4x4
    # Level 2: 4x4 = 2x2 blocks of 2x2
    # Level 3: 2x2 = Strassen (rank 7)
    # Total rank = 7 * 4 * 4 = 112
    
    # But we can be smarter. Let me use a mixed approach.
    # Actually, let's just implement clean Strassen for 8x8*8x8 and extract the 4x5 part.
    
    # Build the full 8x8 * 8x8 Strassen decomposition
    # U shape: (8*8, rank), V shape: (8*8, rank), W shape: (8*8, rank)
    # W index: k*8 + i (output row i, col k) -> W[k*n_pad + i]
    
    rank = 112
    U_full = [[0] * rank for _ in range(n_pad * m_pad)]
    V_full = [[0] * rank for _ in range(m_pad * p_pad)]
    W_full = [[0] * rank for _ in range(p_pad * n_pad)]
    
    col = 0
    
    # Strassen for 2x2 blocks
    def add_strassen_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, block_size):
        """
        Add Strassen decomposition for a block multiplication.
        A is block_size x block_size, B is block_size x block_size, C is block_size x block_size
        A_rows, A_cols: top-left corner of A in the full matrix
        B_rows, B_cols: top-left corner of B
        C_rows, C_cols: top-left corner of C
        """
        nonlocal col
        s = block_size
        # A11: rows A_rows:A_rows+s, cols A_cols:A_cols+s
        # A12: rows A_rows:A_rows+s, cols A_cols+s:A_cols+2*s
        # A21: rows A_rows+s:A_rows+2*s, cols A_cols:A_cols+s
        # A22: rows A_rows+s:A_rows+2*s, cols A_cols+s:A_cols+2*s
        # Similarly for B and C
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        def add_term(U_row_offset, V_row_offset, W_row_offset, u_sign, v_sign, w_sign):
            """Add a single multiplication term."""
            nonlocal col
            # U index: (i*m_pad + j) where i is row in A, j is col in A
            # V index: (j*p_pad + k) where j is row in B, k is col in B
            # W index: (k*n_pad + i) where k is col in C, i is row in C
            
            # For a block, we need to add contributions from all elements
            # But Strassen combines blocks, so we add the block-level terms
            
            # This is getting complex. Let me use a different approach.
            pass
        
        # Simpler: directly enumerate the 7 Strassen products
        # Each product is a sum of element-wise products
        
        # M1 = (A11+A22)(B11+B22)
        # For each (i,j) in A11+A22 and (j',k) in B11+B22, contribute to C
        # This is O(s^4) per Strassen level, which is fine for s=2
        
        # Let me just hardcode for s=2 (2x2 blocks)
        if s == 2:
            # A11: (A_rows, A_cols), (A_rows+1, A_cols+1)
            # A12: (A_rows, A_cols+2), (A_rows+1, A_cols+3)
            # A21: (A_rows+2, A_cols), (A_rows+3, A_cols+1)
            # A22: (A_rows+2, A_cols+2), (A_rows+3, A_cols+3)
            
            # M1 = (A11+A22)(B11+B22)
            for di in range(2):
                for dj in range(2):
                    # A11[i,j] and A22[i,j]
                    U_idx1 = (A_rows + di) * m_pad + (A_cols + dj)
                    U_idx2 = (A_rows + 2 + di) * m_pad + (A_cols + 2 + dj)
                    for dk in range(2):
                        V_idx1 = (dj) * p_pad + (B_cols + dk)  # B11[j,k]
                        V_idx2 = (dj) * p_pad + (B_cols + 2 + dk)  # B12[j,k]... wait
                        # B11: rows B_rows:B_rows+2, cols B_cols:B_cols+2
                        # B12: rows B_rows:B_rows+2, cols B_cols+2:B_cols+4
                        # B21: rows B_rows+2:B_rows+4, cols B_cols:B_cols+2
                        # B22: rows B_rows+2:B_rows+4, cols B_cols+2:B_cols+4
                        
                        # M1 = (A11+A22)(B11+B22)
                        # A11[i,j] * B11[j,k] -> C11[i,k]
                        # A11[i,j] * B22[j,k] -> C12[i,k]... no
                        # (A11+A22) is 2x2, (B11+B22) is 2x2
                        # Product is 2x2, contributes to C11 and C22 etc.
                        
                        # Let me be more careful
                        # (A11+A22)[i,j] * (B11+B22)[j,k] contributes to (C11+C22)[i,k]... no
                        # Actually M1 contributes to C11 and C22 via the formulas
                        # C11 = M1+M4-M5+M7, C22 = M1-M2+M3+M6
                        # So M1 contributes +1 to C11 and +1 to C22
                        # But M1 is a 2x2 matrix itself
                        # M1[i,k] = sum_j (A11+A22)[i,j] * (B11+B22)[j,k]
                        # This M1[i,k] adds to C11[i,k] and C22[i,k]
                        
                        # So for each (i,j,k):
                        # (A11+A22)[i,j] * (B11+B22)[j,k]
                        # = A11[i,j]*B11[j,k] + A11[i,j]*B22[j,k] + A22[i,j]*B11[j,k] + A22[i,j]*B22[j,k]
                        # Each of these contributes to C11[i,k] and C22[i,k]
                        
                        # C11[i,k] is at row C_rows+i, col C_cols+k
                        # C22[i,k] is at row C_rows+2+i, col C_cols+2+k
                        
                        # W index for C11[i,k]: k_full * n_pad + i_full
                        # where i_full = C_rows + i, k_full = C_cols + k
                        # W index for C22[i,k]: k_full2 * n_pad + i_full2
                        # where i_full2 = C_rows + 2 + i, k_full2 = C_cols + 2 + k
                        
                        # For A11[i,j]*B11[j,k] -> C11[i,k] and C22[i,k]
                        U_idx = (A_rows + i) * m_pad + (A_cols + j)
                        V_idx = (B_rows + j) * p_pad + (B_cols + k)
                        W_idx1 = (C_cols + k) * n_pad + (C_rows + i)
                        W_idx2 = (C_cols + 2 + k) * n_pad + (C_rows + 2 + i)
                        
                        if col < rank:
                            U_full[U_idx][col] += 1
                            V_full[V_idx][col] += 1
                            W_full[W_idx1][col] += 1
                            W_full[W_idx2][col] += 1
                            col += 1
                        else:
                            return False
                        
                        # A11[i,j]*B22[j,k] -> C11[i,k] and C22[i,k]
                        U_idx = (A_rows + i) * m_pad + (A_cols + j)
                        V_idx = (B_rows + j) * p_pad + (B_cols + 2 + k)
                        W_idx1 = (C_cols + k) * n_pad + (C_rows + i)
                        W_idx2 = (C_cols + 2 + k) * n_pad + (C_rows + 2 + i)
                        
                        if col < rank:
                            U_full[U_idx][col] += 1
                            V_full[V_idx][col] += 1
                            W_full[W_idx1][col] += 1
                            W_full[W_idx2][col] += 1
                            col += 1
                        else:
                            return False
                        
                        # A22[i,j]*B11[j,k] -> C11[i,k] and C22[i,k]
                        U_idx = (A_rows + 2 + i) * m_pad + (A_cols + j)
                        V_idx = (B_rows + j) * p_pad + (B_cols + k)
                        W_idx1 = (C_cols + k) * n_pad + (C_rows + i)
                        W_idx2 = (C_cols + 2 + k) * n_pad + (C_rows + 2 + i)
                        
                        if col < rank:
                            U_full[U_idx][col] += 1
                            V_full[V_idx][col] += 1
                            W_full[W_idx1][col] += 1
                            W_full[W_idx2][col] += 1
                            col += 1
                        else:
                            return False
                        
                        # A22[i,j]*B22[j,k] -> C11[i,k] and C22[i,k]
                        U_idx = (A_rows + 2 + i) * m_pad + (A_cols + j)
                        V_idx = (B_rows + j) * p_pad + (B_cols + 2 + k)
                        W_idx1 = (C_cols + k) * n_pad + (C_rows + i)
                        W_idx2 = (C_cols + 2 + k) * n_pad + (C_rows + 2 + i)
                        
                        if col < rank:
                            U_full[U_idx][col] += 1
                            V_full[V_idx][col] += 1
                            W_full[W_idx1][col] += 1
                            W_full[W_idx2][col] += 1
                            col += 1
                        else:
                            return False
            
            # This is getting very complex and I'm running out of time.
            # Let me fall back to schoolbook for now.
            return False
        
        return False
    
    # Given the complexity, let me just use schoolbook but with a smaller effective rank
    # by exploiting the zero padding.
    
    # Actually, let me just return the schoolbook decomposition. It's correct and safe.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
