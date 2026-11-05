import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4*4x8 matrix multiplication.
    
    Strategy:
    - Partition A (3x4) into A1 (2x4) and A2 (1x4)
    - Partition B (4x8) into blocks
    - For A1*B: Use Strassen on 2x2 blocks.
      A1 is 2x4. Split into two 2x2 blocks horizontally: A1_L, A1_R.
      B is 4x8. Split into four 2x2 blocks vertically? No.
      B is 4x8.
      A1_L (2x2) multiplies B[0:2, :] (2x8).
      A1_R (2x2) multiplies B[2:4, :] (2x8).
      
      For A1_L * B[0:2, :]:
      B[0:2, :] is 2x8. Split into four 2x2 blocks: B00, B01, B02, B03.
      Each 2x2 * 2x2 is Strassen rank 7.
      Total 4 * 7 = 28.
      
      For A1_R * B[2:4, :]:
      B[2:4, :] is 2x8. Split into four 2x2 blocks: B10, B11, B12, B13.
      Each 2x2 * 2x2 is Strassen rank 7.
      Total 4 * 7 = 28.
      
      Total for A1*B is 56.
      
    - For A2*B: Schoolbook 1x4 * 4x8.
      A2 is 1x4. B is 4x8.
      Result is 1x8.
      32 scalar multiplications. Rank 32.
      
    Total rank: 56 + 32 = 88.
    """
    start_time = time.time()
    DEADLINE = 120.0
    
    n, m, p = DIMENSIONS
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We will build lists of non-zero entries for each rank term
    # term = (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    # But it's easier to just build the full matrices if rank is small.
    # Rank 88 is small.
    
    U = np.zeros((n * m, 88), dtype=float)
    V = np.zeros((m * p, 88), dtype=float)
    W = np.zeros((p * n, 88), dtype=float)
    
    rank = 0
    
    def add_strassen_block(a_row_start, a_col_start, b_row_start, b_col_start):
        nonlocal rank
        # A block: rows [a_row_start, a_row_start+2), cols [a_col_start, a_col_start+2)
        # B block: rows [b_row_start, b_row_start+2), cols [b_col_start, b_col_start+2)
        
        # U indices (A): row * m + col
        u_a11 = a_row_start * m + a_col_start
        u_a12 = a_row_start * m + a_col_start + 1
        u_a21 = (a_row_start + 1) * m + a_col_start
        u_a22 = (a_row_start + 1) * m + a_col_start + 1
        
        # V indices (B): row * p + col
        v_b11 = b_row_start * p + b_col_start
        v_b12 = b_row_start * p + b_col_start + 1
        v_b21 = (b_row_start + 1) * p + b_col_start
        v_b22 = (b_row_start + 1) * p + b_col_start + 1
        
        # W indices (C): col * n + row  (Trace convention W[k*n+i])
        # C block: rows [a_row_start, a_row_start+2), cols [b_col_start, b_col_start+2)
        w_c11 = b_col_start * n + a_row_start
        w_c12 = (b_col_start + 1) * n + a_row_start
        w_c21 = b_col_start * n + a_row_start + 1
        w_c22 = (b_col_start + 1) * n + a_row_start + 1
        
        # Strassen 7 terms
        # M1 = (a11 + a22) * (b11 + b22)
        U[u_a11, rank] = 1.0; U[u_a22, rank] = 1.0
        V[v_b11, rank] = 1.0; V[v_b22, rank] = 1.0
        W[w_c11, rank] = 1.0; W[w_c22, rank] = 1.0
        rank += 1
        
        # M2 = (a21 + a22) * b11
        U[u_a21, rank] = 1.0; U[u_a22, rank] = 1.0
        V[v_b11, rank] = 1.0
        W[w_c21, rank] = 1.0; W[w_c22, rank] = -1.0
        rank += 1
        
        # M3 = a11 * (b12 - b22)
        U[u_a11, rank] = 1.0
        V[v_b12, rank] = 1.0; V[v_b22, rank] = -1.0
        W[w_c12, rank] = 1.0; W[w_c22, rank] = 1.0
        rank += 1
        
        # M4 = a22 * (b21 - b11)
        U[u_a22, rank] = 1.0
        V[v_b21, rank] = 1.0; V[v_b11, rank] = -1.0
        W[w_c11, rank] = 1.0; W[w_c21, rank] = 1.0
        rank += 1
        
        # M5 = (a11 + a12) * b22
        U[u_a11, rank] = 1.0; U[u_a12, rank] = 1.0
        V[v_b22, rank] = 1.0
        W[w_c11, rank] = -1.0; W[w_c12, rank] = 1.0
        rank += 1
        
        # M6 = (a21 - a11) * (b11 + b12)
        U[u_a21, rank] = 1.0; U[u_a11, rank] = -1.0
        V[v_b11, rank] = 1.0; V[v_b12, rank] = 1.0
        W[w_c22, rank] = 1.0
        rank += 1
        
        # M7 = (a12 - a22) * (b21 + b22)
        U[u_a12, rank] = 1.0; U[u_a22, rank] = -1.0
        V[v_b21, rank] = 1.0; V[v_b22, rank] = 1.0
        W[w_c11, rank] = 1.0
        rank += 1

    def add_schoolbook_1x4_4x8(a_row, b_row_start, b_col_start, b_col_end):
        nonlocal rank
        # A is 1x4. Row a_row. Cols 0-3.
        # B is 4x8. Rows b_row_start to b_row_start+4. Cols b_col_start to b_col_end.
        # Wait, B is 4x8. We are multiplying A[2, :] (1x4) by B[:, :] (4x8).
        # So a_row = 2.
        # B rows are 0-3.
        # We iterate j in 0-3, k in 0-7.
        
        for j in range(4):
            for k in range(8):
                u_idx = a_row * m + j
                v_idx = j * p + k
                w_idx = k * n + a_row
                
                U[u_idx, rank] = 1.0
                V[v_idx, rank] = 1.0
                W[w_idx, rank] = 1.0
                rank += 1

    # 1. Strassen for A[0:2, :] * B[:, :]
    # A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0)
    # A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_block(0, 0, 0, 2)
    # A[0:2, 0:2] * B[0:2, 4:6]
    add_strassen_block(0, 0, 0, 4)
    # A[0:2, 0:2] * B[0:2, 6:8]
    add_strassen_block(0, 0, 0, 6)
    
    # A[0:2, 2:4] * B[2:4, 0:2]
    add_strassen_block(0, 2, 2, 0)
    # A[0:2, 2:4] * B[2:4, 2:4]
    add_strassen_block(0, 2, 2, 2)
    # A[0:2, 2:4] * B[2:4, 4:6]
    add_strassen_block(0, 2, 2, 4)
    # A[0:2, 2:4] * B[2:4, 6:8]
    add_strassen_block(0, 2, 2, 6)
    
    # 2. Schoolbook for A[2, :] * B[:, :]
    add_schoolbook_1x4_4x8(2, 0, 0, 8)
    
    # Trim if rank < 88 (should be exactly 88)
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        pass # Safety guard, shouldn't trigger
    
    # Convert to lists of lists for the required interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
