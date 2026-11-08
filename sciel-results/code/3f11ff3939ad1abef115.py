"""Rewritten program for 2x5x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    
    # We'll build the decomposition using block partitioning
    # A is n×m = 2×5, B is m×p = 5×6, C is n×p = 2×6
    # Partition middle dimension: 5 = 2 + 2 + 1
    # So A = [A1 | A2 | A3] where A1,A2 are 2×2, A3 is 2×1
    # B = [B1; B2; B3] where B1,B2 are 2×6, B3 is 1×6
    # C = A1*B1 + A2*B2 + A3*B3
    
    # For each 2×2 × 2×6 block, use Strassen (rank 7 per 2×2×2, so 3*7=21 per block)
    # For 2×1 × 1×6, rank 1
    
    # Strassen 2×2×2 decomposition:
    # Given X (2×2), Y (2×2), Z = X*Y
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y10-Y11)
    # M4 = X11*(Y01-Y10)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y10)
    # M7 = (X01-X11)*(Y01+Y11)
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    # For 2×2 × 2×p: we have p/2 column pairs, each needing 7 terms
    # Total rank for 2×2×p = 7 * (p//2) if p is even
    
    # Build U, V, W matrices
    # U shape: (n*m, rank) = (10, rank)
    # V shape: (m*p, rank) = (30, rank)
    # W shape: (p*n, rank) = (12, rank)
    
    # Let's compute the total rank
    rank_A1B1 = 7 * 3  # 2×2 × 2×6: 3 column pairs, each with 7 Strassen terms
    rank_A2B2 = 7 * 3  # same
    rank_A3B3 = 1      # 2×1 × 1×6: rank 1
    
    total_rank = rank_A1B1 + rank_A2B2 + rank_A3B3  # 21 + 21 + 1 = 43
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # Helper: add Strassen 2×2×2 decomposition for columns c0, c1 of B
    # A block is 2×2 at rows 0-1, cols start_col to start_col+1
    # B block is 2×2 at rows start_row to start_row+1, cols c0 to c1
    # C block is 2×2 at rows 0-1, cols c0 to c1
    
    def add_strassen_2x2x2(a_row_start, a_col_start, b_row_start, c_col_start, col_off):
        """Add 7 Strassen terms for 2×2 × 2×2 multiplication."""
        # A indices: (i, j) where i in {0,1}, j in {a_col_start, a_col_start+1}
        # B indices: (j, k) where j in {b_row_start, b_row_start+1}, k in {c_col_start, c_col_start+1}
        # C indices: (i, k) where i in {0,1}, k in {c_col_start, c_col_start+1}
        
        # Strassen formulas:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B10)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A11)*(B01+B11)
        
        # Z00 = M1+M4-M5+M7
        # Z01 = M3+M5
        # Z10 = M2+M4
        # Z11 = M1-M2+M3+M6
        
        terms = [
            # (U_coeffs, V_coeffs, W_coeffs) for each of the 7 terms
            # Term 1: M1
            (
                {(0, a_col_start): 1, (1, a_col_start + 1): 1},  # (A00 + A11)
                {(b_row_start, c_col_start): 1, (b_row_start + 1, c_col_start + 1): 1},  # (B00 + B11)
                {(c_col_start, 0): 1, (c_col_start + 1, 0): 1, (c_col_start, 1): 1, (c_col_start + 1, 1): 1}  # contributes to all 4 C entries
            ),
            # Term 2: M2
            (
                {(1, a_col_start): 1, (1, a_col_start + 1): 1},  # (A10 + A11)
                {(b_row_start, c_col_start): 1},  # B00
                {(c_col_start, 1): 1, (c_col_start + 1, 1): 1}  # contributes to C10, C11
            ),
            # Term 3: M3
            (
                {(0, a_col_start): 1},  # A00
                {(b_row_start + 1, c_col_start): 1, (b_row_start + 1, c_col_start + 1): -1},  # (B10 - B11)
                {(c_col_start + 1, 0): 1, (c_col_start + 1, 1): 1}  # contributes to C01, C11
            ),
            # Term 4: M4
            (
                {(1, a_col_start + 1): 1},  # A11
                {(b_row_start, c_col_start + 1): 1, (b_row_start + 1, c_col_start): -1},  # (B01 - B10)
                {(c_col_start, 0): 1, (c_col_start + 1, 0): 1, (c_col_start, 1): 1, (c_col_start + 1, 1): 1}  # contributes to all 4
            ),
            # Term 5: M5
            (
                {(0, a_col_start): 1, (0, a_col_start + 1): 1},  # (A00 + A01)
                {(b_row_start + 1, c_col_start + 1): 1},  # B11
                {(c_col_start, 0): -1, (c_col_start + 1, 0): -1, (c_col_start + 1, 1): 1}  # -C00, -C10, +C11
            ),
            # Term 6: M6
            (
                {(1, a_col_start): 1, (0, a_col_start): -1},  # (A10 - A00)
                {(b_row_start, c_col_start): 1, (b_row_start + 1, c_col_start): 1},  # (B00 + B10)
                {(c_col_start + 1, 1): 1}  # +C11
            ),
            # Term 7: M7
            (
                {(0, a_col_start + 1): 1, (1, a_col_start + 1): -1},  # (A01 - A11)
                {(b_row_start, c_col_start + 1): 1, (b_row_start + 1, c_col_start + 1): 1},  # (B01 + B11)
                {(c_col_start, 0): 1, (c_col_start + 1, 1): 1}  # +C00, +C11
            ),
        ]
        
        for t, (u_dict, v_dict, w_dict) in enumerate(terms):
            col = col_off + t
            for (i, j), val in u_dict.items():
                U[i * m + j, col] += val
            for (j, k), val in v_dict.items():
                V[j * p + k, col] += val
            for (k, i), val in w_dict.items():
                W[k * n + i, col] += val
    
    # Block 1: A1 (2×2, cols 0-1) * B1 (2×6, rows 0-1)
    # Process 3 column pairs: (0,1), (2,3), (4,5)
    for c_pair in range(3):
        c0 = c_pair * 2
        add_strassen_2x2x2(0, 0, 0, c0, col_offset)
        col_offset += 7
    
    # Block 2: A2 (2×2, cols 2-3) * B2 (2×6, rows 2-3)
    for c_pair in range(3):
        c0 = c_pair * 2
        add_strassen_2x2x2(0, 2, 2, c0, col_offset)
        col_offset += 7
    
    # Block 3: A3 (2×1, col 4) * B3 (1×6, row 4)
    # Rank 1: U[i*5+4, col] = 1, V[4*6+k, col] = 1, W[k*2+i, col] = 1
    col = col_offset
    for i in range(2):
        U[i * m + 4, col] = 1.0
    for k in range(6):
        V[4 * p + k, col] = 1.0
    for k in range(6):
        for i in range(2):
            W[k * n + i, col] = 1.0
    col_offset += 1
    
    # Verify the decomposition
    # Check that for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    max_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(total_rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                error = abs(total - 1.0)
                if error > max_error:
                    max_error = error
    
    if max_error > 1e-10:
        # Fallback to trivial decomposition
        trivial_rank = n * m * p
        U = np.zeros((n * m, trivial_rank))
        V = np.zeros((m * p, trivial_rank))
        W = np.zeros((p * n, trivial_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
