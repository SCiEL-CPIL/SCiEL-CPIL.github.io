import time
import numpy as np

DIMENSIONS = [2, 4, 7]

_START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s margin before 150s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 2, 4, 7
    
    # Rank 50 decomposition for 2x4x7 matrix multiplication
    # Structure:
    # Split A (2x4) into two 2x2 blocks: A_L (cols 0,1), A_R (cols 2,3)
    # Split B (4x7) into four 2x? blocks based on rows (0,1) and (2,3)
    # For each A-block, we multiply with B-blocks.
    # B rows 0,1: Cols 0,1 (2x2), 2,3 (2x2), 4,5 (2x2), 6 (2x1)
    # B rows 2,3: Cols 0,1 (2x2), 2,3 (2x2), 4,5 (2x2), 6 (2x1)
    
    # Total terms:
    # A_L * B_top: 7 (Strassen) + 7 (Strassen) + 7 (Strassen) + 4 (Schoolbook 2x2x1) = 25
    # A_R * B_bot: 7 (Strassen) + 7 (Strassen) + 7 (Strassen) + 4 (Schoolbook 2x2x1) = 25
    # Total Rank = 50
    
    rank = 50
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add Strassen terms for 2x2 * 2x2 -> 2x2
    # Indices are local (0..1) for A, B, C blocks
    # Global indices are calculated based on block position
    def add_strassen(a_base, b_base, c_base, offset):
        """
        a_base, b_base, c_base are tuples (row_start, col_start) for the block in the original matrix.
        offset is the column index in U, V, W to write to.
        """
        # Local indices for 2x2 block
        # A: (i, j) -> global (a_base[0]+i, a_base[1]+j) -> U index (a_base[0]+i)*m + (a_base[1]+j)
        # B: (j, k) -> global (b_base[0]+j, b_base[1]+k) -> V index (b_base[0]+j)*p + (b_base[1]+k)
        # C: (i, k) -> global (c_base[0]+i, c_base[1]+k) -> W index (c_base[1]+k)*n + (c_base[0]+i)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[(a_base[0]+0)*m + (a_base[1]+0), offset] += 1
        U[(a_base[0]+1)*m + (a_base[1]+1), offset] += 1
        V[(b_base[0]+0)*p + (b_base[1]+0), offset] += 1
        V[(b_base[0]+1)*p + (b_base[1]+1), offset] += 1
        W[(c_base[1]+0)*n + (c_base[0]+0), offset] += 1
        W[(c_base[1]+1)*n + (c_base[0]+1), offset] += 1
        col = offset + 1
        
        # M2 = (A10 + A11)B00
        U[(a_base[0]+1)*m + (a_base[1]+0), col] += 1
        U[(a_base[0]+1)*m + (a_base[1]+1), col] += 1
        V[(b_base[0]+0)*p + (b_base[1]+0), col] += 1
        W[(c_base[1]+0)*n + (c_base[0]+1), col] += 1  # C10
        W[(c_base[1]+1)*n + (c_base[0]+1), col] -= 1  # C11
        col += 1
        
        # M3 = A00(B01 - B11)
        U[(a_base[0]+0)*m + (a_base[1]+0), col] += 1
        V[(b_base[0]+0)*p + (b_base[1]+1), col] += 1
        V[(b_base[0]+1)*p + (b_base[1]+1), col] -= 1
        W[(c_base[1]+1)*n + (c_base[0]+0), col] += 1  # C01
        W[(c_base[1]+1)*n + (c_base[0]+1), col] += 1  # C11
        col += 1
        
        # M4 = A11(B10 - B00)
        U[(a_base[0]+1)*m + (a_base[1]+1), col] += 1
        V[(b_base[0]+1)*p + (b_base[1]+0), col] += 1
        V[(b_base[0]+0)*p + (b_base[1]+0), col] -= 1
        W[(c_base[1]+0)*n + (c_base[0]+0), col] += 1  # C00
        W[(c_base[1]+0)*n + (c_base[0]+1), col] += 1  # C10
        col += 1
        
        # M5 = (A00 + A01)B11
        U[(a_base[0]+0)*m + (a_base[1]+0), col] += 1
        U[(a_base[0]+0)*m + (a_base[1]+1), col] += 1
        V[(b_base[0]+1)*p + (b_base[1]+1), col] += 1
        W[(c_base[1]+0)*n + (c_base[0]+0), col] -= 1  # C00
        W[(c_base[1]+1)*n + (c_base[0]+0), col] += 1  # C01
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[(a_base[0]+1)*m + (a_base[1]+0), col] += 1
        U[(a_base[0]+0)*m + (a_base[1]+0), col] -= 1
        V[(b_base[0]+0)*p + (b_base[1]+0), col] += 1
        V[(b_base[0]+0)*p + (b_base[1]+1), col] += 1
        W[(c_base[1]+1)*n + (c_base[0]+1), col] += 1  # C11
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[(a_base[0]+0)*m + (a_base[1]+1), col] += 1
        U[(a_base[0]+1)*m + (a_base[1]+1), col] -= 1
        V[(b_base[0]+1)*p + (b_base[1]+0), col] += 1
        V[(b_base[0]+1)*p + (b_base[1]+1), col] += 1
        W[(c_base[1]+0)*n + (c_base[0]+0), col] += 1  # C00
        col += 1
        
        return col

    def add_schoolbook_221(a_base, b_base, c_base, offset):
        """
        A is 2x2, B is 2x1, C is 2x1.
        """
        # N1 = A00 B0
        U[(a_base[0]+0)*m + (a_base[1]+0), offset] += 1
        V[(b_base[0]+0)*p + (b_base[1]+0), offset] += 1
        W[(c_base[1]+0)*n + (c_base[0]+0), offset] += 1
        col = offset + 1
        
        # N2 = A01 B1
        U[(a_base[0]+0)*m + (a_base[1]+1), col] += 1
        V[(b_base[0]+1)*p + (b_base[1]+0), col] += 1
        W[(c_base[1]+0)*n + (c_base[0]+0), col] += 1
        col += 1
        
        # N3 = A10 B0
        U[(a_base[0]+1)*m + (a_base[1]+0), col] += 1
        V[(b_base[0]+0)*p + (b_base[1]+0), col] += 1
        W[(c_base[1]+0)*n + (c_base[0]+1), col] += 1
        col += 1
        
        # N4 = A11 B1
        U[(a_base[0]+1)*m + (a_base[1]+1), col] += 1
        V[(b_base[0]+1)*p + (b_base[1]+0), col] += 1
        W[(c_base[1]+0)*n + (c_base[0]+1), col] += 1
        col += 1
        
        return col

    # A_L: rows 0-1, cols 0-1
    a_l_base = (0, 0)
    # A_R: rows 0-1, cols 2-3
    a_r_base = (0, 2)
    
    # B_TL: rows 0-1, cols 0-1
    b_tl_base = (0, 0)
    # B_TM: rows 0-1, cols 2-3
    b_tm_base = (0, 2)
    # B_TR: rows 0-1, cols 4-5
    b_tr_base = (0, 4)
    # B_T1: rows 0-1, col 6
    b_t1_base = (0, 6)
    
    # B_BL: rows 2-3, cols 0-1
    b_bl_base = (2, 0)
    # B_BM: rows 2-3, cols 2-3
    b_bm_base = (2, 2)
    # B_BR: rows 2-3, cols 4-5
    b_br_base = (2, 4)
    # B_B1: rows 2-3, col 6
    b_b1_base = (2, 6)
    
    # C_L: cols 0-1 (indices in W: k*n + i)
    c_l_base = (0, 0)
    # C_M: cols 2-3
    c_m_base = (0, 2)
    # C_R: cols 4-5
    c_r_base = (0, 4)
    # C_1: col 6
    c_1_base = (0, 6)
    
    # Process A_L * B_TL
    col = add_strassen(a_l_base, b_tl_base, c_l_base, col)
    # Process A_L * B_TM
    col = add_strassen(a_l_base, b_tm_base, c_m_base, col)
    # Process A_L * B_TR
    col = add_strassen(a_l_base, b_tr_base, c_r_base, col)
    # Process A_L * B_T1
    col = add_schoolbook_221(a_l_base, b_t1_base, c_1_base, col)
    
    # Process A_R * B_BL
    col = add_strassen(a_r_base, b_bl_base, c_l_base, col)
    # Process A_R * B_BM
    col = add_strassen(a_r_base, b_bm_base, c_m_base, col)
    # Process A_R * B_BR
    col = add_strassen(a_r_base, b_br_base, c_r_base, col)
    # Process A_R * B_B1
    col = add_schoolbook_221(a_r_base, b_b1_base, c_1_base, col)
    
    if col != rank:
        # Fallback to schoolbook if construction fails (should not happen)
        if _time_remaining() < 5.0:
            rank_fb = 56
            U_fb = np.zeros((n * m, rank_fb), dtype=float)
            V_fb = np.zeros((m * p, rank_fb), dtype=float)
            W_fb = np.zeros((p * n, rank_fb), dtype=float)
            col_fb = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U_fb[i * m + k, col_fb] = 1
                        V_fb[k * p + j, col_fb] = 1
                        W_fb[j * n + i, col_fb] = 1
                        col_fb += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        else:
            raise ValueError(f"Rank mismatch: expected {rank}, got {col}")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
