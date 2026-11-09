"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition using Strassen for 2x2x2 blocks
    # Structure:
    # A is 3x4. Split into A_top (2x4) and A_bot (1x4).
    # B is 4x4.
    # C is 3x4. C_top (2x4) and C_bot (1x4).
    
    # A_top * B contributes to C_top.
    # A_bot * B contributes to C_bot.
    
    # C_top = A_top * B.
    # A_top is 2x4. B is 4x4.
    # Split A_top into A_top_left (2x2) and A_top_right (2x2).
    # Split B into B_tl (2x2), B_tr (2x2), B_bl (2x2), B_br (2x2).
    # C_top_left = A_top_left * B_tl + A_top_right * B_bl
    # C_top_right = A_top_left * B_tr + A_top_right * B_br
    
    # Each term like A_top_left * B_tl is a 2x2x2 multiplication.
    # We use Strassen for each 2x2x2 block.
    # There are 4 such blocks for C_top.
    # Each Strassen block has rank 7.
    # Total rank for C_top = 4 * 7 = 28.
    
    # C_bot = A_bot * B.
    # A_bot is 1x4.
    # Split A_bot into A_bot_left (1x2) and A_bot_right (1x2).
    # C_bot_left = A_bot_left * B_tl + A_bot_right * B_bl
    # C_bot_right = A_bot_left * B_tr + A_bot_right * B_br
    
    # Each term like A_bot_left * B_tl is a 1x2x2 multiplication.
    # Rank of 1x2x2 is 4 (schoolbook).
    # There are 4 such blocks for C_bot.
    # Total rank for C_bot = 4 * 4 = 16.
    
    # Total rank = 28 + 16 = 44.

    rank = 44
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # i_off: row offset in A (0 or 1 for top part)
    # j_off: col offset in A (0 or 2)
    # k_off: col offset in B (0 or 2)
    # Note: The "row" of B corresponds to "col" of A in the multiplication chain?
    # A (n x m) * B (m x p) -> C (n x p)
    # Indices:
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        # Local indices for A block (2x2):
        # (i_off, j_off), (i_off, j_off+1)
        # (i_off+1, j_off), (i_off+1, j_off+1)
        # Local indices for B block (2x2):
        # (j_off, k_off), (j_off, k_off+1)
        # (j_off+1, k_off), (j_off+1, k_off+1)
        # Local indices for C block (2x2):
        # (i_off, k_off), (i_off, k_off+1)
        # (i_off+1, k_off), (i_off+1, k_off+1)
        
        # Strassen terms for M = A * B:
        # M1 = (A00 + A11)(B00 + B11) -> C00, C11
        # M2 = (A10 + A11)B00 -> C10, C11
        # M3 = A00(B10 - B00) -> C00, C01
        # M4 = A11(B01 - B11) -> C00, C10  -- Wait, standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's map these to U, V, W.
        # U indices:
        # A00: (i_off, j_off)
        # A01: (i_off, j_off+1)
        # A10: (i_off+1, j_off)
        # A11: (i_off+1, j_off+1)
        
        # V indices:
        # B00: (j_off, k_off)
        # B01: (j_off, k_off+1)
        # B10: (j_off+1, k_off)
        # B11: (j_off+1, k_off+1)
        
        # W indices:
        # C00: (i_off, k_off)
        # C01: (i_off, k_off+1)
        # C10: (i_off+1, k_off)
        # C11: (i_off+1, k_off+1)
        
        # 1. M1 = (A00 + A11)(B00 + B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        U[(i_off)*m + (j_off), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off)*p + (k_off), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off)*n + (i_off), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # 2. M2 = (A10 + A11)B00
        # U: A10=1, A11=1
        # V: B00=1
        # W: C10=1, C11=-1
        U[(i_off+1)*m + (j_off), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off)*p + (k_off), col] = 1
        W[(k_off)*n + (i_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1
        
        # 3. M3 = A00(B10 - B00)
        # U: A00=1
        # V: B10=1, B00=-1
        # W: C00=1, C01=1
        U[(i_off)*m + (j_off), col] = 1
        V[(j_off+1)*p + (k_off), col] = 1
        V[(j_off)*p + (k_off), col] = -1
        W[(k_off)*n + (i_off), col] = 1
        W[(k_off+1)*n + (i_off), col] = 1
        col += 1
        
        # 4. M4 = A11(B01 - B11)
        # U: A11=1
        # V: B01=1, B11=-1
        # W: C00=1, C10=1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off)*p + (k_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off)*n + (i_off), col] = 1
        W[(k_off)*n + (i_off+1), col] = 1
        col += 1
        
        # 5. M5 = (A00 + A01)B11
        # U: A00=1, A01=1
        # V: B11=1
        # W: C00=-1, C01=1
        U[(i_off)*m + (j_off), col] = 1
        U[(i_off)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off)*n + (i_off), col] = -1
        W[(k_off+1)*n + (i_off), col] = 1
        col += 1
        
        # 6. M6 = (A10 - A00)(B00 + B01)
        # U: A10=1, A00=-1
        # V: B00=1, B01=1
        # W: C11=1
        U[(i_off+1)*m + (j_off), col] = 1
        U[(i_off)*m + (j_off), col] = -1
        V[(j_off)*p + (k_off), col] = 1
        V[(j_off)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # 7. M7 = (A01 - A11)(B10 + B11)
        # U: A01=1, A11=-1
        # V: B10=1, B11=1
        # W: C00=1
        U[(i_off)*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + (k_off), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off)*n + (i_off), col] = 1
        col += 1

    # Helper to add a 1x2x2 block (rank 4)
    # i_off: row offset in A (2 for bottom part)
    # j_off: col offset in A (0 or 2)
    # k_off: col offset in B (0 or 2)
    def add_122(i_off, j_off, k_off):
        nonlocal col
        # A block (1x2): A00, A01
        # B block (2x2): B00, B01, B10, B11
        # C block (1x2): C00, C01
        
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        
        # Term 1: A00*B00 -> C00
        U[(i_off)*m + (j_off), col] = 1
        V[(j_off)*p + (k_off), col] = 1
        W[(k_off)*n + (i_off), col] = 1
        col += 1
        
        # Term 2: A01*B10 -> C00
        U[(i_off)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off), col] = 1
        W[(k_off)*n + (i_off), col] = 1
        col += 1
        
        # Term 3: A00*B01 -> C01
        U[(i_off)*m + (j_off), col] = 1
        V[(j_off)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off), col] = 1
        col += 1
        
        # Term 4: A01*B11 -> C01
        U[(i_off)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off), col] = 1
        col += 1

    # Top part: A[0:2, :] * B[:, :]
    # A_top is 2x4.
    # B is 4x4.
    
    # Block 1: A_top[:, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # i_off=0, j_off=0, k_off=0
    add_strassen_222(0, 0, 0)
    
    # Block 2: A_top[:, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # i_off=0, j_off=2, k_off=0
    add_strassen_222(0, 2, 0)
    
    # Block 3: A_top[:, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # i_off=0, j_off=0, k_off=2
    add_strassen_222(0, 0, 2)
    
    # Block 4: A_top[:, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # i_off=0, j_off=2, k_off=2
    add_strassen_222(0, 2, 2)
    
    # Bottom part: A[2, :] * B[:, :]
    # A_bot is 1x4.
    
    # Block 5: A_bot[0:2] * B[0:2, 0:2] -> C[2, 0:2]
    # i_off=2, j_off=0, k_off=0
    add_122(2, 0, 0)
    
    # Block 6: A_bot[2:4] * B[2:4, 0:2] -> C[2, 0:2]
    # i_off=2, j_off=2, k_off=0
    add_122(2, 2, 0)
    
    # Block 7: A_bot[0:2] * B[0:2, 2:4] -> C[2, 2:4]
    # i_off=2, j_off=0, k_off=2
    add_122(2, 0, 2)
    
    # Block 8: A_bot[2:4] * B[2:4, 2:4] -> C[2, 2:4]
    # i_off=2, j_off=2, k_off=2
    add_122(2, 2, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
