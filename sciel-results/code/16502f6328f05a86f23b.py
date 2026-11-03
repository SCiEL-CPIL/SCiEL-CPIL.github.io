"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Strategy: Strassen decomposition for 2x4 * 4x6
    # A (2x4) split into [A0 | A1] where A0, A1 are 2x2
    # B (4x6) split into [B_top; B_bot] where B_top, B_bot are 2x6
    # C = A0 * B_top + A1 * B_bot
    # Each 2x2 * 2x6 uses 3 Strassen 2x2x2 multiplications
    # Total: 6 * 7 = 42 rank
    
    rank = 42
    U = np.zeros((8, rank), dtype=float)
    V = np.zeros((24, rank), dtype=float)
    W = np.zeros((12, rank), dtype=float)
    
    # Strassen for 2x2 * 2x2 (P * Q = R):
    # M1 = (P00+P11)*(Q00+Q11)  -> R00 += M1, R11 += M1
    # M2 = (P10+P11)*Q00        -> R10 += M2, R11 -= M2
    # M3 = P00*(Q01-Q11)        -> R01 += M3, R11 += M3
    # M4 = P11*(Q10-Q00)        -> R00 += M4, R10 += M4
    # M5 = (P00+P01)*Q11        -> R00 -= M5, R01 += M5
    # M6 = (P10-P00)*(Q00+Q01)  -> R11 += M6
    # M7 = (P01-P11)*(Q10+Q11)  -> R00 += M7
    
    col_idx = 0
    for source in range(2):  # 0=top (B[0:2,:]), 1=bot (B[2:4,:])
        for c in range(3):   # column block 0,1,2 of the 6 columns
            for t in range(7):  # 7 Strassen terms
                # U indices: A is 2x4, row i, col j -> U[i*4+j]
                # V indices: B is 4x6, row i, col j -> V[i*6+j]
                # W indices: C is 2x6, row i, col j -> W[i*6+j]
                
                if t == 0:
                    # M1 = (P00+P11)*(Q00+Q11)
                    U[0*4 + (0 + 2*source), col_idx] += 1.0
                    U[1*4 + (1 + 2*source), col_idx] += 1.0
                    V[(0 + 2*source)*6 + 2*c, col_idx] += 1.0
                    V[(1 + 2*source)*6 + 2*c+1, col_idx] += 1.0
                    W[0*6 + 2*c, col_idx] += 1.0
                    W[1*6 + 2*c+1, col_idx] += 1.0
                elif t == 1:
                    # M2 = (P10+P11)*Q00
                    U[1*4 + (0 + 2*source), col_idx] += 1.0
                    U[1*4 + (1 + 2*source), col_idx] += 1.0
                    V[(0 + 2*source)*6 + 2*c, col_idx] += 1.0
                    W[1*6 + 2*c, col_idx] += 1.0
                    W[1*6 + 2*c+1, col_idx] += -1.0
                elif t == 2:
                    # M3 = P00*(Q01-Q11)
                    U[0*4 + (0 + 2*source), col_idx] += 1.0
                    V[(0 + 2*source)*6 + 2*c+1, col_idx] += 1.0
                    V[(1 + 2*source)*6 + 2*c+1, col_idx] += -1.0
                    W[0*6 + 2*c+1, col_idx] += 1.0
                    W[1*6 + 2*c+1, col_idx] += 1.0
                elif t == 3:
                    # M4 = P11*(Q10-Q00)
                    U[1*4 + (1 + 2*source), col_idx] += 1.0
                    V[(1 + 2*source)*6 + 2*c, col_idx] += 1.0
                    V[(0 + 2*source)*6 + 2*c, col_idx] += -1.0
                    W[0*6 + 2*c, col_idx] += 1.0
                    W[1*6 + 2*c, col_idx] += 1.0
                elif t == 4:
                    # M5 = (P00+P01)*Q11
                    U[0*4 + (0 + 2*source), col_idx] += 1.0
                    U[0*4 + (1 + 2*source), col_idx] += 1.0
                    V[(1 + 2*source)*6 + 2*c+1, col_idx] += 1.0
                    W[0*6 + 2*c, col_idx] += -1.0
                    W[0*6 + 2*c+1, col_idx] += 1.0
                elif t == 5:
                    # M6 = (P10-P00)*(Q00+Q01)
                    U[1*4 + (0 + 2*source), col_idx] += 1.0
                    U[0*4 + (0 + 2*source), col_idx] += -1.0
                    V[(0 + 2*source)*6 + 2*c, col_idx] += 1.0
                    V[(0 + 2*source)*6 + 2*c+1, col_idx] += 1.0
                    W[1*6 + 2*c+1, col_idx] += 1.0
                elif t == 6:
                    # M7 = (P01-P11)*(Q10+Q11)
                    U[0*4 + (1 + 2*source), col_idx] += 1.0
                    U[1*4 + (1 + 2*source), col_idx] += -1.0
                    V[(1 + 2*source)*6 + 2*c, col_idx] += 1.0
                    V[(1 + 2*source)*6 + 2*c+1, col_idx] += 1.0
                    W[0*6 + 2*c, col_idx] += 1.0
                
                col_idx += 1
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

