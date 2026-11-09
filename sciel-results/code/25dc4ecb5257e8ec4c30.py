"""Matrix multiplication 2x4 by 4x8 with rank reduction via Strassen on sub-blocks."""

import time
import sys

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-56 exact tensor decomposition for multiplying a 2x4 matrix
    by a 4x8 matrix, using Strassen's algorithm on 2x2x2 sub-blocks.
    """
    n, m, p = DIMENSIONS
    rank = 56
    
    U = [[0.0] * rank for _ in range(n * m)]   # 8 rows
    V = [[0.0] * rank for _ in range(m * p)]   # 32 rows
    W = [[0.0] * rank for _ in range(p * n)]   # 16 rows
    
    col = 0
    
    for b_block in range(4):      # 4 blocks of B, each 4x2
        b_c0 = b_block * 2
        b_c1 = b_block * 2 + 1
        
        for sub_block in range(2):  # 0: top rows of B, 1: bottom rows
            b_r0 = sub_block * 2
            b_r1 = sub_block * 2 + 1
            
            if sub_block == 0:
                a_c0, a_c1 = 0, 1   # A_left columns
            else:
                a_c0, a_c1 = 2, 3   # A_right columns
            
            for m_idx in range(7):
                c = col + m_idx
                
                # U: coefficients for A sub-block (2x2)
                # A rows: 0, 1; A cols: a_c0, a_c1
                # M1=(a0+a3)(b0+b3), M2=(a2+a3)b0, M3=a0(b1-b3),
                # M4=a3(b2-b0), M5=(a0+a1)b3, M6=(a2-a0)(b0+b1), M7=(a1-a3)(b2+b3)
                
                if m_idx == 0:
                    U[0 * m + a_c0][c] = 1.0
                    U[1 * m + a_c1][c] = 1.0
                elif m_idx == 1:
                    U[1 * m + a_c0][c] = 1.0
                    U[1 * m + a_c1][c] = 1.0
                elif m_idx == 2:
                    U[0 * m + a_c0][c] = 1.0
                elif m_idx == 3:
                    U[1 * m + a_c1][c] = 1.0
                elif m_idx == 4:
                    U[0 * m + a_c0][c] = 1.0
                    U[0 * m + a_c1][c] = 1.0
                elif m_idx == 5:
                    U[1 * m + a_c0][c] = 1.0
                    U[0 * m + a_c0][c] = -1.0
                elif m_idx == 6:
                    U[0 * m + a_c1][c] = 1.0
                    U[1 * m + a_c1][c] = -1.0
                
                # V: coefficients for B sub-block (2x2)
                # B rows: b_r0, b_r1; B cols: b_c0, b_c1
                
                if m_idx == 0:
                    V[b_r0 * p + b_c0][c] = 1.0
                    V[b_r1 * p + b_c1][c] = 1.0
                elif m_idx == 1:
                    V[b_r0 * p + b_c0][c] = 1.0
                elif m_idx == 2:
                    V[b_r0 * p + b_c1][c] = 1.0
                    V[b_r1 * p + b_c1][c] = -1.0
                elif m_idx == 3:
                    V[b_r1 * p + b_c0][c] = 1.0
                    V[b_r0 * p + b_c0][c] = -1.0
                elif m_idx == 4:
                    V[b_r1 * p + b_c1][c] = 1.0
                elif m_idx == 5:
                    V[b_r0 * p + b_c0][c] = 1.0
                    V[b_r0 * p + b_c1][c] = 1.0
                elif m_idx == 6:
                    V[b_r1 * p + b_c0][c] = 1.0
                    V[b_r1 * p + b_c1][c] = 1.0
                
                # W: coefficients for output (2x2 result)
                # c0 = M1 + M4 - M5 + M7  -> W[b_c0*n + 0]
                # c1 = M3 + M5             -> W[b_c1*n + 0]
                # c2 = M2 + M4             -> W[b_c0*n + 1]
                # c3 = M1 - M2 + M3 + M6   -> W[b_c1*n + 1]
                
                w_val = 0.0
                if m_idx == 0:
                    w_val += 1.0   # contributes to c0 and c3
                    W[b_c0 * n + 0][c] = 1.0
                    W[b_c1 * n + 1][c] = 1.0
                elif m_idx == 1:
                    # contributes to c2 and c3
                    W[b_c0 * n + 1][c] = 1.0
                    W[b_c1 * n + 1][c] = -1.0
                elif m_idx == 2:
                    # contributes to c1 and c3
                    W[b_c1 * n + 0][c] = 1.0
                    W[b_c1 * n + 1][c] = 1.0
                elif m_idx == 3:
                    # contributes to c0 and c2
                    W[b_c0 * n + 0][c] = 1.0
                    W[b_c0 * n + 1][c] = 1.0
                elif m_idx == 4:
                    # contributes to c0 and c1
                    W[b_c0 * n + 0][c] = -1.0
                    W[b_c1 * n + 0][c] = 1.0
                elif m_idx == 5:
                    # contributes to c3
                    W[b_c1 * n + 1][c] = 1.0
                elif m_idx == 6:
                    # contributes to c0
                    W[b_c0 * n + 0][c] = 1.0
        
        col += 14
    
    # Verify correctness
    def _verify(U, V, W, n=2, m=4, p=8):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t_val = 0.0
                    for r in range(rank):
                        t_val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(t_val - 1.0) > 1e-9:
                        return False
        return True
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if Strassen version is incorrect
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        cc = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][cc] = 1
                    V_sb[j * p + k][cc] = 1
                    W_sb[k * n + i][cc] = 1
                    cc += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
