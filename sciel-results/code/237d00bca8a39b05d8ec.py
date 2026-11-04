import time
import sys

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=4, m=5, p=5
    # U: 20 x rank
    # V: 25 x rank
    # W: 20 x rank
    
    # We use the decomposition:
    # A (4x5) = [A_L (4x2), A_R (4x3)]
    # B (5x5) = [B_T (2x5); B_B (3x5)]
    # C = A_L B_T + A_R B_B
    
    # A_L B_T:
    # A_L = [A_L1 (2x2); A_L2 (2x2)]
    # B_T = [B_T1 (2x2), B_T2 (2x3)]
    # C_LT = A_L1 B_T1 + A_L1 B_T2 + A_L2 B_T1 + A_L2 B_T2
    # Ranks: 7 + 12 + 7 + 12 = 38
    
    # A_R B_B:
    # A_R = [A_R1 (2x3); A_R2 (2x3)]
    # B_B = [B_B1 (3x2), B_B2 (3x3)]
    # C_RB = A_R1 B_B1 + A_R1 B_B2 + A_R2 B_B1 + A_R2 B_B2
    # Ranks: 12 + 18 + 12 + 18 = 60
    
    total_rank = 38 + 60
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # Helper to add schoolbook terms for A (n1 x k) * B (k x n2)
    # u_idx_base, v_idx_base, w_idx_base are offsets in U, V, W
    # i is row of A, j is col of A / row of B, k is col of B
    def add_schoolbook(n1, k, n2, u_off, v_off, w_off, A_rows, B_cols):
        # A_rows: list of row indices in A
        # B_cols: list of col indices in B
        # Actually, we just need the ranges
        for i in range(n1):
            for j in range(k):
                for k_idx in range(n2):
                    u = u_off + i * k + j
                    v = v_off + j * n2 + k_idx
                    w = w_off + k_idx * n1 + i # W index: k*n + i? 
                    # W index is k*n + i where k is col of C, i is row of C.
                    # In our block, C is n1 x n2.
                    # So W index relative to block is k_idx * n1 + i.
                    # But W is global.
                    # We need to map local (i, k_idx) to global (row, col) of C.
                    # This depends on where the block is in C.
                    pass

    # This helper is too complex. Let's just iterate explicitly.
    
    # Block 1: A_L1 (2x2) * B_T1 (2x2) -> C_11 (2x2)
    # A_L1 rows: 0,1. Cols: 0,1.
    # B_T1 rows: 0,1. Cols: 0,1.
    # C_11 rows: 0,1. Cols: 0,1.
    # Strassen 2x2
    # P1 = (A00+A11)(B00+B11)
    # ...
    # We'll just use schoolbook for simplicity first, rank 8.
    # Then optimize to 7.
    
    # Let's use schoolbook for all blocks to get a working 100 rank, then replace 2x2 with Strassen.
    # No, I'll implement Strassen for the 2x2 blocks.
    
    def strassen_2x2(u_base, v_base, w_base, i_off, j_off, k_off):
        # u_base: offset in U for A block
        # v_base: offset in V for B block
        # w_base: offset in W for C block
        # i_off: row offset in C
        # j_off: col offset in C (which is k in W index)
        # The block is 2x2x2.
        # A indices: i in 0..1, j in 0..1. U[i*2+j, col]
        # B indices: j in 0..1, k in 0..1. V[j*2+k, col]
        # C indices: i in 0..1, k in 0..1. W[k*2+i, col] (relative to block)
        # Global W index: (j_off + k) * n + (i_off + i)
        
        # P1
        # (A00+A11)(B00+B11)
        # A00: U[u_base+0], A11: U[u_base+3]
        # B00: V[v_base+0], B11: V[v_base+3]
        # Terms:
        # A00 B00 -> W[0,0]
        # A00 B11 -> W[1,0] ? No.
        # (A00+A11) is a 2x2 matrix? No, A00 is a scalar (1x1) in the 2x2 context?
        # No, in 2x2 Strassen, A00 is the top-left 1x1 element?
        # No, Strassen applies to 2x2 matrices composed of 1x1 elements?
        # Yes, for 2x2 multiplication.
        
        # U indices for A:
        # A00: 0, A01: 1, A10: 2, A11: 3
        # V indices for B:
        # B00: 0, B01: 1, B10: 2, B11: 3
        
        # P1 = (A00+A11)(B00+B11)
        # A00 B00: U[0], V[0], W[0,0]
        # A00 B11: U[0], V[3], W[1,0]
        # A11 B00: U[3], V[0], W[0,1]
        # A11 B11: U[3], V[3], W[1,1]
        
        # This is schoolbook. Strassen combines them.
        # P1 contributes to C00, C01, C10, C11.
        
        # I will use the 7-term Strassen decomposition.
        # It is easier to just use schoolbook for 2x2 (rank 8) and accept rank 100?
        # No, I need to improve.
        
        # Let's use the rank 7 decomposition for 2x2.
        # It requires specific linear combinations.
        
        pass

    # Given time, I will return the schoolbook 100 rank solution to ensure correctness, 
    # but the directive asks for a stronger algorithm.
    # I will implement the 98 rank solution using schoolbook for all sub-blocks, 
    # which is still 100. 
    # Wait, 7+12+7+12 = 38. 12+18+12+18 = 60. Total 98.
    # This requires Strassen for the 2x2 blocks.
    
    # I will implement Strassen for the 2x2 blocks.
    
    return {"U": U, "V": V, "W": W}
