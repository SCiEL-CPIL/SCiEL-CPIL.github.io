"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    # We use a rank 32 decomposition.
    # Split A into A_L (2x2) and A_R (2x2).
    # Split B into B_L (4x5) and B_R (4x5) is not right.
    # B is 4x5.
    # A_L multiplies first 2 rows of B? No.
    # A is 2x4. A = [A_L, A_R] where A_L, A_R are 2x2.
    # B is 4x5. B = [B_top; B_bottom] where B_top, B_bottom are 2x5.
    # C = A_L B_top + A_R B_bottom.
    # A_L B_top is 2x2 * 2x5. Rank 16.
    # A_R B_bottom is 2x2 * 2x5. Rank 16.
    # Total rank 32.
    
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def add_2x2xn_decomp(a_base, b_base, w_base, start_col):
        # a_base: (i, j) tuple for top-left of A_L or A_R (2x2)
        # b_base: (j, k) tuple for top-left of B_top or B_bottom (2x5)
        # w_base: (k, i) tuple for top-left of C (2x5)
        # start_col: starting column in U, V, W
        
        # A_L indices: (0,0), (0,1), (1,0), (1,1) relative to a_base
        # B_top indices: (0,0)..(0,4), (1,0)..(1,4) relative to b_base
        # C indices: (0,0)..(0,4), (1,0)..(1,4) relative to w_base
        
        col = start_col
        
        # Strassen for first 2 columns of B (cols 0, 1)
        # B_00 = b_base[0,0], B_01 = b_base[0,1]
        # B_10 = b_base[1,0], B_11 = b_base[1,1]
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[u_idx(a_base[0], a_base[1])][col] = 1
        U[u_idx(a_base[0]+1, a_base[1]+1)][col] = 1
        V[v_idx(b_base[0], b_base[1])][col] = 1
        V[v_idx(b_base[0]+1, b_base[1]+1)][col] = 1
        W[w_idx(w_base[0], w_base[1])][col] += 1
        W[w_idx(w_base[0]+1, w_base[1]+1)][col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[u_idx(a_base[0]+1, a_base[1])][col] = 1
        U[u_idx(a_base[0]+1, a_base[1]+1)][col] = 1
        V[v_idx(b_base[0], b_base[1])][col] = 1
        W[w_idx(w_base[0]+1, w_base[1])][col] += 1
        W[w_idx(w_base[0]+1, w_base[1]+1)][col] -= 1
        col += 1
        
        # M3 = A00 * (B01 - B11)
        U[u_idx(a_base[0], a_base[1])][col] = 1
        V[v_idx(b_base[0], b_base[1]+1)][col] = 1
        V[v_idx(b_base[0]+1, b_base[1]+1)][col] = -1
        W[w_idx(w_base[0], w_base[1]+1)][col] += 1
        W[w_idx(w_base[0]+1, w_base[1]+1)][col] += 1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[u_idx(a_base[0]+1, a_base[1]+1)][col] = 1
        V[v_idx(b_base[0]+1, b_base[1])][col] = 1
        V[v_idx(b_base[0], b_base[1])][col] = -1
        W[w_idx(w_base[0], w_base[1])][col] += 1
        W[w_idx(w_base[0]+1, w_base[1])][col] += 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[u_idx(a_base[0], a_base[1])][col] = 1
        U[u_idx(a_base[0], a_base[1]+1)][col] = 1
        V[v_idx(b_base[0]+1, b_base[1]+1)][col] = 1
        W[w_idx(w_base[0], w_base[1])][col] -= 1
        W[w_idx(w_base[0], w_base[1]+1)][col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[u_idx(a_base[0]+1, a_base[1])][col] = 1
        U[u_idx(a_base[0], a_base[1])][col] = -1
        V[v_idx(b_base[0], b_base[1])][col] = 1
        V[v_idx(b_base[0], b_base[1]+1)][col] = 1
        W[w_idx(w_base[0]+1, w_base[1]+1)][col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[u_idx(a_base[0], a_base[1]+1)][col] = 1
        U[u_idx(a_base[0]+1, a_base[1]+1)][col] = -1
        V[v_idx(b_base[0]+1, b_base[1])][col] = 1
        V[v_idx(b_base[0]+1, b_base[1]+1)][col] = 1
        W[w_idx(w_base[0], w_base[1])][col] += 1
        col += 1
        
        # For columns 2, 3, 4 of B (indices 2, 3, 4)
        # We need 3 mults per column.
        # C_0k = A00 B_0k + A01 B_1k
        # C_1k = A10 B_0k + A11 B_1k
        
        # M_8k = A00 B_0k
        # M_9k = A01 B_1k
        # M_10k = (A10 + A11) B_0k + A11 B_1k ? No.
        
        # Let's use:
        # M_8k = A00 B_0k
        # M_9k = A01 B_1k
        # M_10k = A10 B_0k + A11 B_1k (This is 2 mults, but we need 3 total for the column?)
        # If we use 4 mults, rank is 7 + 4*3 = 19.
        # If we use 3 mults, rank is 7 + 3*3 = 16.
        
        # How to do it in 3?
        # C_0k = M_8k + M_9k
        # C_1k = M_10k + M_11k?
        # We need to compute C_0k and C_1k.
        # Maybe M_10k = (A10 + A11) B_0k?
        # Then C_1k = M_10k + A11 B_1k - A10 B_0k? No.
        
        # Actually, the rank 3n+1 decomposition for 2x2xn is:
        # For k >= 2:
        # M_{3k-2} = A00 B_0k
        # M_{3k-1} = A01 B_1k
        # M_{3k} = (A10 + A11) B_0k + A11 B_1k - A10 B_0k ?
        
        # Let's assume we can just add 4 mults per column for simplicity if 3 is hard,
        # but 32 is the target.
        # Let's try to find the 3-mult decomposition.
        # It is known that r(2,2,n) = 3n+1.
        # The decomposition uses:
        # For k=0,1: Strassen (7 mults)
        # For k>=2:
        # M_{3k-2} = A00 B_0k
        # M_{3k-1} = A01 B_1k
        # M_{3k} = A10 B_0k + A11 B_1k (This is 2 mults, but maybe combined?)
        
        # If I use 4 mults per column, rank is 19. Total 38.
        # If I use 3 mults per column, rank is 16. Total 32.
        
        # I will implement the 4-mult version first to ensure correctness, then try to optimize.
        # But 38 > 36.
        
        # Let's try to implement the 3-mult version.
        # C_1k = A10 B_0k + A11 B_1k.
        # Let M_10 = (A10 + A11) B_0k.
        # Let M_11 = A11 B_1k.
        # Then C_1k = M_10 + M_11 - A10 B_0k.
        # We don't have A10 B_0k.
        
        # Alternative:
        # M_8 = A00 B_0k
        # M_9 = A01 B_1k
        # M_10 = (A10 + A11) (B_0k + B_1k)
        # C_0k = M_8 + M_9
        # C_1k = M_10 - A11 B_1k - A10 B_0k ?
        
        # I will use the 4-mult version for now, which gives rank 19 per block, total 38.
        # This is worse than 36.
        
        # I need the 3-mult version.
        # Reference: "Rank of 2x2xn matrix multiplication"
        # The decomposition is:
        # M_1 = (A00+A11)(B00+B11)
        # ... Strassen ...
        # For k >= 2:
        # M_{3k-2} = A00 B_0k
        # M_{3k-1} = A01 B_1k
        # M_{3k} = (A10+A11) B_0k + A11 B_1k - A10 B_0k ?
        
        # Actually, I can just use 4 mults for the first 2 columns (Strassen is 7, which is 3.5 avg)
        # and 3 for the rest.
        
        # Let's assume the 3-mult decomposition is:
        # M_a = A00 B_0k
        # M_b = A01 B_1k
        # M_c = (A10 + A11) B_0k + A11 B_1k - M_a ? No.
        
        # I will use a safe 4-mult decomposition for the extra columns, which gives rank 19.
        # Total rank 38.
        # This is not an improvement.
        
        # I will return the rank 36 solution for now to avoid failure.
        return col

    # Since I cannot guarantee the 3-mult decomposition, I will stick to the rank 36 solution.
    # But I want to try rank 32.
    
    # Let's try to implement the rank 32 solution with the assumption that 3 mults per column is possible.
    # If it fails, I can fall back.
    
    # For now, I will return the rank 36 solution.
    # The current program is rank 36.
    
    # I will just return the current program's logic.
    
    # To save time, I will output the rank 36 solution.
    
    # Actually, I can just copy the current program.
    
    # Let's try to improve by using 4 mults for the 5th column only?
    # No.
    
    # I will return the rank 36 solution.
    
    # Re-using the current logic.
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M2
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
        col += 1
        
        # M3
        U[u_idx(y00[0], y00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M4
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        col += 1
        
        # M5
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y01[0], y01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        col += 1
        
        # M6
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y00[0], y00[1])][col] = -1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M7
        U[u_idx(y01[0], y01[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = -1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        col += 1

    # Block 1: A_L B_11
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y1, z1, w1)
    
    # Block 2: A_R B_21
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A_L B_12
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A_R B_22
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A_L B_15
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(0,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(1,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # Block 6: A_R B_25
    U[u_idx(0,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(0,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(1,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    U[u_idx(1,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
