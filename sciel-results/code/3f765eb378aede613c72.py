"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # A (3x4) = [A_L (3x2) | A_R (3x2)]
    # B (4x6) = [B_T (2x6); B_B (2x6)]
    # C = A_L B_T + A_R B_B
    
    # We decompose each 3x2x6 multiplication into rank 27 (3 * rank 9 for 3x2x2)
    # Total rank = 2 * 27 = 54
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Helper to add a 3x2x2 Strassen-like decomposition (Rank 9)
    # X is 3x2, Y is 2x2.
    # X indices: i in 0..2, k in 0..1. U row: i*2 + k
    # Y indices: k in 0..1, j in 0..1. V row: k*2 + j
    # Z indices: i in 0..2, j in 0..1. W row: j*3 + i (Note: W row is k*n+i in general, here output is 3x2, so j is col, i is row. W row = j*n + i? 
    # Task W row: k*n + i where k is col, i is row. Yes.)
    
    def add_3x2x2(base_u, base_v, base_w, offset_col):
        # Strassen for 2x2 part (rows 0,1 of X)
        # P1 = (X00 + X11) * (Y00 + Y11)
        # P2 = (X10 + X11) * Y00
        # P3 = X00 * (Y10 + Y11)
        # P4 = X11 * Y11
        # P5 = (X00 + X01) * Y10
        # P6 = (X10 - X00) * (Y00 + Y01)
        # P7 = (X01 + X11) * (Y11 - Y10)
        
        # Mapping to our indices
        # X00 -> u_idx 0, X01 -> u_idx 1
        # X10 -> u_idx 2, X11 -> u_idx 3
        # Y00 -> v_idx 0, Y01 -> v_idx 1
        # Y10 -> v_idx 2, Y11 -> v_idx 3
        
        # We need to add 7 terms for the 2x2 part and 2 terms for the 3rd row.
        
        # Term 1: P1
        add_term(base_u + 0, base_v + 0, base_w + 0, 1, 1, 1) # Z00 += P1
        add_term(base_u + 0, base_v + 1, base_w + 1, 1, 1, 1) # Z01 += P1
        add_term(base_u + 1, base_v + 0, base_w + 0, 1, 1, 1) # Z00 += P1 (Wait, U/V are linear combos)
        
        # This manual construction is error-prone. 
        # Instead, I will use the known rank 7 Strassen coefficients for 2x2 and rank 2 for 1x2.
        
        # Let's use a simpler, verified Rank 9 construction for 3x2x2.
        # Rank 7 for top 2 rows (2x2x2).
        # Rank 2 for bottom row (1x2x2).
        
        # Strassen 2x2:
        # C00 = P1 + P4 - P5 - P7
        # C01 = P3 + P7
        # C10 = P2 + P5
        # C11 = P1 - P2 + P3 + P4
        
        # Indices for U (3x2 input):
        # u00=0, u01=1, u10=2, u11=3, u20=4, u21=5
        # Indices for V (2x2 input):
        # v00=0, v01=1, v10=2, v11=3
        # Indices for W (3x2 output):
        # w00=0, w10=1, w01=2, w11=3, w20=4, w21=5 (Row i, Col j -> j*3+i)
        
        # P1 = (u00+u11)(v00+v11)
        add_term(base_u+0, base_v+0, base_w+0, 1, 1, 1)
        add_term(base_u+0, base_v+3, base_w+0, 1, 1, 1)
        add_term(base_u+3, base_v+0, base_w+0, 1, 1, 1)
        add_term(base_u+3, base_v+3, base_w+0, 1, 1, 1)
        # Wait, Strassen terms are products of linear combos.
        # The decomposition adds to the rank space.
        # Each "term" in Strassen is a product of two linear combos.
        # In our tensor decomposition, each rank-1 term is a product of one linear combo from U, one from V, and a coefficient from W.
        # So for P1, we need to represent (u00+u11)(v00+v11).
        # This requires 4 rank-1 terms if we expand? No, Strassen uses 7 rank-1 terms for the whole 2x2 multiplication.
        
        # The 7 terms are:
        # T1: (u00+u11) * (v00+v11) -> contributes to C00, C11
        # T2: (u10+u11) * v00 -> contributes to C10, C11
        # T3: u00 * (v10+v11) -> contributes to C00, C01
        # T4: u11 * v11 -> contributes to C00, C11
        # T5: (u00+u01) * v10 -> contributes to C00, C10
        # T6: (u10-u00) * (v00+v01) -> contributes to C10, C01
        # T7: (u01+u11) * (v11-v10) -> contributes to C01, C11
        
        # We add these 7 terms to the rank space.
        
        # T1
        add_term(base_u+0, base_v+0, base_w+0, 1, 1, 1) # C00
        add_term(base_u+0, base_v+3, base_w+0, 1, 1, 1) # C00 (v11)
        add_term(base_u+3, base_v+0, base_w+0, 1, 1, 1) # C00 (u11)
        add_term(base_u+3, base_v+3, base_w+0, 1, 1, 1) # C00 (u11 v11)
        # This is getting complicated to map manually.
        
        # I will fall back to the schoolbook rank 72 if I cannot guarantee correctness of the complex Strassen mapping in this short time.
        # However, Rank 54 is a significant improvement.
        
        pass

    # Given the complexity and risk of error in manual Strassen mapping for rectangular blocks, 
    # and the 120s limit, I will return the schoolbook rank 72 to ensure validity.
    # A safer improvement is Rank 48?
    # 3x4x6.
    # If we use Strassen on 4x4x4 (Rank 49) and pad?
    # 3x4x6 fits in 4x4x8?
    
    # Let's just return the seed program to be safe.
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
# EVOLVE-BLOCK-END
