import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 7
    # Target Rank: 20
    # Strategy:
    # Split B (4x7) into B_L (4x4) and B_R (4x3)
    # C = A B_L + A B_R
    #
    # A B_L (3x4 by 4x4 -> 3x4)
    # Split A (3x4) into A_L (3x3) and A_R (3x1)
    # B_L (4x4) into B_LL (3x4) and B_LR (1x4)
    # A B_L = A_L B_LL + A_R B_LR
    # A_L B_LL (3x3 by 3x4 -> 3x4). Split 4 into 2+2.
    #   A_L B_LL1 (3x3x2) Rank 6
    #   A_L B_LL2 (3x3x2) Rank 6
    #   Total Rank 12
    # A_R B_LR (3x1 by 1x4 -> 3x4) Rank 1
    # Total A B_L Rank = 13
    #
    # A B_R (3x4 by 4x3 -> 3x3)
    # A_L B_RL (3x3 by 3x3 -> 3x3) Rank 7
    # A_R B_RR (3x1 by 1x3 -> 3x3) Rank 1
    # Total A B_R Rank = 8
    #
    # Total Rank = 21.
    
    # Wait, I calculated 13 + 8 = 21.
    # Let's refine A_L B_LL (3x3x4).
    # Rank of 3x3x4 is 9?
    # If I use Rank 9 for 3x3x4, total is 9 + 1 (for A_R B_LR) + 7 (for 3x3x3) + 1 (for 3x1x3) = 18.
    
    # I will implement Rank 21 using simple blocks to ensure validity.
    # 12 (3x3x4 via 2x 3x3x2) + 1 (3x1x4) + 7 (3x3x3) + 1 (3x1x3) = 21.
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(len(u_vec)):
            if u_vec[i] != 0:
                U[i][col] = u_vec[i]
        for j in range(len(v_vec)):
            if v_vec[j] != 0:
                V[j][col] = v_vec[j]
        for k in range(len(w_vec)):
            if w_vec[k] != 0:
                W[k][col] = w_vec[k]
        col += 1

    # --- Block 1: A_L B_LL1 (3x3x2) ---
    # A_L is rows 0-2 of A (cols 0-2 of A).
    # B_LL1 is rows 0-2 of B_L (cols 0-1 of B_L).
    # This is 3x3x2. Split 3x3 into 2x2 + rest?
    # Use 3x1x2 + 1x3x2?
    # Simpler: Schoolbook 3x3x2 = 18? No, we want low rank.
    # Use 2x2x2 Strassen (Rank 7) for top-left 2x2.
    # Plus terms for row 3 and col 3.
    # This is getting complicated.
    
    # Let's just use Rank 24 with 4=2+2, 7=4+3, and schoolbook for 3x2x4 and 3x2x3?
    # 3x2x4 Rank 24? No.
    
    # I will use the Rank 16 construction I identified earlier.
    # It requires the 7 terms for 3x3x3.
    # I will hardcode the 7 terms for 3x3x3.
    
    # 3x3x3 Decomposition (Rank 7)
    # Source: Generalized Strassen / Known minimal rank
    # Terms (u, v, w) where u is 3x3 vec (flattened), v is 3x3 vec, w is 3x3 vec (flattened k*n+i)
    # Actually, let's just use Rank 24 to be safe and fast.
    
    # RESTART: Rank 24 Construction
    # Split A (3x4) into A1 (3x2), A2 (3x2)
    # Split B (4x7) into B1 (2x7), B2 (2x7)
    # C = A1 B1 + A2 B2
    #
    # A1 B1 (3x2x7). Split 7 into 4+3.
    # A1 B1_1 (3x2x4). Rank 8? (4x4x4 is 16? No).
    # 3x2x4: Split 3 into 2+1.
    #   2x2x4 (Rank 14? 2x2x2 is 7, so 2x2x4 is 14).
    #   1x2x4 (Rank 1? 1x2 by 2x4. U(1x2), V(2x4), W(4x1). Rank 1).
    #   Total 3x2x4 Rank 15.
    # A1 B1_2 (3x2x3).
    #   2x2x3 (Rank 7? 2x2x2 is 7, 2x2x1 is 2. Total 9?).
    #   1x2x3 (Rank 1).
    #   Total 3x2x3 Rank 10.
    # Total A1 B1 Rank 25.
    # Total C Rank 50.
    
    # This is too high. The seed is 84.
    # I need a better decomposition.
    
    # Let's go back to Rank 16.
    # I will implement the 7 terms for 3x3x3 manually.
    # The 7 terms are:
    # 1. (e1+e2+e3) (f1+f2+f3) (g1+g2+g3) - NO.
    
    # I will use a valid Rank 20 decomposition.
    # 4=3+1.
    # C = A_L B_L + A_R B_R.
    # A_R B_R (3x1x7) Rank 1.
    # A_L B_L (3x3x7). Split 7=4+3.
    # A_L B_L2 (3x3x3) Rank 7.
    # A_L B_L1 (3x3x4). Rank 9.
    # Total 1 + 7 + 9 = 17.
    
    # I will implement Rank 17.
    # I need the 9 terms for 3x3x4.
    # 3x3x4: Split 4=2+2.
    # 3x3x2 Rank 6. (3x3x1 is 3, so 3x3x2 is 6).
    # 2 * 6 = 12.
    # Total 1 + 7 + 12 = 20.
    
    # I will implement Rank 20.
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add(u, v, w):
        nonlocal col
        for i in range(len(u)):
            if u[i]: U[i][col] = u[i]
        for j in range(len(v)):
            if v[j]: V[j][col] = v[j]
        for k in range(len(w)):
            if w[k]: W[k][col] = w[k]
        col += 1
        
    # Helper for 3x3x2 (Rank 6)
    # A (3x3), B (3x2).
    # Split A into A1 (2x3), A2 (1x3).
    # A1 B (2x3x2) Rank 4? (2x2x2 is 7? No, 2x3x2).
    # Just use schoolbook for 3x3x2? Rank 18. Too high.
    
    # I will use Schoolbook for the sub-blocks but combined with Strassen where possible.
    # Actually, just returning Rank 24 is safe.
    # 84 -> 24 is a good improvement.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Split A into A1 (3x2), A2 (3x2)
    # Split B into B1 (2x7), B2 (2x7)
    # C = A1 B1 + A2 B2
    
    # A1 B1 (3x2x7). Split 7 into 4+3.
    # A1 B1_1 (3x2x4).
    # A1 B1_2 (3x2x3).
    
    # I'll just use Schoolbook for 3x2x4 (Rank 24) and 3x2x3 (Rank 18)?
    # No, 3x2x4 is small.
    # 3x2x4: 12 terms.
    # 3x2x3: 9 terms.
    # Total 21.
    # Plus A2 B2 (21).
    # Total 42.
    
    # I'll return Rank 42.
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    for i in range(3):
        for j in range(2):
            for k in range(7):
                U[i*4 + j][col] = 1
                V[j*7 + k][col] = 1
                W[k*3 + i][col] = 1
                col += 1
    for i in range(3):
        for j in range(2):
            for k in range(7):
                U[i*4 + (j+2)][col] = 1
                V[(j+2)*7 + k][col] = 1
                W[k*3 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
