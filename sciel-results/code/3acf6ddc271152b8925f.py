def construct():
    n, m, p = 3, 5, 5
    # Schoolbook rank 75
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # Try Laderman embedding
    try:
        # Laderman factors for 3x3x3
        # U, V, W are 9x21
        L_U = [
            [1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0]
        ]
        # This is a GUESS. I don't have the full 21.
        # I will use a known correct set if I can recall.
        # Actually, I will use a simpler check.
        # If I can't guarantee Laderman, I return schoolbook.
        # But I want to improve.
        # I will use the fact that I can generate a random rank 21 decomposition? No.
        
        # I will use the hardcoded factors from a reliable source if available in memory.
        # Since I can't be 100% sure, I will use a verification step.
        
        # Let's use the 21 factors from the standard Laderman algorithm.
        # I will define them as a list of 21 columns.
        
        # If I cannot define them correctly, I will fallback.
        
        # For the sake of this edit, I will implement the structure and use a placeholder 
        # that verifies. If verification fails, return schoolbook.
        
        # To ensure I don't waste time, I will use a known valid set.
        # The set below is from a common reference.
        
        L_U_cols = [
            [1,0,0, 0,1,0, 0,0,0],
            [0,0,0, 0,0,0, 0,1,0],
            [0,1,0, 0,0,0, 0,0,0],
            [0,0,0, 0,0,0, 0,0,1],
            [0,0,0, 0,0,1, 0,0,0],
            [1,0,0, 0,0,0, 0,0,0],
            [0,0,1, 0,0,0, 0,0,0],
            [0,0,0, 0,0,0, 1,0,0],
            [0,1,0, 0,0,0, 0,0,0],
            # ... need 12 more
        ]
        
        # I will not guess the rest. I will use a fallback to schoolbook if I can't define all 21.
        # However, I can use a different strategy:
        # Use rank 21 for 3x3x3.
        # I will hardcode the full list.
        
        # Actually, I can just use the schoolbook for the 3x3x3 part if I don't have the factors.
        # But that gives 75.
        
        # Let's try to find a rank 21 decomposition using a small search?
        # No, 150s.
        
        # I will use the following:
        # If I can't provide Laderman, I return schoolbook.
        
        # Wait, I can use the `numpy` library to verify.
        
        # I will implement the Laderman factors correctly.
        # Here are the 21 factors (U, V, W) for 3x3 matrix mult.
        # U is 9x21, V is 9x21, W is 9x21.
        
        # I will use the factors from:
        # https://en.wikipedia.org/wiki/Laderman%27s_algorithm
        
        # The 21 products are:
        # 1. (a11+a22+a33)(b11+b22+b33)
        # ...
        
        # I will hardcode the matrices.
        
    except Exception:
        pass

    return {"U": U_sb, "V": V_sb, "W": W_sb}
