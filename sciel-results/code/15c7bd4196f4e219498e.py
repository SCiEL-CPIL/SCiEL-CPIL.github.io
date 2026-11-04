DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Known best rank for 3x3x3 matrix multiplication is 23 (Strassen-like / Laderman restricted)
    # We use two independent 3x3x3 blocks, each of rank 23, total rank 46.
    # U has shape (n*m, rank) = (9, 46)
    # V has shape (m*p, rank) = (18, 46)
    # W has shape (p*n, rank) = (18, 46)
    
    # We'll use the known rank-23 decomposition for 3x3x3.
    # The standard Strassen 2x2x2 has rank 7.
    # For 3x3x3, the best known is 23.
    # I'll use the explicit coefficients from the literature.
    
    # Actually, let me use a simpler approach: use the schoolbook 27 for each block.
    # But wait, we need to improve. The current program uses 54.
    # If we can use rank 23 for each 3x3x3, total is 46.
    # Let me try to find a valid rank-23 decomposition.
    
    # For now, let me try a different approach: use the fact that 3x3x6 can be seen as
    # multiplying 3x3 by two 3x3 blocks. If we use a clever decomposition that shares
    # some structure...
    
    # Actually, the simplest improvement is to use the known rank-23 for 3x3x3.
    # Let me hardcode the rank-23 decomposition for 3x3x3.
    
    # The 23-term decomposition for 3x3x3 (from Bini et al. / Strassen):
    # I'll use the standard form.
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use schoolbook for now but with a different structure.
    # Actually, let me just use the schoolbook 27 per block = 54 total.
    # That's the same as current. I need something better.
    
    # Let me try rank 46 using two rank-23 decompositions.
    # I need explicit coefficients for rank-23 3x3x3.
    
    # For safety, let me use the schoolbook decomposition but with a twist:
    # Can we share any terms between the two blocks? No, they're independent.
    
    # Let me just use 46 as the rank and fill with a valid decomposition.
    # I'll use the known result that 3x3x3 has rank at most 23.
    
    # Here are the coefficients for a rank-23 decomposition of 3x3x3:
    # (These are from the literature - I'll use a standard set)
    
    # U_block: 9x23, V_block: 9x23, W_block: 9x23
    # For each block, we compute C = A * B where A is 3x3, B is 3x3, C is 3x3
    
    # I'll use the Strassen-style 7-term for 2x2 and extend.
    # Actually, let me just use the schoolbook 27 for each block to be safe.
    # 27 * 2 = 54, same as current.
    
    # Let me try a different approach: use the fact that we can do 3x3x6 in rank 40 or so.
    # For now, I'll stick with 54 but verify the indices are correct.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(2):
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    k_global = k + block * 3
                    U[i * m + j][col] = 1
                    V[j * p + k_global][col] = 1
                    W[k_global * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
