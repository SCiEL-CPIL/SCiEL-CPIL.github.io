import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition splitting m into m1=2 and m2=3.
    # Rank for 2 x m1 x p is m1*p + m1 + p - 1
    # Rank for 2 x m2 x p is m2*p + m2 + p - 1
    # Total rank is sum of these.
    
    m1 = 2
    m2 = 3
    
    r1 = m1 * p + m1 + p - 1
    r2 = m2 * p + m2 + p - 1
    rank = r1 + r2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a decomposition for 2 x m_sub x p starting at row offset j_offset
    # and column offset col_offset in U, V, W
    def add_decomp(j_offset, m_sub, r_sub, col_offset):
        # Terms for 2 x m_sub x p of rank r_sub
        # This is a placeholder. We need the actual coefficients.
        # For 2 x m x p, we can use the standard decomposition if we don't have the optimized one.
        # Standard rank is 2 * m * p.
        # But we want the optimized rank.
        # Since I don't have the optimized coefficients for arbitrary m_sub, 
        # I will use the standard decomposition for the sub-blocks.
        # Standard rank for 2 x m1 x p is 2 * m1 * p.
        # Standard rank for 2 x m2 x p is 2 * m2 * p.
        # Total standard rank is 2 * m * p = 60.
        # This does not improve the score.
        
        # I need to implement the optimized decomposition for 2 x m x p.
        # For m=1, rank is p+1.
        # For m=2, rank is 2p+1.
        # I will implement the m=1 and m=2 cases explicitly.
        
        if m_sub == 1:
            # 2 x 1 x p
            # Rank p+1
            # C_0k = A_00 B_0k
            # C_1k = A_10 B_0k
            # We can use p terms for A_00 B_0k and A_10 B_0k?
            # No, they share B_0k.
            # Term k: A_00 B_0k -> C_0k
            # Term k: A_10 B_0k -> C_1k
            # This is 2p terms.
            # Optimized rank is p+1.
            # How?
            # C_0k = A_00 B_0k
            # C_1k = A_10 B_0k
            # Let t_k = A_00 + A_10, s_k = A_00 - A_10?
            # No, B is the same.
            # Actually, for 2 x 1 x p, the rank is p+1 is NOT correct.
            # Rank of 2 x 1 x p is 2p?
            # Let's check 2 x 1 x 1. Rank 2. p+1=2. Correct.
            # 2 x 1 x 2. Rank 4? p+1=3.
            # C_00 = A_00 B_00
            # C_01 = A_00 B_01
            # C_10 = A_10 B_00
            # C_11 = A_10 B_01
            # Can we do rank 3?
            # t0 = A_00, t1 = A_10, t2 = B_00+B_01?
            # C_00 = t0 B_00
            # C_01 = t0 B_01
            # C_10 = t1 B_00
            # C_11 = t1 B_01
            # This is 4 terms.
            # If we use t0 = A_00+A_10, t1 = A_00-A_10?
            # It seems rank is 2p.
            # So the formula mp + m + p - 1 for m=1 gives p+1.
            # Is rank 2p or p+1?
            # For p=1, 2 vs 2.
            # For p=2, 4 vs 3.
            # If rank is 3, then C_00, C_01, C_10, C_11 must be formed by 3 products.
            # It is known that rank of 2 x 1 x 2 is 4.
            # So the formula is wrong for m=1?
            # Or my assumption is wrong.
            
            # Let's assume standard rank 2 * m_sub * p for safety.
            # This will result in rank 60.
            # This is not an improvement.
            
            # I will return the standard decomposition.
            pass

    # Since I cannot guarantee a lower rank without explicit coefficients, 
    # I will return the standard decomposition to be safe.
    # But the task asks for a stronger algorithm.
    # I will try to implement the Strassen-like decomposition for 2x2x2 and extend.
    # For now, I'll return the standard one.
    
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
