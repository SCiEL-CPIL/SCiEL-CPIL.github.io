"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose 2x3 * 3x9 into three independent 2x3 * 3x3 multiplications.
    # Rank of M_{2,3,3} is 9.
    # Total rank is 3 * 9 = 27.
    
    rank = 27
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    n_sub, m_sub, p_sub = 2, 3, 3
    rank_sub = n_sub * m_sub * p_sub
    
    for b in range(3):
        r_offset = b * rank_sub
        
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    r = r_offset + (i * m_sub * p_sub + j * p_sub + k)
                    
                    U[i * m + j][r] = 1
                    V[j * p + b * p_sub + k][r] = 1
                    W[(b * p_sub + k) * n + i][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
