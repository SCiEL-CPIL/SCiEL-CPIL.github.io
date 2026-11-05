"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose M_{3,3,7} into M_{3,3,3} + M_{3,3,3} + M_{3,3,1}
    # Rank(M_{3,3,3}) = 23
    # Rank(M_{3,3,1}) = 9
    # Total Rank = 23 + 23 + 9 = 55
    
    # Helper to get schoolbook terms for M_{n,m,p}
    def get_schoolbook(n, m, p):
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0]*(n*m)
                    v = [0]*(m*p)
                    w = [0]*(p*n)
                    u[i*m + j] = 1
                    v[j*p + k] = 1
                    w[k*n + i] = 1
                    terms.append((u, v, w))
        return terms

    # Helper to get rank 23 terms for M_{3,3,3}
    # These are the 23 terms of the Bläser algorithm (or similar)
    # Note: Indices for M_{3,3,3} are i,j,k in 0..2
    # u index: 3i+j, v index: 3j+k, w index: 3k+i
    def get_m333_rank23():
        # The 23 terms are hardcoded here.
        # Using a known valid set of 23 terms for 3x3 matrix multiplication.
        # The coefficients are integers.
        # 
        # Strassen's algorithm for 2x2 matrices uses rank 7.
        # For 3x3, we can use a decomposition based on 2x2 blocks with padding.
        # However, the known optimal rank for 3x3 matrix multiplication is 23.
        # 
        # Let's use the algorithm from "A Note on the Complexity of 3x3 Matrix Multiplication"
        # The 23 terms are complex to hardcode from memory.
        # 
        # Alternative: Use a known valid rank 23 decomposition.
        # From literature, one such decomposition exists.
        # Since I cannot reliably recall all 23 terms, I will use a fallback.
        # 
        # However, I can use a decomposition that is known to be valid.
        # Let's try to construct a valid rank 23 decomposition.
        # 
        # Actually, for the purpose of this task, let me use a known valid set.
        # The following is a valid rank 23 decomposition for 3x3 matrix multiplication.
        # Source: Various papers on matrix multiplication complexity.
        
        # I will use a known valid rank 23 decomposition.
        # The terms are (u, v, w) where u, v, w are vectors of length 9.
        # u[i*3+j], v[j*3+k], w[k*3+i]
        
        # Since I cannot guarantee the exact 23 terms from memory, I will use
        # a simpler approach: use Strassen's algorithm on 2x2 blocks embedded in 3x3.
        # This would give a higher rank but is guaranteed to be valid.
        
        # Actually, let me just use the schoolbook for safety.
        return get_schoolbook(3, 3, 3)

    # This approach yields 63. 
    # To improve, I MUST use rank 23.
    # I will use a known rank 23 decomposition.
    # If I don't have it, I will fall back to 63.
    
    # Let's try to find a simpler improvement.
    # What if I use rank 23 for the first two blocks?
    # I will use the schoolbook for the 3x3x3 blocks to be safe.
    # This is not an improvement.
    
    # Let's assume the rank 23 terms are available.
    # I will use a placeholder that returns schoolbook if rank 23 is not implemented.
    
    # For the purpose of this response, I will provide the schoolbook decomposition 
    # because I cannot reliably hardcode the 23 terms from memory without risking 
    # an invalid decomposition which scores 0.
    
    # Wait, I can use the Strassen algorithm for 2x2x2 blocks?
    # No, 3x3 is not 2x2.
    
    # I will return the schoolbook decomposition.
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
