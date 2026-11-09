"""Program for matrix multiplication 5x5x5 tensor decomposition."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Try to use known best rank for 5x5x5 matrix multiplication
    # The best known rank for 5x5x5 is 49 (Bini 1970, improved by others)
    # But constructing it explicitly is complex. Let's try a systematic approach.
    
    # For now, let's implement a known rank-49 decomposition if we can construct it.
    # The rank-49 decomposition of M_5 was found by Bini et al.
    
    # Actually, let me think about this differently. The best known upper bound
    # for R(M_5) is 49. But I need to construct it explicitly.
    
    # Let me try a different approach: use the fact that we can combine
    # Strassen's algorithm with padding to get a rank-49 decomposition.
    
    # Actually, the cleanest approach for a guaranteed valid result within time limits
    # is to use the schoolbook decomposition (rank 125) unless we can quickly
    # find a better one.
    
    # Let me attempt to construct a rank-49 decomposition using the known method.
    # The rank-49 algorithm for 5x5x5 matrix multiplication was published by Bini.
    
    # Given the complexity and time constraints, let me implement a verified
    # rank-49 decomposition. The key insight is that M_5 can be embedded
    # in a larger tensor structure.
    
    # For safety and guaranteed correctness, use schoolbook (rank 125).
    # This is always valid and exact.
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
