import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _verify_full(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[l * p + k][r] * W[k * n + i][r]
                    expected = 1 if j == l else 0
                    if total != expected:
                        return False
    return True

def _schoolbook_2x3x6():
    n, m, p = DIMENSIONS
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

def construct():
    deadline = time.time() + 110.0
    # Attempt to find a better decomposition if time permits,
    # but for now, ensure the valid schoolbook decomposition is returned.
    # The previous rank-3 attempt was invalid for general multiplication 
    # despite passing the specific delta check due to index coupling issues 
    # in the tensor definition interpretation. We stick to the known correct 
    # schoolbook rank 36 to guarantee a valid score.
    
    result = _schoolbook_2x3x6()
    
    # Optional: Verify before returning (cost is low for rank 36)
    if _verify_full(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: If verification fails (should not happen), return empty or raise
    # But schoolbook is correct, so this is just a safety net.
    return result

# EVOLVE-BLOCK-END
