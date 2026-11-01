"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    rank = n * m * p  # 52

    U = [[0] * rank for _ in range(n * m)]  # 4 x 52
    V = [[0] * rank for _ in range(m * p)]  # 26 x 52
    W = [[0] * rank for _ in range(p * n)]  # 26 x 52

    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # Term: A[i, k] * B[k, j] contributes to C[i, j]
                
                # U index: i*m + k
                U[i * m + k][col] += 1
                
                # V index: k*p + j
                V[k * p + j][col] += 1
                
                # W index: j*n + i
                # Note: W[k*n+i, r] in prompt? 
                # Prompt says: "output factor follows the official trace convention W[k*n+i,r]"
                # Wait, the prompt says: "W[k*n+i,r]". This is unusual.
                # Standard is W[j*n + i, r] or similar.
                # Let's re-read carefully: "W[k*n+i,r]"
                # Usually W is of size (p*n, rank).
                # If W index is k*n+i, then k must be the column index of C (j) and i is row index?
                # Let's assume the prompt's "k" in "W[k*n+i]" corresponds to the column index of the result matrix C, which is usually denoted j.
                # And "i" corresponds to the row index of C.
                # So W[j*n + i, r].
                
                W[j * n + i][col] += 1
                
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
