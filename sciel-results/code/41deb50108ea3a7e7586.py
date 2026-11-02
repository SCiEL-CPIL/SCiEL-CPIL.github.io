"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition based on splitting the output columns into two blocks of size p/2
    # and splitting the input rows into n blocks of size 1.
    # This results in a rank of n * (p / 2) * m? 
    # Actually, the logic derived was:
    # Split B into p/2 blocks of width 2? No, width 5.
    # Split A into n rows.
    # For each row of A and each block of B, we have a multiplication of 1xm by mx(p/2).
    # Rank of 1xm by mxq is m.
    # Total rank = n * (p / (p/2)) * m = n * 2 * m?
    # Wait, p=10. Split into 2 blocks of 5.
    # Number of blocks = 2.
    # For each block, we process n rows.
    # For each row, rank is m.
    # Total rank = 2 (blocks) * n (rows) * m (middle dim) = 2 * 2 * 3 = 12.
    
    rank = 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # k_block: 0 for left half (cols 0-4), 1 for right half (cols 5-9)
    for k_block in range(2):
        start_k = k_block * (p // 2)
        end_k = start_k + (p // 2)
        for i in range(n):
            for j in range(m):
                # U: index i*m + j
                U[i * m + j][col] = 1
                
                # V: index j*p + k for k in [start_k, end_k)
                for k in range(start_k, end_k):
                    V[j * p + k][col] = 1
                    
                # W: index k*n + i for k in [start_k, end_k)
                for k in range(start_k, end_k):
                    W[k * n + i][col] = 1
                    
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
