"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=6
    # Split A (3x4) into 3 rows (1x4)
    # Split B (4x6) into 3 blocks of (4x2)
    # C (3x6) is formed by C[i, 2j-1:2j] = A[i, :] @ B[:, 2j-1:2j]
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(3): # 3 blocks of width 2
            # Block (i, j) corresponds to A row i, B columns 2j, 2j+1
            # Term 1: k=1 of block
            # U: 1s at i*m + 0..m-1
            for jj in range(m):
                U[i * m + jj][col] = 1
            # V: 1s at jj*p + 2j for jj in 0..m-1
            for jj in range(m):
                V[jj * p + 2 * j][col] = 1
            # W: 1 at i*p + 2j
            W[(2 * j) * n + i][col] = 1 # Wait, W index is k*n + i?
            # Task says W[k*n+i, r].
            # C is n x p. Index k (0..p-1), i (0..n-1).
            # So W index is k*n + i.
            # Here k = 2j.
            W[(2 * j) * n + i][col] = 1
            col += 1
            
            # Term 2: k=2 of block
            # U: 1s at i*m + 0..m-1
            for jj in range(m):
                U[i * m + jj][col] = 1
            # V: 1s at jj*p + 2j+1 for jj in 0..m-1
            for jj in range(m):
                V[jj * p + 2 * j + 1][col] = 1
            # W: 1 at i*p + 2j+1
            W[(2 * j + 1) * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
