"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split m=3 into m1=2, m2=1
    # Split p=10 into 5 blocks of p_t=2
    # Rank = 5 * (2*2*2) + 1 = 41
    
    rank = 41
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 5 blocks of 2x2 multiplication
    # A1 is 2x2 (rows 0,1; cols 0,1)
    # B1 is 2x10 (rows 0,1; cols 0..9)
    # Split B1 into 5 blocks of 2x2
    for t in range(5):
        # Block t uses k in {2t, 2t+1}
        # Rank 8 for this block
        base_col = t * 8
        for i in range(2):
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * t + k_local
                    col = base_col + (i * 4 + j * 2 + k_local) # 0..7
                    
                    U[i * m + j][col] = 1
                    V[j * p + k_global][col] = 1
                    W[k_global * n + i][col] = 1
                    
    # 1 block for A2 B2
    # A2 is 2x1 (rows 0,1; col 2)
    # B2 is 1x10 (row 2; cols 0..9)
    # Rank 1
    col = 40
    # U[i*m+2, col] = 1
    # V[2*p+k, col] = 1
    # W[k*n+i, col] = 1
    # This works if we sum over l.
    # For a fixed i, k:
    # Sum_l U[i,2,l] V[2,k,l] W[k,i,l]
    # Only l=40 is non-zero.
    # U[i,2,40]=1, V[2,k,40]=1, W[k,i,40]=1.
    # Sum = 1.
    for i in range(2):
        U[i * m + 2][col] = 1
    for k in range(10):
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1 # Wait, W index is k*n+i.
        # We need W[k*n+i, col] = 1 for all i?
        # If W[k*n+i, col] = 1 for all i, then for a specific i, it is 1.
        # But W is a matrix. W[k*n+i, col] is a single entry.
        # We need to set it for each i.
    for i in range(2):
        for k in range(10):
            W[k * n + i][col] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
