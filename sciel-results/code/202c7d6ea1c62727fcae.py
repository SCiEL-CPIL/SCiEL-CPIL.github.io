"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to construct a lower-rank decomposition
    # For 2x3x7, attempt rank 35 using constructive method
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify a decomposition
    def verify(U, V, W, rank):
        # Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 iff (i,j,k) is a valid entry
        # Actually for matrix mult: (AB)[i,k] = sum_j A[i,j]*B[j,k]
        # Tensor: T[i*m+j, j*p+k, k*n+i]... let me re-read the convention
        # W[k*n+i, r] - so W is indexed by (k, i)
        # U[i*m+j, r] - U indexed by (i, j)
        # V[j*p+k, r] - V indexed by (j, k)
        # Product: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should give 1 for all i,j,k
        # That's the full tensor - this is the rank decomposition of the matrix multiplication tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    # Attempt rank 35: use the construction where we split into two parts
    # Part 1: 2x1x7 (rank 14) + Part 2: 1x3x7 (rank 21) = 35 total
    # But this needs careful construction
    
    # Actually, let me try a verified rank-35 construction
    # The key idea: for each j, the slice across i and k can be shared
    
    # Try direct construction with rank 35
    rank_try = 35
    U = [[0] * rank_try for _ in range(n * m)]
    V = [[0] * rank_try for _ in range(m * p)]
    W = [[0] * rank_try for _ in range(p * n)]
    
    col = 0
    # For j=0: handle i=0 with all k (7 terms), i=1 with all k (7 terms) = 14 terms
    # For j=1: similar 14 terms
    # For j=2: 7 terms (only one i value effectively used with sharing)
    # Total: 14 + 14 + 7 = 35
    
    # Actually this doesn't work directly. Let me just return schoolbook to be safe.
    # The rank-35 construction requires specific coefficient choices.
    
    return schoolbook()
# EVOLVE-BLOCK-END
