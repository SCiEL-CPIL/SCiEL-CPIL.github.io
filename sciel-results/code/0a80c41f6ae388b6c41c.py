"""Program for matrix multiplication 2x3x8 with reduced-rank decomposition."""

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (24, rank)
    # W has shape (p*n, rank) = (16, rank)
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term to the decomposition."""
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    # We'll build the decomposition using:
    # 1. Split B into two 3x4 blocks
    # 2. For each 2x3x4 block, use further decomposition
    # 
    # Strategy: Use the known fact that rank(2x3x4) <= 12 (schoolbook)
    # but we can potentially do better.
    #
    # Actually, let's use a direct approach with Strassen-like tricks.
    # 
    # For 2x3x8, we can think of it as 2x3x2 * (4 copies) but that gives 4*12=48.
    # 
    # Better: decompose as (2x3x4) + (2x3x4), and find rank(2x3x4).
    # rank(2x3x4) <= 12 via schoolbook, but we can try to do better.
    #
    # Let's try: A is 2x3, B is 3x4, C is 2x4.
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # We can write this as:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3] + A[0,2]*B[2,3]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3] + A[1,2]*B[2,3]
    
    # Let's use a systematic approach. For 2xmxp, we can achieve rank m*p if we
    # use the standard method, but there might be savings.
    
    # Actually, for 2x3x8, the schoolbook rank is 48.
    # Let me try to find a better decomposition.
    
    # One approach: use the fact that we can combine some terms.
    # 
    # Let's define the 2x3x4 case first, then replicate for 2x3x8.
    
    # For 2x3x4:
    # We'll use 12 terms (schoolbook) but try to reduce.
    
    # Actually, let me just implement the schoolbook for now to get a valid answer,
    # then optimize.
    
    # Schoolbook for 2x3x8: rank = 2*3*8 = 48
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
# EVOLVE-BLOCK-END
