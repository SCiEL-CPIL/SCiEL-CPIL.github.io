"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # Build a rank-18 decomposition by splitting the 9 output columns into
    # three independent 3-column blocks. Each block performs a 2x3 * 3x3
    # multiplication. The 2x3x3 matrix-multiplication tensor has rank 6,
    # so three such blocks give a total rank of 18.
    
    num_blocks = p // m  # 3 blocks
    rank_per_block = 6   # rank of 2x3x3 multiplication tensor
    rank = num_blocks * rank_per_block  # 18
    
    # Precompute the 6 rank-1 tensors for 2x3 * 3x3 -> 2x3
    # Each rank-1 term: U_term (length 6), V_term (length 9), W_term (length 6)
    # U index: i*m + j for i in 0..1, j in 0..2
    # V index: j*p + k for j in 0..2, k in 0..2
    # W index: k*n + i for k in 0..2, i in 0..1
    
    # We use the standard rank-6 decomposition of the 2x3x3 MM tensor.
    # The 2x3x3 tensor T[i,j,k] = delta_{j,k} (i.e., C[i,k] = sum_j A[i,j] B[j,k])
    # A rank-6 decomposition can be constructed as follows:
    # For each of the 3 output columns k, we need to compute C[0,k] and C[1,k].
    # 
    # Actually, let's use a known rank-6 decomposition. The 2x3x3 MM tensor
    # can be decomposed into 6 rank-1 terms. One such decomposition:
    # 
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    # Term 1: A[0,1]*B[1,0] -> C[0,0]
    # Term 2: A[0,2]*B[2,0] -> C[0,0]
    # Term 3: A[1,0]*B[0,1] -> C[1,1]
    # Term 4: A[1,1]*B[1,1] -> C[1,1]
    # Term 5: A[1,2]*B[2,1] -> C[1,1]
    # 
    # Wait, that's not right. We need C[0,k] = sum_j A[0,j] B[j,k] for all k.
    # 
    # Let me think more carefully. The 2x3x3 MM tensor has 2*3*3 = 18 entries.
    # A rank-1 term is a tensor T[i,j,k] = u_i * v_j * w_k.
    # 
    # For the MM tensor, T[i,j,k] = 1 if j==k, else 0 (for the identity-like structure).
    # Actually no: T[i,j,k] = delta_{jk} means the tensor is zero unless j=k,
    # and when j=k, T[i,j,j] = 1 for all i.
    # 
    # So the tensor has nonzero entries only at positions (i,j,j) for i in {0,1}, j in {0,1,2}.
    # That's 2*3 = 6 nonzero entries.
    # 
    # A rank-1 decomposition: each term picks one (i,j,j) position and sets
    # u_i=1, v_j=1, w_j=1, with all other entries 0.
    # This gives 6 rank-1 terms, each with exactly one nonzero entry.
    # 
    # So the rank-6 decomposition is:
    # For each i in {0,1}, j in {0,1,2}:
    #   U[i*m+j] = 1, V[j*p+j] = 1, W[j*n+i] = 1
    
    terms = []
    for i in range(n):
        for j in range(m):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            u[i * m + j] = 1
            v[j * p + j] = 1
            w[j * n + i] = 1
            terms.append((u, v, w))
    
    # Now assemble the full rank-18 decomposition
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        for idx, (u, v, w) in enumerate(terms):
            col = t * rank_per_block + idx
            for row in range(n * m):
                if u[row] != 0:
                    U[row][col] = u[row]
            for row in range(m * p):
                # Shift V indices by t*m for the block
                shifted_row = row + t * m
                if shifted_row < m * p and v[row] != 0:
                    V[shifted_row][col] = v[row]
            for row in range(p * n):
                # Shift W indices by t*m for the block
                shifted_row = row + t * m * n
                if shifted_row < p * n and w[row] != 0:
                    W[shifted_row][col] = w[row]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
