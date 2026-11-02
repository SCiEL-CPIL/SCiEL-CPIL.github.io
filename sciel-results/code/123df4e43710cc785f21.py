"""Matrix multiplication 4,4,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-32 exact tensor decomposition for 4x4 * 4x8 -> 4x8.
    
    Strategy: Use the structure of rectangular matrix multiplication.
    A (4x4) * B (4x8) = C (4x8).
    
    We can exploit that this is equivalent to multiplying two 4x4 blocks 
    where the right matrix has two such blocks side-by-side.
    
    Actually, let's think more carefully. We need rank r such that:
    C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..3, j in 0..3, k in 0..7.
    
    Schoolbook gives rank 4*4*8 = 128.
    
    Let's use a block-based approach. We can view:
    A is 4x4, B is 4x8 (two 4x4 blocks: B1, B2).
    C is 4x8 (two 4x4 blocks: C1, C2).
    
    C1 = A * B1, C2 = A * B2.
    
    If we had a rank-r decomposition for 4x4 * 4x4, we could apply it twice,
    giving rank 2r for the 4x4 * 4x8 case. But that's not necessarily optimal
    because the same A is shared.
    
    Actually, since A is shared, we can do better. Let's think of it as:
    C = A * [B1 | B2]
    
    Using bilinear decomposition: C[i,k] = sum_c U[i,k,c] * V_A[i,j,c] * V_B[j,k,c]
    
    Wait, the standard tensor decomposition for matrix mult (n,m,p) has:
    U of shape (n*m, r), V of shape (m*p, r), W of shape (p*n, r)
    where C = sum_c W_c ⊗ (U_c^T * V_c) ... 
    
    Let me re-read the interface:
    U: (n*m, rank) -> indices (i,j)
    V: (m*p, rank) -> indices (j,k)  
    W: (p*n, rank) -> indices (k,i)
    
    The reconstruction is: C[i,k] = sum_{j,c} A[i,j] * B[j,k] * (coefficient)
    
    Actually, the standard Strassen-style decomposition works as:
    For each term c, we have:
    - U picks out a bilinear combination of A entries
    - V picks out a bilinear combination of B entries
    - W places the product into the output
    
    More precisely, if we define for each c:
    - alpha_c: a vector of length n*m (coefficients for A)
    - beta_c: a vector of length m*p (coefficients for B)
    - gamma_c: a vector of length p*n (coefficients for C)
    
    Then: sum_c (alpha_c · A) * (beta_c · B) = gamma_c · C (flattened)
    
    Or equivalently: C = sum_c gamma_c ⊗ (alpha_c^T A beta_c) ... 
    
    Let me just use the known small-rank decompositions.
    
    For 4x4x4, the best known rank is 21 (Lader). But let me use a simpler approach.
    
    Actually, let me use the fact that 4x4x4 can be done in rank 21, but that's complex.
    
    Simpler: Use the 2x2x2 Strassen (rank 7) to build up.
    
    4x4x4: We can partition into 2x2 blocks. That gives 4 sub-problems of 2x2x2.
    Using Strassen for each: 4 * 7 = 28? No, that's not right either.
    
    Let me think differently. For 4x4 * 4x8:
    
    Partition A into four 2x2 blocks: A = [[A11, A12], [A21, A22]]
    Partition B into blocks: B is 4x8, so B = [[B11, B12], [B21, B22]] where each is 2x4
    C = [[C11, C12], [C21, C22]] where each is 2x4
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    This is getting complex. Let me just implement a known rank-32 decomposition.
    
    Actually, the simplest improvement: note that 4*4*8 = 128 for schoolbook.
    
    Let me try a direct construction with rank 32.
    
    Key insight: C[i,k] = sum_j A[i,j] B[j,k].
    
    We can group the k dimension. For each pair of i values, we can share some computations.
    
    Let me try: for each i, C[i,:] = A[i,:] * B. This is a row of A times B.
    A[i,:] is a 1x4 vector, B is 4x8. The product is 1x8.
    
    We can compute A[i,:] * B using rank-4 (just dot products), but we share across i.
    
    Alternative: Use the identity that we can compute all 4 rows of A times B with shared sub-products.
    
    Let me just go with a careful rank-32 construction based on the following:
    
    For the 4x4x4 case, there's a known rank-21 algorithm, but let me use a simpler rank-24 or so.
    
    Actually, let me just use the straightforward block approach:
    
    Split the 8 columns of B into two groups of 4: B = [B_left | B_right], each 4x4.
    C = [C_left | C_right], each 4x4.
    C_left = A * B_left
    C_right = A * B_right
    
    If I use a rank-r decomposition for 4x4x4, I can apply it to (A, B_left) and (A, B_right).
    But A is shared, so I can share the U factors.
    
    If the 4x4x4 decomposition has U (16 x r), V (16 x r), W (16 x r), then:
    For C_left = A * B_left: use U_A (for A), V_left (for B_left), W (for C_left)
    For C_right = A * B_right: use U_A (same!), V_right (for B_right), W (for C_right)
    
    Total rank: r (for A part) + r (for B_left) + r (for B_right) ... no that's not how it works.
    
    In the tensor decomposition, each term c has:
    - A-coefficients (16 values)
    - B-coefficients (16 values for 4x4, or 32 for 4x8)
    - C-coefficients (16 values for 4x4, or 32 for 4x8)
    
    For the combined problem, if I use rank r for each 4x4x4 sub-problem:
    - Terms 1..r: handle C_left = A * B_left
      U[:, 1..r] = U_44 (A coefficients)
      V[:, 1..r] = V_44_left (B_left coefficients, padded to 32)
      W[:, 1..r] = W_44_left (C_left coefficients, padded to 32)
    - Terms r+1..2r: handle C_right = A * B_right
      U[:, r+1..2r] = U_44 (A coefficients, same!)
      V[:, r+1..2r] = V_44_right (B_right coefficients, padded to 32)
      W[:, r+1..2r] = W_44_right (C_right coefficients, padded to 32)
    
    Total rank = 2r. If r=16 (schoolbook for 4x4x4), total = 32.
    
    So rank 32 is achievable by just doing two 4x4x4 schoolbook decompositions!
    That's a 4x improvement over 128.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Schoolbook for 4x4x4 has rank 16
    # We do two such decompositions, one for each half of the 8 columns
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 32
    V = [[0] * rank for _ in range(m * p)]  # 32 x 32
    W = [[0] * rank for _ in range(p * n)]  # 32 x 32
    
    # For the left half (k=0..3):
    # U[i*4+j, c] = 1 if c corresponds to (i,j)
    # V[j*8+k, c] = 1 if k < 4 and c corresponds to (j,k)
    # W[k*4+i, c] = 1 if k < 4 and c corresponds to (k,i)
    
    # For the right half (k=4..7):
    # U[i*4+j, c] = 1 if c corresponds to (i,j) [same A indices]
    # V[j*8+k, c] = 1 if k >= 4 and c corresponds to (j, k-4)
    # W[k*4+i, c] = 1 if k >= 4 and c corresponds to (k,i)
    
    col = 0
    # Left half: k = 0,1,2,3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # A coefficient
                U[i * 4 + j][col] = 1
                # B coefficient: B is 4x8, so index is j*8 + k
                V[j * 8 + k][col] = 1
                # C coefficient: C is 4x8, so index is k*4 + i
                W[k * 4 + i][col] = 1
                col += 1
    
    # Right half: k = 4,5,6,7
    for i in range(4):
        for j in range(4):
            for k in range(4, 8):
                # A coefficient (same as left)
                U[i * 4 + j][col] = 1
                # B coefficient: B is 4x8, so index is j*8 + k
                V[j * 8 + k][col] = 1
                # C coefficient: C is 4x8, so index is k*4 + i
                W[k * 4 + i][col] = 1
                col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
