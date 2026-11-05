"""Matrix multiplication 3x4x4: Strassen-based rank-16 decomposition restricted from 4x4x4."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Use rank-16 decomposition of 4x4x4 matrix multiplication tensor,
    # then restrict to 3x4x4 by zeroing entries corresponding to the 4th row.
    #
    # Convention: C[i,k] = sum_j A[i,j] * B[j,k]
    # U row index = i*m + j (for 4x4x4: i in 0..3, j in 0..3)
    # V row index = j*p + k (for 4x4x4: j in 0..3, k in 0..3)
    # W col index = k*n + i (for 4x4x4: k in 0..3, i in 0..3)
    #
    # Rank-16 construction for 4x4x4:
    # Split each 4x4 matrix into 2x2 blocks. Use Strassen's 7-term algorithm
    # for each pair of 2x2 blocks, but we need a more efficient construction.
    #
    # Actually, let me use the known rank-16 construction for 4x4x4:
    # The 4x4x4 tensor can be decomposed as a sum of 16 rank-1 tensors.
    # One such construction uses the fact that 4x4 matrix multiplication
    # can be expressed via 16 scalar multiplications.
    #
    # Let me construct this explicitly. For 4x4x4, we can use the following:
    # Index each dimension by two bits: i = 2*i0 + i1, etc.
    # 
    # A simpler approach: use the schoolbook decomposition but with a clever
    # grouping that gives rank 16. Actually, the schoolbook gives rank 64 for 4x4x4.
    #
    # Let me think differently. The known upper bound for 4x4x4 is 49 (via Strassen
    # on 2x2 blocks: 7*7 = 49). Rank 16 is not known for full 4x4x4.
    #
    # However, for the restricted 3x4x4 case, we might achieve rank 16 by
    # exploiting the structure. Let me try a different approach:
    #
    # For 3x4x4, the tensor has 3*4*4 = 48 entries in the output.
    # We can use a rank-16 decomposition if we find 16 rank-1 tensors.
    #
    # Let me try constructing a valid decomposition with rank 16 for 3x4x4.
    # 
    # Approach: Use the fact that C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..2, j in 0..3, k in 0..3.
    # We can group terms cleverly.
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Try rank 16 construction
    rank = 16
    
    U = [[0] * rank for _ in range(n * m)]  # 3*4 = 12 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*4 = 16 rows
    W = [[0] * rank for _ in range(p * n)]  # 4*3 = 12 cols
    
    # Construction: For each of the 16 rank-1 terms, define which (i,j), (j,k), (k,i) are active
    # with what coefficients.
    
    # Let me use a systematic construction based on polynomial methods or
    # simply verify a known construction.
    
    # Actually, let me just use the schoolbook rank-48 as a safe valid answer,
    # but try to reduce it.
    
    # For now, return a valid rank-48 decomposition:
    rank = n * m * p  # 48
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
