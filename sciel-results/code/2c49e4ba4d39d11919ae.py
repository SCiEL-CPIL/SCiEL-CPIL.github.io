"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 2 x 3 x 9 matrix multiplication
    # We can decompose this into 3 blocks of 2 x 3 x 3
    # Each 2 x 3 x 3 block can be done in rank 16 (known result)
    # Total rank = 3 * 16 = 48
    
    # Actually, let's use a more careful construction.
    # For 2 x 3 x 3, the rank is at most 16 (Bini et al.)
    # We'll construct a rank-16 decomposition for each 2x3x3 block.
    
    # For 2 x 3 x 3 matrix multiplication:
    # A is 2x3, B is 3x3, C is 2x3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Known rank-16 decomposition for 2x3x3:
    # We can use the following approach:
    # The tensor has 18 entries. We need to find a rank-16 decomposition.
    # One way: use the standard decomposition and merge two terms.
    
    # Let me use a known rank-16 scheme for 2x3x3.
    # Actually, let me just use the trivial rank-18 for each block to be safe,
    # but try to reduce.
    
    # For now, let me construct a valid decomposition.
    # 2 x 3 x 9: we can split B into three 3x3 blocks.
    # C[:, 0:3] = A * B[:, 0:3]  (2x3 * 3x3 = 2x3)
    # C[:, 3:6] = A * B[:, 3:6]
    # C[:, 6:9] = A * B[:, 6:9]
    
    # Each sub-problem is 2x3x3.
    # If we can do 2x3x3 in rank 16, total is 48.
    # If we can do it in rank 15, total is 45.
    # The trivial is rank 18 per block, total 54.
    
    # Let me construct a rank-16 decomposition for 2x3x3.
    # Using the method of Bini, Capovani, Lotti, Romani (1980):
    # For 2x2x2, rank is 7 (Strassen).
    # For 2x3x3, we can extend this.
    
    # Actually, let me use a simpler approach:
    # The 2x3x3 tensor can be seen as computing 2x3 = A(2x3) * B(3x3)
    # We can write this as:
    # C = A * B
    # where A has 6 entries, B has 9 entries, C has 6 entries.
    
    # Trivial: 18 multiplications.
    # We can try to save 2 by using Strassen-like tricks on a 2x2x2 sub-block.
    
    # Let me define the rank-16 decomposition explicitly.
    # I'll use the following known construction:
    
    # For 2x3x3, one rank-16 decomposition uses the following 16 terms.
    # Let me index A[i,j] for i in {0,1}, j in {0,1,2}
    # B[j,k] for j in {0,1,2}, k in {0,1,2}
    # C[i,k] for i in {0,1}, k in {0,1,2}
    
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # We can group:
    # Let P0 = A[0,0]*B[0,0]
    # P1 = A[0,1]*B[1,0]
    # P2 = A[0,2]*B[2,0]
    # P3 = A[1,0]*B[0,1]
    # P4 = A[1,1]*B[1,1]
    # P5 = A[1,2]*B[2,1]
    # ... this is getting complicated.
    
    # Let me just use the trivial decomposition for now with rank 54,
    # but with a slightly different indexing to make it distinct.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
