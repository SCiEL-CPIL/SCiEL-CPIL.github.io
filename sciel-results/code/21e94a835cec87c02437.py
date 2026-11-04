"""Matrix multiplication 4,4,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8

    # Split B (4x8) into two 4x4 blocks: B_left = B[:, :4], B_right = B[:, 4:8]
    # C = A * B = [A * B_left | A * B_right]
    # Each 4x4*4x4 multiplication can be done with rank 49 (Bürgisser et al. 2005)
    # But implementing that is very complex. Let's use a simpler approach.
    
    # For 4x4*4x4, we can use Strassen recursively:
    # 4x4 = 2x2 blocks of 2x2 matrices
    # Strassen for 2x2*2x2 has rank 7
    # So 4x4*4x4 via Strassen has rank 7*4 = 28? No, that's not right either.
    
    # Actually, Strassen for 4x4*4x4:
    # Split into 2x2 blocks of 2x2 matrices
    # Apply Strassen: 7 recursive multiplications of 2x2 matrices
    # Each 2x2*2x2 is schoolbook rank 4
    # Total rank = 7 * 4 = 28
    
    # But we can do better. Let's just use schoolbook for now but with the block structure.
    # Actually, let me think about this differently.
    
    # The key insight: for 4x4*4x8, we can share the computation of A*B_left and A*B_right
    # if we structure the decomposition properly.
    
    # Let's use the following approach:
    # For each output C[i,k] = sum_j A[i,j] * B[j,k]
    # We can group by i and share the A[i,j] factors.
    
    # Actually, the simplest correct approach with some sharing:
    # Use the fact that we can compute partial sums.
    
    # Let me just implement a correct decomposition with rank < 128.
    # One simple approach: for each i, compute the row A[i,:] * B (4x8)
    # This is a 1x4 * 4x8 multiplication, which has rank 4 (schoolbook)
    # Total: 4 * 4 = 16? No, that's not right.
    
    # Actually, C[i,k] = sum_j A[i,j] * B[j,k]
    # For fixed i, this is a dot product of A[i,:] and column k of B
    # The tensor for fixed i has rank 4 (the 4 terms in the sum)
    # Total rank = 4 * 4 = 16? No, that's still not right.
    
    # Let me just do the straightforward thing: rank = 128 with schoolbook
    # but let me try to find a better decomposition.
    
    # Actually, I recall that for m*n*p matrix multiplication, the rank is at most
    # min(m*n*p, ...) and there are known bounds.
    
    # For 4x4*4x8:
    # - Schoolbook: 128
    # - Strassen-style: we can potentially do better
    
    # Let me implement a simple block decomposition:
    # Split A into 2x2 blocks of 2x2 matrices: A = [[A00, A01], [A10, A11]]
    # Split B into 2x4 blocks of 2x4 matrices: B = [[B00, B01, B02, B03], [B10, B11, B12, B13]]
    # Wait, B is 4x8, so split into 2x2 blocks of 2x4 matrices:
    # B = [[B00, B01], [B10, B11]] where each block is 2x4
    
    # C = A*B = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
    #            [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    
    # Each 2x2 * 2x4 multiplication has rank 8 (schoolbook)
    # Number of such multiplications: 8 (4 in each row, 2 rows)
    # Total rank: 8 * 8 = 64
    
    # That's already better than 128!
    
    # Let me implement this:
    rank = 64
    U = [[0] * rank for _ in range(n * m)]  # 16 x 64
    V = [[0] * rank for _ in range(m * p)]  # 32 x 64
    W = [[0] * rank for _ in range(p * n)]  # 32 x 64
    
    col = 0
    # A is 4x4, indexed as A[i,j] where i in 0..3, j in 0..3
    # B is 4x8, indexed as B[j,k] where j in 0..3, k in 0..7
    # C is 4x8, indexed as C[i,k] where i in 0..3, k in 0..7
    
    # Block structure:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B00 = B[0:2, 0:4], B01 = B[0:2, 4:8]
    # B10 = B[2:4, 0:4], B11 = B[2:4, 4:8]
    
    # C00 = A00*B00 + A01*B10  (2x4)
    # C01 = A00*B01 + A01*B11  (2x4)
    # C10 = A10*B00 + A11*B10  (2x4)
    # C11 = A10*B01 + A11*B11  (2x4)
    
    # For each 2x2 * 2x4 multiplication, we have 8 terms (schoolbook)
    
    # A00*B00: A[i,j] * B[j,k] for i in 0,1; j in 0,1; k in 0..3
    # A01*B10: A[i,j] * B[j,k] for i in 0,1; j in 2,3; k in 0..3
    # etc.
    
    # Let me enumerate all 64 terms:
    # Group 1: A00*B00 (8 terms)
    for i in range(2):
        for j in range(2):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Group 2: A01*B10 (8 terms)
    for i in range(2):
        for j in range(2, 4):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Group 3: A00*B01 (8 terms)
    for i in range(2):
        for j in range(2):
            for k in range(4, 8):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Group 4: A01*B11 (8 terms)
    for i in range(2):
        for j in range(2, 4):
            for k in range(4, 8):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Group 5: A10*B00 (8 terms)
    for i in range(2, 4):
        for j in range(2):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Group 6: A11*B10 (8 terms)
    for i in range(2, 4):
        for j in range(2, 4):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Group 7: A10*B01 (8 terms)
    for i in range(2, 4):
        for j in range(2):
            for k in range(4, 8):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Group 8: A11*B11 (8 terms)
    for i in range(2, 4):
        for j in range(2, 4):
            for k in range(4, 8):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
