"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose 3x3x8 into two 3x3x4 blocks.
    # Each 3x3x4 block is decomposed into 3x3x3 (rank 9) + 3x3x1 (rank 3) = rank 12.
    # Total rank = 24.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a 3x3x3 schoolbook block for columns start_k to start_k+2
    def add_3x3x3_block(start_k):
        nonlocal col_idx
        for i in range(3):
            for j in range(3):
                U[i * 3 + j][col_idx] = 1
                V[j * 3 + (start_k + 0)][col_idx] = 1 # Wait, B is 3x8. V index is j*p + k.
                # V shape is (m*p, rank) = (3*8, rank).
                # Index is j * p + k.
                V[j * p + (start_k + 0)][col_idx] = 1
                W[(start_k + 0) * n + i][col_idx] = 1
                col_idx += 1
                # Note: The 3x3x3 product computes C[i, start_k..start_k+2]
                # But schoolbook 3x3x3 computes sum_j A[i,j] B[j, k] for a single k?
                # No, schoolbook 3x3x3 computes the full 3x3 matrix product.
                # The tensor for 3x3x3 is sum_{i,j,k} A[i,j] B[j,k] C[i,k].
                # My loop above only adds terms for k=start_k.
                # I need to add terms for k=start_k, start_k+1, start_k+2.
        
    # Let's rewrite the block addition correctly.
    # A 3x3x3 multiplication has 9 terms: sum_{i,j,k} A[i,j] B[j,k] C[i,k].
    # For a block of 3 columns (k0, k1, k2), we have 9 terms.
    
    col_idx = 0
    # Block 1: columns 0, 1, 2
    for k in range(3):
        for i in range(3):
            for j in range(3):
                U[i * 3 + j][col_idx] = 1
                V[j * p + k][col_idx] = 1
                W[k * n + i][col_idx] = 1
                col_idx += 1
                
    # Block 1, col 3: 3x3x1 multiplication (3 terms)
    for i in range(3):
        for j in range(3):
            U[i * 3 + j][col_idx] = 1
            V[j * p + 3][col_idx] = 1
            W[3 * n + i][col_idx] = 1
            col_idx += 1
            
    # Block 2: columns 4, 5, 6
    for k in range(4, 7):
        for i in range(3):
            for j in range(3):
                U[i * 3 + j][col_idx] = 1
                V[j * p + k][col_idx] = 1
                W[k * n + i][col_idx] = 1
                col_idx += 1
                
    # Block 2, col 7: 3x3x1 multiplication (3 terms)
    for i in range(3):
        for j in range(3):
            U[i * 3 + j][col_idx] = 1
            V[j * p + 7][col_idx] = 1
            W[7 * n + i][col_idx] = 1
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
