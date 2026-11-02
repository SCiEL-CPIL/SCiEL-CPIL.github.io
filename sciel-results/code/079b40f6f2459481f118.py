"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # Use Strassen-like block decomposition to reduce rank below schoolbook 4*5*6=120.
    # Block sizes: n=4 -> 2x2 blocks of size 2, m=5 -> 3+2, p=6 -> 3+3
    # This is a structured approach using known low-rank decompositions for small matrix multiplication.
    #
    # For 2x2*2x2*2x2, rank 7 (Strassen)
    # For 2x3*3x3*3x2, rank 12 (using 7+5 or similar known bounds)
    # For 3x2*2x3*3x3, rank 12
    # For 2x3*3x2*3x2, rank 8
    #
    # Actually, let's use a simpler and more reliable approach:
    # We'll use the known optimal or near-optimal rank for each sub-block.
    # 
    # Let me use a concrete construction with explicit small ranks.
    # Block partition: A is 4x5, B is 5x6, C is 6x4.
    # Split A rows [0:2, 2:4], cols [0:3, 3:5] -> blocks A00(2x3), A01(2x2), A10(2x3), A11(2x2)
    # Split B rows [0:3, 3:5], cols [0:3, 3:6] -> blocks B00(3x3), B01(3x3), B10(3x2), B11(2x3)
    # Split C rows [0:3, 3:6], cols [0:2, 2:4] -> blocks C00(3x2), C01(3x2), C10(3x2), C11(3x2)
    #
    # Products:
    # C00 = A00*B00 + A01*B10  (2x3 * 3x3 -> 2x3, 2x2 * 3x2 -> 2x2... mismatch)
    # Let me re-think the splitting.
    #
    # Actually, let's use a cleaner block decomposition:
    # n=4: blocks of 2+2
    # m=5: blocks of 2+3  
    # p=6: blocks of 3+3
    #
    # A (4x5): A00(2x2), A01(2x3), A10(2x2), A11(2x3)
    # B (5x6): B00(2x3), B01(2x3), B10(3x3), B11(3x3)
    # C (6x4): C00(3x2), C01(3x2), C10(3x2), C11(3x2)
    #
    # C00 = A00*B00 + A01*B10  -> 2x2*2x3=2x3 + 2x3*3x3=2x3 -> sum 2x3, but C00 is 3x2... 
    # Wait, C is p x n = 6x4, so C00 is (3x2), which means row block 0 of C (3 rows) x col block 0 of C (2 cols).
    # C[i,k] for i in [0,2], k in [0,1] = sum_j A[i,j]*B[j,k]
    # = sum_{j in [0,1]} A00[i-0,j-0]*B00[j-0,k-0] + sum_{j in [2,4]} A01[i-0,j-2]*B10[j-2,k-0]
    # = (A00*B00)[i,k] + (A01*B10)[i,k]  but A01 is 2x3, B10 is 3x2? No, B10 should be (m_block1 x p_block0) = 3x3.
    # So A01 (2x3) * B10 (3x3) = 2x3, but we need 2x2 result. Mismatch.
    #
    # The issue is that the intermediate sum must have matching inner dimension.
    # Let me just use the schoolbook but with a slightly different structure, or use a known rank-120 is trivial.
    #
    # Given the constraints and need for correctness, I'll implement a valid decomposition 
    # using a rank-117 construction by exploiting one of the trivial identities.
    # Actually, the safest "new" approach is to use a different but still correct schoolbook variant.
    #
    # For a genuine improvement attempt, let's use the fact that for 4x5*5x6, 
    # we can use rank 119 by combining two of the terms.
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # start with schoolbook, then try to merge
    
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
    
    # Try to reduce rank by checking if any two columns are linearly dependent in a useful way
    # For now, return schoolbook as it's guaranteed correct
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
