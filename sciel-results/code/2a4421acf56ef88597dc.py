"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # BCDL rank-19 decomposition for 3x3x3 tensor
    # The 19 rank-1 terms with their (i, j, k) indices and coefficients
    # BCDL decomposition: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # for the 3x3x3 multiplication tensor where T[i,j,k]=1
    
    # BCDL 19-term decomposition coefficients
    # Format: (i, j, k, coeff) for each of the 19 terms
    bcdl_terms = [
        # The BCDL decomposition uses 19 terms
        # Let me use the known BCDL decomposition
        # Terms are indexed 0..18
        # U[i][c], V[j][c], W[k][c]
    ]
    
    # Actually, let me construct the BCDL decomposition properly.
    # The BCDL 3x3x3 tensor rank-19 decomposition:
    # We need U (3x19), V (3x19), W (3x19) such that
    # sum_c U[i,c]*V[j,c]*W[k,c] = 1 for all i,j,k in {0,1,2}
    
    # Known BCDL decomposition (one standard form):
    # I'll use the decomposition from the literature.
    # The 19 terms with entries in {0, 1/2, 1, 3/2, 2, 3}
    
    # Let me use a verified BCDL decomposition:
    bcdl_U = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    # This is getting complicated. Let me use a different approach.
    # I'll directly encode a known valid BCDL decomposition.
    
    # BCDL 3x3x3 rank-19 decomposition (verified form):
    # U is 3x19, V is 3x19, W is 3x19
    bcdl_U = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=object)
    
    # This is the schoolbook for 3x3x3 (rank 9), not BCDL.
    # Let me just use the schoolbook approach but optimized.
    
    # Actually, for 3x4x4, let me try a block approach:
    # Split the 4 columns of A into 3+1, and 4 columns of B into 3+1.
    # 
    # The multiplication C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # For j=0,1,2: this is a 3x3x3 tensor (i in 0..2, j in 0..2, k in 0..2) plus
    #              cross terms where k=3
    # For j=3: this is a 3x1x4 tensor
    #
    # Better decomposition:
    # Term 1: 3x3x3 for (i,j,k) with j in {0,1,2}, k in {0,1,2}: BCDL rank 19
    # Term 2: 3x3x1 for (i,j,k=3) with j in {0,1,2}: schoolbook rank 9
    # Term 3: 3x1x4 for (i,j=3,k) with j=3: schoolbook rank 12
    # Total: 19 + 9 + 12 = 40
    #
    # Even better:
    # Term 1: 3x3x3 for j in {0,1,2}, k in {0,1,2}: BCDL rank 19
    # Term 2: 3x3x1 for j in {0,1,2}, k=3: rank 9 (schoolbook)
    # Term 3: 3x1x4 for j=3, k in {0,1,2,3}: rank 12 (schoolbook)
    # Total: 40
    #
    # Or:
    # Term 1: 3x3x4 for j in {0,1,2}, k in {0,1,2,3}: rank 12 (schoolbook)
    # Term 2: 3x1x4 for j=3, k in {0,1,2,3}: rank 12 (schoolbook)
    # Total: 24 (better than 48!)
    #
    # But can we do better? What if we use BCDL for part of it?
    # 3x3x4 tensor: split k into 3+1
    #   3x3x3: BCDL rank 19
    #   3x3x1: rank 9
    #   Total for 3x3x4: 28 (worse than 12)
    #
    # So the best simple block decomposition is 24.
    #
    # Can we do even better? What about:
    # Use BCDL for 3x3x3 and handle the rest cleverly?
    #
    # Actually, let me reconsider. The 3x4x4 tensor has 48 entries.
    # The minimum rank is at least max(3*4, 4*4, 3*4) = 16.
    #
    # Let me try a more sophisticated approach.
    # 
    # Strategy: Use the fact that 3x3x3 has rank 19 (BCDL).
    # Decompose 3x4x4 as:
    # - 3x3x3 sub-tensor (i,j,k all in 0..2): BCDL, rank 19
    # - Remaining: entries where j=3 or k=3 (but not both, to avoid double counting)
    #   - j=3, k in 0..2: 3x1x3 tensor, schoolbook rank 9
    #   - j in 0..2, k=3: 3x3x1 tensor, schoolbook rank 9
    #   - j=3, k=3: 3x1x1 tensor, rank 3
    # Total: 19 + 9 + 9 + 3 = 40
    #
    # Hmm, that's worse. Let me stick with 24.
    
    # Actually, wait. Let me reconsider the problem structure.
    # A is 3x4, B is 4x4. C = A*B is 3x4.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # The tensor T[i,j,k] = 1 for all i in 0..2, j in 0..3, k in 0..3.
    # This is a 3x4x4 tensor.
    #
    # Rank of 3x4x4 tensor: at least max(12, 16, 12) = 16.
    #
    # Simple block decomposition:
    # Split j into {0,1,2} and {3}:
    # - j in {0,1,2}: 3x3x4 tensor, rank 12 (schoolbook)
    # - j in {3}: 3x1x4 tensor, rank 12 (schoolbook)
    # Total: 24
    #
    # Can we do better by using BCDL?
    # 3x3x4 tensor: split k into {0,1,2} and {3}:
    # - k in {0,1,2}: 3x3x3, BCDL rank 19
    # - k in {3}: 3x3x1, rank 9
    # Total for 3x3x4: 28 (worse)
    #
    # So 24 seems to be the best with simple blocks.
    #
    # Let me try another approach: what if we use a non-trivial decomposition
    # that mixes the blocks?
    
    # For now, let me implement the rank-24 decomposition.
    rank = 24
    U = [[0] * rank for _ in range(n * m)]  # 3*4 = 12 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*4 = 16 rows
    W = [[0] * rank for _ in range(p * n)]  # 4*3 = 12 rows
    
    col = 0
    # Block 1: j in {0,1,2}, k in {0,1,2,3}: 3*3*4 = 12 terms
    for i in range(n):  # i = 0,1,2
        for j in range(3):  # j = 0,1,2
            for k in range(p):  # k = 0,1,2,3
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block 2: j=3, k in {0,1,2,3}: 3*1*4 = 12 terms
    for i in range(n):  # i = 0,1,2
        for k in range(p):  # k = 0,1,2,3
            U[i * m + 3][col] = 1
            V[3 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
