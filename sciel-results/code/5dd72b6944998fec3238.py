import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Split 2x3x6 into two 2x3x3 blocks.
    # Use rank-14 decomposition for each 2x3x3 block.
    # Total rank = 28.
    
    # Rank-14 decomposition for 2x3x3 matrix multiplication.
    # For A (2x3) * B (3x3) = C (2x3):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Indices: i in {0,1}, j in {0,1,2}, k in {0,1,2}
    #
    # The 14 rank-1 terms are defined by (u_index, v_index, w_index)
    # where u_index = i*3+j (for A), v_index = j*3+k (for B), w_index = k*2+i (for C)
    # Wait, let me use the convention from the task:
    # U: (n*m, rank) = (6, rank) - index by i*m+j
    # V: (m*p, rank) = (9, rank) - index by j*p+k
    # W: (p*n, rank) = (6, rank) - index by k*n+i
    
    # Define the 14 terms for 2x3x3.
    # Each term is a triple (u_idx, v_idx, w_idx, coeff)
    # u_idx = i*3 + j (i=0,1; j=0,1,2)
    # v_idx = j*3 + k (j=0,1,2; k=0,1,2)
    # w_idx = k*2 + i (k=0,1,2; i=0,1)
    
    # Known rank-14 decomposition for 2x3x3:
    # Terms 1-7 handle the 2x2x2 subproblem (columns 0,1 of B)
    # Terms 8-14 handle the remaining column 2 and cross terms
    
    terms_233 = [
        # (i, j, k, coeff) for each rank-1 term
        # Term 1: A[0,0]*B[0,0] -> C[0,0]
        (0, 0, 0, 1),
        # Term 2: A[0,1]*B[1,0] -> C[0,0]
        (0, 1, 0, 1),
        # Term 3: A[1,0]*B[0,1] -> C[1,1]
        (1, 0, 1, 1),
        # Term 4: A[1,1]*B[1,1] -> C[1,1]
        (1, 1, 1, 1),
        # Term 5: A[0,0]*B[1,0] -> C[0,0]
        (0, 0, 0, 1),
        # Term 6: A[0,1]*B[0,0] -> C[0,0]
        (0, 1, 0, 1),
        # Term 7: A[1,0]*B[1,1] -> C[1,1]
        (1, 0, 1, 1),
        # Terms 8-14 for column 2
        # Term 8: A[0,2]*B[2,0] -> C[0,0]
        (0, 2, 0, 1),
        # Term 9: A[0,2]*B[2,1] -> C[0,1]
        (0, 2, 1, 1),
        # Term 10: A[0,2]*B[2,2] -> C[0,2]
        (0, 2, 2, 1),
        # Term 11: A[1,2]*B[2,0] -> C[1,0]
        (1, 2, 0, 1),
        # Term 12: A[1,2]*B[2,1] -> C[1,1]
        (1, 2, 1, 1),
        # Term 13: A[1,2]*B[2,2] -> C[1,2]
        (1, 2, 2, 1),
        # Term 14: Cross term to fix dependencies
        (0, 0, 2, -1),
    ]
    
    # This construction may not be correct. Let me use a verified approach.
    # Actually, let me just use schoolbook for safety, but with a note that we aim for 28.
    
    # For now, use schoolbook rank 36 to ensure correctness.
    # TODO: Replace with rank-28 decomposition.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
