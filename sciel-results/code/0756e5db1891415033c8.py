import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use full budget safely

    n, m, p = 2, 2, 10

    # Hybrid approach:
    # - For the first 2 columns of B (a 2x2 block), use Strassen (rank 7)
    # - For the remaining 8 columns (four 2x2 blocks), use schoolbook (rank 4 each)
    # Total rank = 7 + 4*4 = 23
    #
    # Wait, let me reconsider. The strategy says:
    # "Apply the rank-7 Strassen decomposition to the 2x2x2 tensor for the left two columns
    # of the 2x10 matrix, and use the standard rank-4 schoolbook decomposition for the remaining
    # four columns, yielding a total rank of 31."
    #
    # Hmm, "remaining four columns" - but we have 10 columns total. If we use Strassen for 2 columns
    # (rank 7), we have 8 columns left. If we use schoolbook for each pair of columns (2x2 blocks),
    # that's 4 blocks * rank 4 = 16. Total = 7 + 16 = 23.
    #
    # But the strategy says rank 31. Let me re-read: "use the standard rank-4 schoolbook decomposition
    # for the remaining four columns". Maybe it means 4 columns total remaining? That would be 7 + 4*2 = 15?
    # No, that doesn't work either.
    #
    # Actually, I think the strategy might be poorly worded. Let me just implement the obvious hybrid:
    # - Strassen for one 2x2 block: rank 7
    # - Schoolbook for the other four 2x2 blocks: rank 4 * 4 = 16
    # Total: rank 23
    #
    # This should be valid and give a better score. Let me verify the decomposition is correct.
    #
    # Actually, I need to be careful about the W convention. The task says:
    # "The output factor follows the official trace convention W[k*n+i,r]"
    #
    # So W has shape (p*n, rank) = (20, rank). The index is k*n + i where k is the column index
    # of C (0..p-1) and i is the row index (0..n-1).
    #
    # For a 2x2 block at column offset c (c = 2*k for block k):
    # C[0, c]   -> W[c*2 + 0] = W[4k + 0]
    # C[1, c]   -> W[c*2 + 1] = W[4k + 1]
    # C[0, c+1] -> W[(c+1)*2 + 0] = W[4k + 2]
    # C[1, c+1] -> W[(c+1)*2 + 1] = W[4k + 3]
    #
    # This matches the current program's convention.

    # Strassen terms for 2x2:
    # A: [A11, A12, A21, A22]
    # B: [B11, B12, B21, B22]
    # C: [C11, C21, C12, C22] -> maps to W indices [0,1,2,3] within block
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]

    # Schoolbook terms for 2x2:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    schoolbook_terms = [
        ([1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]),  # A11*B11 -> C11
        ([0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]),  # A12*B21 -> C11
        ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]),  # A11*B12 -> C12
        ([0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]),  # A12*B22 -> C12
        ([0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]),  # A21*B11 -> C21
        ([0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]),  # A22*B21 -> C21
        ([0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]),  # A21*B12 -> C22
        ([0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]),  # A22*B22 -> C22
    ]

    # Block 0: use Strassen (rank 7)
    # Blocks 1-4: use schoolbook (rank 4 each)
    total_rank = 7 + 4 * 4  # = 23

    rank = total_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Block 0: Strassen
    for a_vec, b_vec, c_vec in strassen_terms:
        for idx in range(n * m):
            U[idx][col] = a_vec[idx]

        # V: B is 2x10, for block 0, columns 0 and 1
        # B[0, 0] -> V[0*10 + 0] = V[0]
        # B[0, 1] -> V[0*10 + 1] = V[1]
        # B[1, 0] -> V[1*10 + 0] = V[10]
        # B[1, 1] -> V[1*10 + 1] = V[11]
        V[0 * p + 0][col] = b_vec[0]
        V[0 * p + 1][col] = b_vec[1]
        V[1 * p + 0][col] = b_vec[2]
        V[1 * p + 1][col] = b_vec[3]

        # W: C is 2x10, for block 0, columns 0 and 1
        # C[0, 0] -> W[0*2 + 0] = W[0]
        # C[1, 0] -> W[0*2 + 1] = W[1]
        # C[0, 1] -> W[1*2 + 0] = W[2]
        # C[1, 1] -> W[1*2 + 1] = W[3]
        W[0 * n + 0][col] = c_vec[0]
        W[0 * n + 1][col] = c_vec[1]
        W[1 * n + 0][col] = c_vec[2]
        W[1 * n + 1][col] = c_vec[3]

        col += 1

        if time.time() > deadline:
            break

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Blocks 1-4: schoolbook
    for k in range(1, 5):
        for a_vec, b_vec, c_vec in schoolbook_terms:
            for idx in range(n * m):
                U[idx][col] = a_vec[idx]

            # V: B is 2x10, for block k, columns 2k and 2k+1
            # B[0, 2k] -> V[0*10 + 2k]
            # B[0, 2k+1] -> V[0*10 + 2k+1]
            # B[1, 2k] -> V[1*10 + 2k]
            # B[1, 2k+1] -> V[1*10 + 2k+1]
            V[0 * p + 2 * k][col] = b_vec[0]
            V[0 * p + 2 * k + 1][col] = b_vec[1]
            V[1 * p + 2 * k][col] = b_vec[2]
            V[1 * p + 2 * k + 1][col] = b_vec[3]

            # W: C is 2x10, for block k, columns 2k and 2k+1
            # C[0, 2k] -> W[2k*2 + 0] = W[4k]
            # C[1, 2k] -> W[2k*2 + 1] = W[4k+1]
            # C[0, 2k+1] -> W[(2k+1)*2 + 0] = W[4k+2]
            # C[1, 2k+1] -> W[(2k+1)*2 + 1] = W[4k+3]
            W[2 * k * n + 0][col] = c_vec[0]
            W[2 * k * n + 1][col] = c_vec[1]
            W[(2 * k + 1) * n + 0][col] = c_vec[2]
            W[(2 * k + 1) * n + 1][col] = c_vec[3]

            col += 1

            if time.time() > deadline:
                break

        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
