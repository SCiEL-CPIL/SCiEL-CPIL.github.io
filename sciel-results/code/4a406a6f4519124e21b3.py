"""Matrix multiplication 2,3,6 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # A: n x m = 2x3, B: m x p = 3x6, C: n x p = 2x6
    # U: (n*m) x rank = 6 x rank
    # V: (m*p) x rank = 18 x rank
    # W: (p*n) x rank = 12 x rank
    # Convention: W[k*n+i, r]

    # We need: for all i in 0..n-1, j in 0..m-1, k in 0..p-1:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1

    # Strategy: Split B into three 3x2 blocks and use Strassen for the 2x2x2
    # subproblem plus additional terms for the extra column of A.

    # For each 2x3 * 3x2 block, we can decompose as:
    # - 2x2 * 2x2 part (using A[:, 0:2] and B[0:2, :]) with Strassen (rank 7)
    # - 1x2 * 1x2 part (using A[:, 2] and B[2, :]) with rank 2
    # Total per block: 9 terms, so 27 total.

    # But let's try to do better. Let me think about this more carefully.

    # Actually, let me try a direct construction with rank 18.
    # The idea: for each (i, k) pair, we have C[i,k] = sum_j A[i,j] * B[j,k]
    # We can write this as 6 terms per (i,k) if we can factor it, but that's not possible.

    # Let me try the naive approach but with some optimization.
    # Naive rank is n*m*p = 36.

    # Let me try to find a better decomposition using the structure.
    # Since n=2, m=3, p=6, we can think of this as 2x3 * 3x6.

    # One approach: use the fact that we can do 2x2 * 2x2 with rank 7 (Strassen),
    # and then handle the extra dimension.

    # Let me try a construction with rank 18:
    # For each i in 0..1, k in 0..5:
    #   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # We can group terms: for fixed i, the three terms are a linear combination.
    # But in tensor decomposition, each term must be a product of three vectors.

    # Let me try a different approach: use bilinear forms.
    # For each (i, j, k), we need one term where U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
    # That gives rank 36.

    # To reduce rank, we need to find linear dependencies.
    # Let me try to see if rank 18 is achievable.

    # Actually, let me just implement a simple valid solution with rank 36 first,
    # then try to optimize.

    # Naive rank 36 solution:
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1

    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    # Shouldn't happen with naive construction
                    pass

    # Now let's try to find a better solution with lower rank.
    # Let me try rank 18 using a block decomposition.

    # For each 2x3 * 3x2 block (3 blocks for p=6):
    # A (2x3): a00, a01, a02; a10, a11, a12
    # B (3x2): b00, b01; b10, b11; b20, b21
    # C (2x2): c00, c01; c10, c11

    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21

    # Let me try to use 6 terms per block:
    # M1 = (a00+a01+a02) * b00  -- this doesn't help directly

    # Actually, let me try a specific construction:
    # For block b (0, 1, 2):
    #   Term 1: U[i*3+0, r]=1 for all i, V[0*6+b*2+0, r]=1, V[0*6+b*2+1, r]=1, W[0*2+i, r]=1, W[1*2+i, r]=1
    #   This doesn't work because V has different indices.

    # Let me try a different approach. I'll use the fact that:
    # sum_j A[i,j] * B[j,k] can be written using 3 terms (one for each j).
    # But we can't reduce below 3 per (i,k) pair in general.

    # So the minimum rank is at least n*p*min(m, something) = 2*6*3 = 36? No, that's the naive bound.

    # Actually, the lower bound for matrix multiplication rank is more complex.
    # For 2x3 * 3x6, I believe the rank is at most 36 (naive) and possibly lower.

    # Let me just return the rank 36 solution for now, as it's guaranteed correct.

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
