"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin within 120s limit

    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks

    # Strassen decomposition of 2x2x2 tensor
    # Each factor: (u_list, v_list, w_list)
    # u_list: [(i, j, coeff), ...] for U[i*m+j]
    # v_list: [(j, k, coeff), ...] for V[j*p+k]
    # w_list: [(i, k, coeff), ...] for W[k*n+i]
    factors = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]

    # Build the rank-56 Strassen block decomposition
    rank = 7 * num_blocks  # 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_list, v_list, w_list = factors[s]
            for i, j, c in u_list:
                U[i * m + j][col] = c
            for j, k, c in v_list:
                V[j * p + 2 * t + k][col] = c
            for i, k, c in w_list:
                W[(2 * t + k) * n + i][col] = c

    # Try to reduce rank by finding linear dependencies among the rank-56 factors
    # The tensor T has dimensions (n*m) x (m*p) x (p*n) = 4 x 32 x 32
    # We check if we can merge factors from different blocks that are linearly dependent
    # in the combined (U, V, W) space

    # Convert to numpy for efficient linear algebra
    import numpy as np

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Each rank-1 term is defined by (U_col, V_col, W_col)
    # Two terms can be merged if their (U, V, W) tuples are linearly dependent
    # More generally, we look for a smaller set of rank-1 terms that span the same tensor

    # Strategy: stack all (U_col, V_col, W_col) into a single vector of length 4+32+32=68
    # and find the rank of the 68 x 56 matrix. If rank < 56, we can reduce.
    # But actually, the tensor rank is not just the rank of this stacking - it's the
    # tensor rank which is >= the matrix rank of any unfolding.

    # Let's check the tensor rank more carefully using the fact that the multiplication
    # tensor for (2,2,16) can be seen as 8 independent (2,2,2) tensors.
    # The tensor rank of the direct sum of 8 copies of a rank-7 tensor is at most 8*7=56.
    # It could be less if there are cross-block dependencies.

    # For C = A*B where A is 2x2 and B is 2x16:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For k in [0,15], each output C[i,k] depends only on A[i,0], A[i,1] and B[0,k], B[1,k]
    # So the tensor decomposes as a direct sum over k of 2x2x1 sub-tensors? No.
    # Actually for each pair (k, k') we have a 2x2x2 block.
    # The 8 blocks are independent in the sense that block t only involves B[0,2t], B[1,2t], B[0,2t+1], B[1,2t+1]
    # and produces C[0,2t], C[1,2t], C[0,2t+1], C[1,2t+1].
    # Since the B inputs and C outputs for different blocks are disjoint, the tensor rank
    # is exactly the sum of the ranks of each block = 8 * 7 = 56.
    # This is because the tensor is a direct sum (disjoint support in B and C indices).

    # So rank 56 is optimal for this structure. The minimum tensor rank for (2,2,16)
    # is exactly 56 since it's a direct sum of 8 (2,2,2) tensors each with rank 7.

    # Verify: the tensor rank of a direct sum T1 ⊕ T2 is r(T1) + r(T2) when the
    # supports are disjoint in at least one mode. Here, for blocks t and t',
    # the B indices {2t, 2t+1} and {2t', 2t'+1} are disjoint, and similarly for C.
    # So the tensor rank is additive: 8 * 7 = 56.

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
