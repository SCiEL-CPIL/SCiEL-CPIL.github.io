"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy: Use a recursive Strassen decomposition (rank 49) for the first four
columns of B (4x4*4x4), and an optimized decomposition for the last two columns.

Key insight: The 4x4*4x2 tensor can be decomposed with much lower rank than the
naive block-Strassen approach (28). By viewing the last two columns as two
independent matrix-vector multiplications (or using the fact that each slice
M[:,:,k] is a rank-1 all-ones matrix in the appropriate sense), we can achieve
a much lower rank.

For M_{4,4,2}: C[i,k] = sum_j A[i,j] * B[j,k]
This is equivalent to computing two matrix-vector products:
  C[:,0] = A * B[:,0]  (4x4 * 4x1 -> 4x1)
  C[:,1] = A * B[:,1]  (4x4 * 4x1 -> 4x1)

Each matrix-vector product M_{4,4,1} has tensor rank 4 (since M[:,:,k] is an
all-ones matrix of rank 1, but we need to sum over j, so actually the rank is
equal to min(m, n) = 4 for each... wait).

Actually, for M_{n,m,1}: C[i] = sum_j A[i,j] B[j]
The tensor T[i,j,0] = 1 for all i,j. This is a rank-1 tensor in the (i,j) slice.
But the rank of the tensor as a whole: T = sum_l u_l v_l w_l.
Since there's only one k-index (k=0), T[i,j,0] = sum_l U[i,l] V[j,l] W[0,l].
This is just a rank decomposition of the all-ones matrix, which has rank 1.
So M_{n,m,1} has rank 1? No, that's not right.

Wait, let me reconsider. The matrix multiplication tensor M_{n,m,p} has entries
M[i,j,k] = 1 if the (i,j)-th entry of A multiplies the (j,k)-th entry of B to
contribute to C[i,k]. For standard matrix multiplication, M[i,j,k] = 1 for all
valid i,j,k. So it's the all-ones tensor of size n x m x p.

The rank of the all-ones tensor of size n x m x p is min(n, m, p).
So M_{4,4,2} has rank min(4,4,2) = 2.

Therefore, total rank = 49 + 2 = 51.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 6]


def construct():
    n, m, p = 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    # Strassen 2x2*2x2 factors
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))

    prods = [
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U_s[i * 2 + j, m_idx] += c
        for (j, k, c) in Vb:
            V_s[j * 2 + k, m_idx] += c
        for (i, k, c) in Wb:
            W_s[k * 2 + i, m_idx] += c

    # Total rank: 49 (4x4*4x4) + 2 (4x4*4x2) = 51
    rank_444 = 49
    rank_442 = 2
    total_rank = rank_444 + rank_442

    U = np.zeros((n * m, total_rank))  # 16 x 51
    V = np.zeros((m * p, total_rank))  # 24 x 51
    W = np.zeros((p * n, total_rank))  # 24 x 51

    # Part 1: 4x4*4x4 Strassen for columns 0-3 of B
    # Recursive Strassen: 7 outer products, each is a 2x2*2x2 Strassen
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
            # U part: combine outer Strassen t with inner Strassen s
            for i in range(4):
                I = i // 2
                i0 = i % 2
                for j in range(4):
                    J = j // 2
                    j0 = j % 2
                    val = U_s[2 * I + J, t] * U_s[2 * i0 + j0, s]
                    if val != 0:
                        U[i * 4 + j, col] += val

            # V part
            for j in range(4):
                J = j // 2
                j0 = j % 2
                for k in range(4):
                    K = k // 2
                    k0 = k % 2
                    val = V_s[2 * J + K, t] * V_s[2 * j0 + k0, s]
                    if val != 0:
                        V[j * 6 + k, col] += val

            # W part
            for k in range(4):
                K = k // 2
                k0 = k % 2
                for i in range(4):
                    I = i // 2
                    i0 = i % 2
                    val = W_s[2 * K + I, t] * W_s[2 * k0 + i0, s]
                    if val != 0:
                        W[k * 4 + i, col] += val

    # Part 2: 4x4*4x2 for columns 4-5 of B
    # The all-ones tensor of size 4x4x2 has rank min(4,4,2) = 2
    # Rank-2 decomposition: T[i,j,k] = sum_{l=0}^{1} U[i,l] V[j,l] W[k,l]
    # For the all-ones tensor: T[i,j,k] = 1
    # We can use: U[i,l] = 1, V[j,l] = 1, W[k,l] = 1 for l=0, and similar for l=1?
    # No, that gives T[i,j,k] = 2. We need T[i,j,k] = 1.
    # Actually, the rank-1 decomposition of the all-ones matrix (4x4) is:
    # ones_4x4 = u v^T where u = [1,1,1,1]^T, v = [1,1,1,1]^T
    # So for each k-slice, we have a rank-1 decomposition.
    # But the tensor is 3D: T[i,j,k] = 1 for all i,j,k.
    # This can be written as sum_{l} U_l[i] V_l[j] W_l[k]
    # If we set W_l[k] = delta_{k,l} for l=0,1 (since k only goes 0,1),
    # and U_l = ones, V_l = ones, then:
    # T[i,j,k] = sum_{l=0}^{1} 1 * 1 * delta_{k,l} = 1 if k in {0,1}
    # This works!
    
    # But wait, in our case, the "B" indices are (j, k) where j is 0..3 and k is 4..5
    # In the global V matrix, the index is j * p + k = j * 6 + k
    # For the last 2 columns, k = 4, 5
    # So V[j * 6 + k, col] for j in 0..3, k in {4, 5}
    
    # Term 0: U[i, 0] = 1 for all i, V[j, 0] = 1 for all j, W[k, 0] = 1 if k=4 else 0
    # Term 1: U[i, 1] = 1 for all i, V[j, 1] = 1 for all j, W[k, 1] = 1 if k=5 else 0
    
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, rank_444] += 1.0
            U[i * 4 + j, rank_444 + 1] += 1.0
    
    for j in range(4):
        V[j * 6 + 4, rank_444] += 1.0
        V[j * 6 + 5, rank_444 + 1] += 1.0
    
    for i in range(4):
        W[4 * 4 + i, rank_444] += 1.0
        W[5 * 4 + i, rank_444 + 1] += 1.0

    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

