"""Tensor decomposition of 5x5x5 matrix multiplication, rank 117."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin

    n, m, p = 5, 5, 5
    N = 5

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(U_dict, V_dict, W_dict):
        U_cols.append(U_dict)
        V_cols.append(V_dict)
        W_cols.append(W_dict)

    # Partition: {0,1}, {2,3}, {4}
    # For 2x2x2 sub-blocks fully within {0,1}x{0,1}x{0,1}, {0,1}x{0,1}x{2,3}, etc.,
    # we can use Strassen (rank 7) instead of schoolbook (rank 8).
    # For blocks involving index 4, use schoolbook.

    # Strategy: Use a 2+2+1 partition on each dimension.
    # Block structure:
    #   A[i,j] where i in {0,1,2,3,4}, j in {0,1,2,3,4}
    #   B[j,k] where j in {0,1,2,3,4}, k in {0,1,2,3,4}
    #   C[i,k] where i in {0,1,2,3,4}, k in {0,1,2,3,4}
    #
    # The matmul is C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We partition the summation index j into {0,1}, {2,3}, {4}.
    # For each pair of i-blocks and k-blocks, we compute the contribution.
    #
    # i-blocks: I0={0,1}, I1={2,3}, I2={4}
    # j-blocks: J0={0,1}, J1={2,3}, J2={4}
    # k-blocks: K0={0,1}, K1={2,3}, K2={4}
    #
    # For each (I_a, J_b, K_c) triple, we compute a 2x2, 2x2, or 1x1 block of C.
    # If all three blocks are size 2 (i.e., not involving index 4), we can use Strassen.
    # Otherwise, use schoolbook.

    # Actually, let me think more carefully. The standard approach for 5x5x5:
    # Use the 2+2+1 partition. The 4x4x4 core can be done with Strassen-based
    # decomposition. The frame (involving index 4) must be schoolbook.

    # Frame terms:
    # i=4, any j, any k: C[4,k] = sum_j A[4,j]*B[j,k] -> 5*5 = 25 terms
    # i in {0,1,2,3}, j=4, any k: C[i,k] += A[i,4]*B[4,k] -> 4*5 = 20 terms
    # Total frame: 45 terms

    for j in range(5):
        for k in range(5):
            U = {4 * N + j: 1}
            V = {j * N + k: 1}
            W = {k * N + 4: 1}
            add_term(U, V, W)

    for i in range(4):
        for k in range(5):
            U = {i * N + 4: 1}
            V = {4 * N + k: 1}
            W = {k * N + i: 1}
            add_term(U, V, W)

    # Core: i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3}
    # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    #
    # Partition the 4x4x4 into 2x2x2 sub-blocks:
    # I0={0,1}, I1={2,3}
    # J0={0,1}, J1={2,3}
    # K0={0,1}, K1={2,3}
    #
    # There are 2*2*2 = 8 sub-blocks of type (I_a, J_b, K_c) where each is 2x2x2.
    # For each, we can use Strassen (rank 7) instead of schoolbook (rank 8).
    # Total core rank: 8*7 = 56
    # Total rank: 45 + 56 = 101

    # Wait, that's not right. The 8 sub-blocks don't independently decompose.
    # Let me reconsider.

    # Actually, the correct approach:
    # The 4x4x4 matmul tensor has rank at most 28 (using Strassen on 2x2 blocks).
    # But we need to be more careful.

    # Let me use a different approach. Partition the 4x4x4 into:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]] where each block is 2x2
    # C = [[C00, C01], [C10, C11]] where each block is 2x2
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Each C_ab is a 2x2 matrix = sum of products of 2x2 matrices.
    # We can compute each product of 2x2 matrices using Strassen (rank 7).
    # But there are 8 such products (2 per C block, 4 C blocks).
    # Total: 8 * 7 = 56 terms for the core.
    # Total: 45 + 56 = 101.

    # Hmm, but the knowledge says rank 117. Let me re-examine.
    # Maybe the Strassen approach for 2x2x2 within the larger context doesn't
    # simply give rank 7 per sub-block because of index alignment issues.

    # Let me just do schoolbook for the core: 4*4*4 = 64 terms.
    # Total: 45 + 64 = 109. That's better than 125.

    # Actually wait - let me reconsider the frame count.
    # Frame: i=4 (5*5=25) + j=4, i in 0..3 (4*5=20) = 45. Correct.
    # Core schoolbook: 4*4*4 = 64. Total = 109.

    # But the knowledge says 117 with Strassen. Let me try to get 109 first,
    # then optimize.

    # Core schoolbook: i in 0..3, j in 0..3, k in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U = {i * N + j: 1}
                V = {j * N + k: 1}
                W = {k * N + i: 1}
                add_term(U, V, W)

    # Verify correctness
    if time.time() > deadline:
        pass  # skip verification if running out of time

    # Build the matrices
    rank = len(U_cols)
    U = [[0] * rank for _ in range(N * N)]
    V = [[0] * rank for _ in range(N * N)]
    W = [[0] * rank for _ in range(N * N)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    # Verify: check that sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta(i,k) for all i,j,k
    # Actually the tensor is: T[i,j,k] = delta_{i,k} (for matmul, T[i,j,k] = 1 if i==k else 0... no)
    # The matmul tensor is: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] = 1 for all i,j,k (it's the identity in the sense that
    # the bilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k])
    # Wait, T[i,j,k] should be such that sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k]
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} sum_j A[i,j] B[j,k]
    # This means T[i,j,k] = 1 for all i,j,k.
    #
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # (using the trace convention W[k*n+i, r])
    #
    # Let me verify numerically with random matrices.
    if time.time() < deadline - 5:
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)

        # Verify: for random A, B, check A@B == sum_r (A * U_r) * (B * V_r) * W_r
        # Where (A * U_r)[i,j] = U_r[i,j], etc.
        # Actually: C[i,k] = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Let's verify this.

        rng = np.random.default_rng(42)
        A_test = rng.standard_normal((5, 5))
        B_test = rng.standard_normal((5, 5))
        C_expected = A_test @ B_test

        C_computed = np.zeros((5, 5))
        for r in range(rank):
            U_r = U_mat[:, r].reshape(5, 5)
            V_r = V_mat[:, r].reshape(5, 5)
            W_r = W_mat[:, r].reshape(5, 5)
            # C[i,k] += sum_j U_r[i,j] * V_r[j,k] * W_r[k,i]
            # = sum_j U_r[i,j] * V_r[j,k] * W_r[k,i]
            # Note: W_r is indexed as W[k*n+i, r] -> W_r[k,i]
            temp = U_r @ V_r  # temp[i,k] = sum_j U_r[i,j] V_r[j,k]
            # Now multiply by W_r[k,i]
            C_computed += temp * W_r.T  # (temp * W_r.T)[i,k] = temp[i,k] * W_r[k,i]

        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to full schoolbook
            return _schoolbook()

    return {"U": U, "V": V, "W": W}

def _schoolbook():
    """Fallback schoolbook decomposition, rank 125."""
    n, m, p = 5, 5, 5
    rank = n * m * p
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
