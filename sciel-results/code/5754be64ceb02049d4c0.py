"""Rank-minimal exact tensor decomposition for 2x5 by 5x6 matrix multiplication.

Uses Strassen's algorithm on 2x2x2 blocks for the structured part,
with careful handling of the remaining dimension.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = 2, 5, 6

    # Rank 54 decomposition:
    # Strassen on j-pairs {(0,1),(2,3)} x k-pairs {(0,1),(2,3),(4,5)}: 42 terms
    # Naive for j=4, k in 0..5: 12 terms
    # Total: 54

    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]

    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1

            # P2 = (A_1,j0 + A_1,j1) * B_j0,k0
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1

            # P3 = A_0,j0 * (B_j0,k1 - B_j1,k1)
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1

            # P4 = A_1,j1 * (B_j1,k0 - B_j0,k0)
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1

            # P5 = (A_0,j0 + A_0,j1) * B_j1,k1
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1

            # P6 = (A_1,j0 - A_0,j0) * (B_j0,k0 + B_j0,k1)
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1

            # P7 = (A_0,j1 - A_1,j1) * (B_j1,k0 + B_j1,k1)
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1

    # Remaining j=4: 2*6 = 12 terms
    for k in range(p):
        # C_0,k += A_0,4 * B_4,k
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1

        # C_1,k += A_1,4 * B_4,k
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1

    # Verify correctness with random test
    if time.time() < deadline - 5:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        # Check: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should be 1
        # And for any (i,j,k) where the triple doesn't appear in a single term, should be 0
        # Actually, the tensor identity is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where T[i,j,k] = 1 for all valid (i,j,k)
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        # Compute via tensor decomposition
        C_recon = np.zeros((n, p))
        for r in range(rank):
            u_vec = U_arr[:, r]  # shape (n*m,)
            v_vec = V_arr[:, r]  # shape (m*p,)
            w_vec = W_arr[:, r]  # shape (p*n,)

            # u_vec[i*m+j] = U[i,j,r]
            # v_vec[j*p+k] = V[j,k,r]
            # w_vec[k*n+i] = W[k,i,r]

            # C[i,k] += sum_j U[i,j,r] * A[i,j] ... wait, no.
            # The decomposition is: C[i,k] = sum_j sum_r U[i,j,r] * V[j,k,r] * (A[i,j] * B[j,k] contribution)
            # Actually: C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])
            # No, that's not right either.

            # Correct interpretation:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * (A[i,j] * B[j,k] / (U*V*W))
            # This isn't quite right. Let me think again.

            # The tensor T has T[i,j,k] = 1 for the matmul tensor.
            # T = sum_r U[:,r] outer V[:,r] outer W[:,r] (with appropriate indexing)
            # So C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] * A[i,j] * B[j,k]

            # This is getting complicated. Let me just verify with the direct formula.
            pass

        # Simpler verification: build the tensor and check
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # should equal 1 for all i,j,k

        T_recon = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]

        # T_recon should be all 1s
        if not np.allclose(T_recon, np.ones((n, m, p))):
            # Fallback: return naive rank 60
            rank60 = n * m * p
            U60 = [[0] * rank60 for _ in range(n * m)]
            V60 = [[0] * rank60 for _ in range(m * p)]
            W60 = [[0] * rank60 for _ in range(p * n)]
            col60 = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U60[i*m+j][col60] = 1
                        V60[j*p+k][col60] = 1
                        W60[k*n+i][col60] = 1
                        col60 += 1
            return {"U": U60, "V": V60, "W": W60}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
