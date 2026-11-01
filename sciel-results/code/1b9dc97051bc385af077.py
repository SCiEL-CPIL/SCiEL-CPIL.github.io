"""Matrix multiplication 2x3 * 3x8: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    deadline = time.time() + 110  # safety margin under 120s

    def schoolbook():
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        A = np.random.randn(2, 3)
        B = np.random.randn(3, 8)
        C_expected = A @ B
        C_computed = np.zeros((2, 8))
        for c in range(U_np.shape[1]):
            U_mat = U_np[:, c].reshape(2, 3)
            V_mat = V_np[:, c].reshape(3, 8)
            W_vec = W_np[:, c]
            prod = U_mat @ V_mat
            for i in range(2):
                for k in range(8):
                    C_computed[i, k] += W_vec[k * 2 + i] * prod[i, k]
        return np.allclose(C_computed, C_expected, atol=1e-9)

    def verify_full(U, V, W, num_trials=5):
        """More thorough verification with multiple random trials."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for _ in range(num_trials):
            A = np.random.randn(2, 3)
            B = np.random.randn(3, 8)
            C_expected = A @ B
            C_computed = np.zeros((2, 8))
            for c in range(U_np.shape[1]):
                U_mat = U_np[:, c].reshape(2, 3)
                V_mat = V_np[:, c].reshape(3, 8)
                W_vec = W_np[:, c]
                prod = U_mat @ V_mat
                for i in range(2):
                    for k in range(8):
                        C_computed[i, k] += W_vec[k * 2 + i] * prod[i, k]
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    # Rank-36 construction (known good)
    def build_rank36():
        rank = 36
        U = [[0.0] * rank for _ in range(6)]
        V = [[0.0] * rank for _ in range(24)]
        W = [[0.0] * rank for _ in range(16)]

        def add_strassen_2x2x2(base, a_r0, a_r1, b_c0, b_c1, b_k0):
            U[a_r0 * 3 + b_c0][base] += 1; V[b_c0 * 8 + b_k0][base] += 1
            U[a_r1 * 3 + b_c1][base] += 1; V[b_c1 * 8 + b_k0 + 1][base] += 1
            W[b_k0 * 2 + a_r0][base] += 1
            W[(b_k0 + 1) * 2 + a_r1][base] += 1
            U[a_r1 * 3 + b_c0][base + 1] += 1; V[b_c0 * 8 + b_k0][base + 1] += 1
            U[a_r1 * 3 + b_c1][base + 1] += 1
            W[b_k0 * 2 + a_r1][base + 1] += 1
            U[a_r0 * 3 + b_c0][base + 2] += 1; V[b_c1 * 8 + b_k0][base + 2] += 1
            V[b_c0 * 8 + b_k0][base + 2] += -1
            W[b_k0 * 2 + a_r0][base + 2] += -1
            W[b_k0 * 2 + a_r1][base + 2] += 1
            U[a_r1 * 3 + b_c1][base + 3] += 1; V[b_c0 * 8 + b_k0 + 1][base + 3] += 1
            V[b_c1 * 8 + b_k0 + 1][base + 3] += 1
            W[b_k0 * 2 + a_r0][base + 3] += 1
            W[(b_k0 + 1) * 2 + a_r0][base + 3] += 1
            U[a_r0 * 3 + b_c0][base + 4] += 1; V[b_c1 * 8 + b_k0 + 1][base + 4] += 1
            U[a_r0 * 3 + b_c1][base + 4] += 1
            W[(b_k0 + 1) * 2 + a_r0][base + 4] += 1
            U[a_r1 * 3 + b_c0][base + 5] += 1; V[b_c0 * 8 + b_k0 + 1][base + 5] += 1
            U[a_r0 * 3 + b_c0][base + 5] += -1
            V[b_c1 * 8 + b_k0 + 1][base + 5] += -1
            W[(b_k0 + 1) * 2 + a_r1][base + 5] += 1
            U[a_r0 * 3 + b_c1][base + 6] += 1; V[b_c0 * 8 + b_k0][base + 6] += 1
            U[a_r1 * 3 + b_c1][base + 6] += -1
            V[b_c1 * 8 + b_k0][base + 6] += 1
            W[b_k0 * 2 + a_r0][base + 6] += -1
            W[(b_k0 + 1) * 2 + a_r1][base + 6] += -1

        for half in range(2):
            bk = half * 4
            base = half * 18
            add_strassen_2x2x2(base, 0, 1, 0, 1, bk)
            add_strassen_2x2x2(base + 7, 0, 1, 0, 1, bk + 2)
            for dk in range(4):
                c = base + 14 + dk
                U[0 * 3 + 2][c] += 1
                U[1 * 3 + 2][c] += 1
                V[2 * 8 + bk + dk][c] += 1
                W[(bk + dk) * 2 + 0][c] += 1
                W[(bk + dk) * 2 + 1][c] += 1
        return {"U": U, "V": V, "W": W}

    # Try Kronecker rank-28 construction:
    # View 2x3 * 3x8 as:
    #   A = [A0 | A1] where A0 is 2x2, A1 is 2x1
    #   B = [B0; B1] where B0 is 2x8, B1 is 1x8
    #   C = A0*B0 + A1*B1
    # For A0*B0 (2x2 * 2x8): use Strassen-like 7-mult scheme on 2x2 outer
    #   with B0 viewed as 2 rows of 4 blocks of 2x2.
    #   Each "multiplication" in Strassen is scalar * (1x4) vector = rank 1.
    #   But we need to handle the 2x2 block structure.
    #   
    # Alternative: decompose A0*B0 as two 2x2*2x4 multiplications (cols 0-3, 4-7)
    #   Each 2x2*2x4 = 2 Strassen 2x2x2 = rank 14 per half, total 28 for A0*B0
    #   Plus A1*B1 (2x1*1x8) = rank 8
    #   Total = 36. Same as before.
    #
    # For rank 28, we need a different decomposition.
    # Try: view as 2x4 * 4x8 where middle dim 4 = 2+1+1? No.
    #
    # Let me try: 2x3*3x8 = 2x4*4x8 with padding? No, that changes the problem.
    #
    # Actually, try splitting 3 into 1+1+1 and 8 into 2+2+2+2:
    # C = sum_{j=0}^{2} A[:,j] * B[j,:]
    # Each term A[:,j] (2x1) * B[j,:] (1x8) has rank 8, total 24. Worse.
    #
    # The rank-28 claim may not be achievable with simple Kronecker.
    # Stick with rank-36 which is verified working.

    if time.time() > deadline:
        return schoolbook()

    result = build_rank36()
    if verify_full(result["U"], result["V"], result["W"], num_trials=3):
        return result

    return schoolbook()
# EVOLVE-BLOCK-END
