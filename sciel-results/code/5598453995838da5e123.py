"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition.

Combines the clean block-Strassen construction from the inspirations with
verification and a fallback. The key structural idea is to decompose the
2x4 * 4x8 multiplication into 8 independent 2x2 * 2x2 sub-problems
(4 output column pairs x 2 intermediate row pairs), each solved with
Strassen's rank-7 algorithm, giving total rank 56.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    # For 2x2 matrices A, B, C:
    #   P1 = (A11 + A22)(B11 + B22)
    #   P2 = (A21 + A22) B11
    #   P3 = A11 (B12 - B22)
    #   P4 = A22 (B21 - B11)
    #   P5 = (A11 + A12) B22
    #   P6 = (A21 - A11) (B11 + B12)
    #   P7 = (A12 - A22) (B21 + B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    #
    # Each product P_r is defined by:
    #   U_coeffs: dict (local_i, local_j) -> coeff  for A
    #   V_coeffs: dict (local_j, local_k) -> coeff  for B
    #   W_coeffs: dict (local_k, local_i) -> coeff  for C
    #
    # Local indices: i in {0,1} (row of A / row of C), j in {0,1} (col of A / row of B), k in {0,1} (col of B / col of C)

    strassen_products = [
        # P1 = (A11 + A22)(B11 + B22) -> C11: +1, C22: +1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # P2 = (A21 + A22) B11 -> C21: +1, C22: -1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # P3 = A11 (B12 - B22) -> C12: +1, C22: +1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        # P4 = A22 (B21 - B11) -> C11: +1, C21: +1
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # P5 = (A11 + A12) B22 -> C11: -1, C12: +1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
        # P6 = (A21 - A11) (B11 + B12) -> C22: +1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # P7 = (A12 - A22) (B21 + B22) -> C11: +1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    rank = 7 * 8  # 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    # 8 blocks: 4 output column pairs (qc=0..3) x 2 intermediate row pairs (ar=0..1)
    for qc in range(4):
        for ar in range(2):
            # A uses columns 2*ar, 2*ar+1
            # B uses rows 2*ar, 2*ar+1 and columns 2*qc, 2*qc+1
            # C uses columns 2*qc, 2*qc+1 and rows 0, 1
            a_col0 = 2 * ar
            b_row0 = 2 * ar
            b_col0 = 2 * qc
            c_col0 = 2 * qc

            for u_coeffs, v_coeffs, w_coeffs in strassen_products:
                # Fill U column
                for (li, lj), coeff in u_coeffs.items():
                    gi = li
                    gj = a_col0 + lj
                    U[gi * m + gj, col] += coeff
                # Fill V column
                for (lj, lk), coeff in v_coeffs.items():
                    gj = b_row0 + lj
                    gk = b_col0 + lk
                    V[gj * p + gk, col] += coeff
                # Fill W column: W index is k*n + i (trace convention W[k*n+i, r])
                for (lk, li), coeff in w_coeffs.items():
                    gk = c_col0 + lk
                    gi = li
                    W[gk * n + gi, col] += coeff
                col += 1

            if time.time() > deadline:
                break

    # Verify correctness with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B

    # Compute C from decomposition: C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                for r in range(rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
            C_decomp[i, k] = total

    if np.max(np.abs(C_true - C_decomp)) > 1e-10:
        # Fallback: return schoolbook rank-64 solution
        rank = n * m * p  # 64
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)

    n, m, p = DIMENSIONS

    # Verify: for random A and B, check that sum_{j,r} U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] gives correct C
    np.random.seed(123)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B

    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                for r in range(U.shape[1]):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
            C_decomp[i, k] = total

    error = np.max(np.abs(C_true - C_decomp))
    print(f"Max error: {error}")
    print(f"Rank: {U.shape[1]}")
