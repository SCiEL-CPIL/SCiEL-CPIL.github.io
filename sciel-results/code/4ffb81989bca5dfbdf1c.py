"""Kronecker product construction for 6x6x6 matrix multiplication rank minimization.
Uses Strassen 2x2 (rank 7) x schoolbook 3x3 (rank 27) = rank 189."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock limit

    # Strassen 2x2 patterns: (u_coeffs, v_coeffs, w_coeffs)
    # Each is a dict mapping (row, col) -> coefficient
    # For 2x2: indices are (0,0), (0,1), (1,0), (1,1)
    # u acts on A (2x2), v acts on B (2x2), w acts on C (2x2)
    strassen_2x2 = [
        # m1 = (a00+a11)*(b00+b11) -> c00 += m1, c11 += m1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # m2 = (a10+a11)*(b00) -> c10 += m2, c11 -= m2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # m3 = (a00)*(b01-b11) -> c01 += m3, c11 += m3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # m4 = (a11)*(b10-b00) -> c00 += m4, c10 += m4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # m5 = (a00+a01)*(b11) -> c00 -= m5, c01 += m5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # m6 = (a10-a00)*(b00+b01) -> c11 += m6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # m7 = (a01-a11)*(b10+b11) -> c00 += m7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # Schoolbook 3x3: 27 patterns, each is a single (i,j) entry of A times (j,k) entry of B
    # giving (i,k) entry of C
    schoolbook_3x3 = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u_c = {(i, j): 1}
                v_c = {(j, k): 1}
                w_c = {(i, k): 1}
                schoolbook_3x3.append((u_c, v_c, w_c))

    # Kronecker product construction:
    # 6x6 matrix = 3x3 blocks of 2x2 matrices
    # Entry (I*2+pi, J*2+pj) where I,J in {0,1,2} (block indices), pi,pj in {0,1} (within-block)
    #
    # The tensor M_6 = M_3 ⊗ M_2 in the sense:
    # If M_3 = sum_{a=1}^{27} u_a ⊗ v_a ⊗ w_a (each on 3x3 space, 9-dim)
    # and M_2 = sum_{b=1}^{7} u'_b ⊗ v'_b ⊗ w'_b (each on 2x2 space, 4-dim)
    # Then M_6 = sum_{a,b} (u_a ⊗ u'_b) ⊗ (v_a ⊗ v'_b) ⊗ (w_a ⊗ w'_b)
    #
    # Flattening: A 6x6 matrix entry (r, c) where r = I*2+pi, c = J*2+pj
    # maps to index r*m + c in the U matrix (which has n*m = 36 rows)
    # Similarly V has m*p = 36 rows, W has p*n = 36 rows
    # W uses convention W[k*n+i, r] so W index is k*n + i where i is row of C, k is col of C

    rank = 27 * 7  # 189
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col_idx = 0
    for a_idx, (u3, v3, w3) in enumerate(schoolbook_3x3):
        for b_idx, (u2, v2, w2) in enumerate(strassen_2x2):
            # Combine: for each (I,J) in u3 and (pi,pj) in u2:
            #   row of A = I*2+pi, col of A = J*2+pj
            #   U[(I*2+pi)*m + (J*2+pj), col_idx] += u3[(I,J)] * u2[(pi,pj)]
            for (I, J), val_u3 in u3.items():
                for (pi, pj), val_u2 in u2.items():
                    r = I * 2 + pi
                    c = J * 2 + pj
                    U[r * m + c, col_idx] += val_u3 * val_u2

            for (I, J), val_v3 in v3.items():
                for (pi, pj), val_v2 in v2.items():
                    r = I * 2 + pi
                    c = J * 2 + pj
                    V[r * p + c, col_idx] += val_v3 * val_v2

            for (I, J), val_w3 in w3.items():
                for (pi, pj), val_w2 in w2.items():
                    i = I * 2 + pi
                    k = J * 2 + pj
                    W[k * n + i, col_idx] += val_w3 * val_w2

            col_idx += 1

    # Verify the decomposition
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    C_computed = np.zeros((n, p))
    for r in range(rank):
        M_u = np.dot(U[:, r], A_test.flatten())
        M_v = np.dot(V[:, r], B_test.flatten())
        M_r = M_u * M_v
        C_computed += M_r * W[:, r].reshape(p, n).T

    error = np.max(np.abs(C_computed - C_expected))

    if error > 1e-10:
        # Fallback: schoolbook 6x6x6 (rank 216)
        rank_fallback = n * m * p
        U_f = np.zeros((n * m, rank_fallback))
        V_f = np.zeros((m * p, rank_fallback))
        W_f = np.zeros((p * n, rank_fallback))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, idx] = 1
                    V_f[j * p + k, idx] = 1
                    W_f[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
