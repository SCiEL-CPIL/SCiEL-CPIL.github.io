import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0

def construct():
    global _START_TIME, _DEADLINE

    n, m, p = 3, 4, 8

    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: schoolbook rank 96
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Strassen 2x2x2 decomposition coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)B01
    # M7 = (A01-A11)(B00+B10)

    # A coeffs (7 products x 4 elements: [i0j0, i0j1, i1j0, i1j1])
    S_A = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=np.int64)

    # B coeffs (7 products x 4 elements: [j0k0, j0k1, j1k0, j1k1])
    S_B = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=np.int64)

    # C coeffs (7 products x 4 elements: [i0k0, i0k1, i1k0, i1k1])
    S_C = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=np.int64)

    # Rank-88 decomposition:
    # i=0: 1x4x8 schoolbook = 32 terms
    # i=1,2: 2x4x8 with Strassen on (i,j) for each (j-pair, k-pair) = 8*7 = 56 terms
    rank = 88
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0

    # Part 1: i=0 (1x4x8 schoolbook)
    for j in range(m):
        for k in range(p):
            U[j, col] = 1
            V[j * p + k, col] = 1
            W[k * n, col] = 1
            col += 1

    # Part 2: i=1,2 (2x4x8 Strassen on (i,j) plane)
    for jb in range(2):
        j_base = jb * 2
        for kb in range(4):
            k_base = kb * 2
            for t in range(7):
                # U: rows i=1,2; cols j in [j_base, j_base+2)
                for l in range(4):
                    c = S_A[t, l]
                    if c != 0:
                        li = l >> 1
                        lj = l & 1
                        U[(1 + li) * m + j_base + lj, col] = c

                # V: rows j in [j_base, j_base+2); cols k in [k_base, k_base+2)
                for l in range(4):
                    c = S_B[t, l]
                    if c != 0:
                        lj = l >> 1
                        lk = l & 1
                        V[(j_base + lj) * p + k_base + lk, col] = c

                # W: rows k in [k_base, k_base+2); cols i=1,2
                for l in range(4):
                    c = S_C[t, l]
                    if c != 0:
                        li = l >> 1
                        lk = l & 1
                        W[(k_base + lk) * n + 1 + li, col] = c

                col += 1

    assert col == rank, f"Rank mismatch: {col} != {rank}"

    # Quick sanity check: verify tensor identity on a random test
    rng = np.random.default_rng(42)
    A_test = rng.integers(-2, 3, size=(n, m))
    B_test = rng.integers(-2, 3, size=(m, p))
    C_test = A_test @ B_test

    # Reconstruct C from decomposition: C[i,k] = sum_r W[k*n+i, r] * sum_j U[i*m+j, r] * V[j*p+k, r]
    C_recon = np.zeros((n, p), dtype=np.float64)
    Uf = U.astype(np.float64)
    Vf = V.astype(np.float64)
    Wf = W.astype(np.float64)
    for r in range(rank):
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += Uf[i * m + j, r] * Vf[j * p + k, r]
                C_recon[i, k] += Wf[k * n + i, r] * s

    if not np.allclose(C_recon, C_test, atol=1e-10):
        # If verification fails, fall back to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=np.int64)
        V_fb = np.zeros((m * p, rank_fb), dtype=np.int64)
        W_fb = np.zeros((p * n, rank_fb), dtype=np.int64)
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col_fb] = 1
                    V_fb[j * p + k, col_fb] = 1
                    W_fb[k * n + i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
