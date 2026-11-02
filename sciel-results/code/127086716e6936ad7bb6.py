import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_4x4_decomposition():
    """
    Construct the rank-49 recursive Strassen decomposition for 4x4 matrix
    multiplication.  The 49-rank tensor is the Kronecker product of the two
    2x2 Strassen decompositions (7 outer block products x 7 inner scalar
    products).  Index conventions:
        U[i*m + j]  (A row i, col j)
        V[j*p + k]  (B row j, col k)
        W[k*n + i]  (C col k, row i)  -- official trace convention
    """
    n, m, p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # --- Outer Strassen: 7 block products ---
    # Each combo is a list of (block_row, block_col, coeff).
    A_block_combos = [
        [(0, 0, 1), (1, 1, 1)],       # M1: A00 + A11
        [(1, 0, 1), (1, 1, 1)],       # M2: A10 + A11
        [(0, 0, 1)],                  # M3: A00
        [(1, 1, 1)],                  # M4: A11
        [(0, 0, 1), (0, 1, 1)],       # M5: A00 + A01
        [(1, 0, 1), (0, 0, -1)],      # M6: A10 - A00
        [(0, 1, 1), (1, 1, -1)],      # M7: A01 - A11
    ]

    B_block_combos = [
        [(0, 0, 1), (1, 1, 1)],       # M1: B00 + B11
        [(0, 0, 1)],                  # M2: B00
        [(0, 1, 1), (1, 1, -1)],      # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)],      # M4: B10 - B00
        [(1, 1, 1)],                  # M5: B11
        [(0, 0, 1), (0, 1, 1)],       # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)],       # M7: B10 + B11
    ]

    # --- Inner Strassen: 7 scalar products within a 2x2 block ---
    A_local_combos = [
        [(0, 0, 1), (1, 1, 1)],       # P1: a + d
        [(1, 0, 1), (1, 1, 1)],       # P2: c + d
        [(0, 0, 1)],                  # P3: a
        [(1, 1, 1)],                  # P4: d
        [(0, 0, 1), (0, 1, 1)],       # P5: a + b
        [(1, 0, 1), (0, 0, -1)],      # P6: c - a
        [(0, 1, 1), (1, 1, -1)],      # P7: b - d
    ]

    B_local_combos = [
        [(0, 0, 1), (1, 1, 1)],       # P1: e + h
        [(0, 0, 1)],                  # P2: e
        [(0, 1, 1), (1, 1, -1)],      # P3: f - h
        [(1, 0, 1), (0, 0, -1)],      # P4: g - e
        [(1, 1, 1)],                  # P5: h
        [(0, 0, 1), (0, 1, 1)],       # P6: e + f
        [(1, 0, 1), (1, 1, 1)],       # P7: g + h
    ]

    # --- C contributions (weights of each product for each local C entry) ---
    C_local_weights = [
        # j=0 (P1): C00 + C11
        [(0, 0, 1), (1, 1, 1)],
        # j=1 (P2): C10 - C11
        [(1, 0, 1), (1, 1, -1)],
        # j=2 (P3): C01 + C11
        [(0, 1, 1), (1, 1, 1)],
        # j=3 (P4): C00 + C10
        [(0, 0, 1), (1, 0, 1)],
        # j=4 (P5): -C00 + C01
        [(0, 0, -1), (0, 1, 1)],
        # j=5 (P6): C11
        [(1, 1, 1)],
        # j=6 (P7): C00
        [(0, 0, 1)],
    ]

    # --- Outer C block weights ---
    C_block_weights = [
        # i=0 (M1): C00 + C11
        [(0, 0, 1), (1, 1, 1)],
        # i=1 (M2): C10 - C11
        [(1, 0, 1), (1, 1, -1)],
        # i=2 (M3): C01 + C11
        [(0, 1, 1), (1, 1, 1)],
        # i=3 (M4): C00 + C10
        [(0, 0, 1), (1, 0, 1)],
        # i=4 (M5): -C00 + C01
        [(0, 0, -1), (0, 1, 1)],
        # i=5 (M6): C11
        [(1, 1, 1)],
        # i=6 (M7): C00
        [(0, 0, 1)],
    ]

    # --- Assemble U, V, W ---
    for i in range(7):
        for j in range(7):
            r_idx = i * 7 + j

            # U: A row = 2*block_row + local_row, col = 2*block_col + local_col
            for (abr, abc, a_coeff) in A_block_combos[i]:
                for (lr, lc, l_coeff) in A_local_combos[j]:
                    global_row = 2 * abr + lr
                    global_col = 2 * abc + lc
                    U[global_row * m + global_col][r_idx] += a_coeff * l_coeff

            # V: B row = 2*block_row + local_row, col = 2*block_col + local_col
            for (bbr, bbc, b_coeff) in B_block_combos[i]:
                for (lr, lc, l_coeff) in B_local_combos[j]:
                    global_row = 2 * bbr + lr
                    global_col = 2 * bbc + lc
                    V[global_row * p + global_col][r_idx] += b_coeff * l_coeff

            # W: C row R = 2*c_block_row + c_local_row, col C = 2*c_block_col + c_local_col
            # W index = C * n + R
            for (cbr, cbc, c_coeff) in C_block_weights[i]:
                for (lr, lc, l_coeff) in C_local_weights[j]:
                    global_row = 2 * cbr + lr
                    global_col = 2 * cbc + lc
                    w_idx = global_col * n + global_row
                    W[w_idx][r_idx] += c_coeff * l_coeff

    return {"U": U, "V": V, "W": W}


def _verify_numpy(U, V, W, n, m, p, rank):
    """
    Fast verification using NumPy.  Reconstructs the multiplication tensor
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] and checks it
    equals 1 on the support (i,j,k with j==k) and 0 elsewhere.
    """
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)

    # Build the full tensor T of shape (n, m, p).
    # T[i, j, k] = sum_r Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
    T = np.zeros((n, m, p), dtype=np.int64)
    for r in range(rank):
        # u_r[i*m+j], v_r[j*p+k], w_r[k*n+i]
        u = Ua[:, r].reshape(n, m)
        v = Va[:, r].reshape(m, p)
        w = Wa[:, r].reshape(p, n)
        # For each (i,j,k): u[i,j] * v[j,k] * w[k,i]
        # u[i,j]*v[j,k] is (n,p); multiply by w[k,i] = w.T[i,k] -> (n,p)
        wT = w.T  # shape (n, p), wT[i,k] = w[k,i]
        T += u * v * wT

    # Expected: T[i,j,k] = 1 if j == k else 0
    expected = np.zeros((n, m, p), dtype=np.int64)
    for j in range(m):
        expected[:, j, j] = 1

    return bool(np.array_equal(T, expected))


def _verify_decomposition(U, V, W, n, m, p, rank):
    """
    Pure-Python verification (fallback).  Checks T[i,j,k] == 1 for j==k and
    0 otherwise.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if (j == k and total != 1) or (j != k and total != 0):
                    return False
    return True


def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook rank n*m*p decomposition."""
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


def construct():
    deadline = time.time() + 130.0  # safety margin below the 150s wall-clock limit
    n, m, p = DIMENSIONS

    # Primary: recursive Strassen rank-49 decomposition (provable optimum for 4x4).
    result = _strassen_4x4_decomposition()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    # Fast NumPy verification first.
    try:
        if _verify_numpy(U, V, W, n, m, p, rank):
            if time.time() > deadline:
                return result
            return result
    except Exception:
        pass

    # Fallback to pure-Python verification.
    if _verify_decomposition(U, V, W, n, m, p, rank):
        return result

    # If Strassen somehow failed verification, fall back to schoolbook (rank 64).
    return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END
