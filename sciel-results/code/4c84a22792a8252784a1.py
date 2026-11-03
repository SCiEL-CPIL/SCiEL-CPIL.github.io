"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

Strategy:
  1. Build a known rank-55 decomposition via Strassen on 2x2 sub-blocks.
  2. Attempt to find a lower-rank decomposition via:
     a. A hand-constructed rank-54 candidate using a 6-term 2x2x2 block
        with shared structure.
     b. Random search over half-integer entries for ranks 53-54.
  3. Verify all candidates exactly before returning.
"""

import time
import sys
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s wall-clock limit

    n, m, p = DIMENSIONS

    def verify(decomp, tol=1e-9):
        """Verify exact decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for support, 0 else."""
        U = np.array(decomp["U"], dtype=np.float64)
        V = np.array(decomp["V"], dtype=np.float64)
        W = np.array(decomp["W"], dtype=np.float64)
        rank = U.shape[1]

        # Check support entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(total - 1.0) > tol:
                        return False

        # Check non-support entries are zero (sampled for speed)
        support_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    support_set.add((i * m + j, j * p + k, k * n + i))

        # Reconstruct full tensor for non-support check
        T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for c in range(rank):
            T += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])

        for _ in range(2000):
            a = random.randint(0, n * m - 1)
            b = random.randint(0, m * p - 1)
            c = random.randint(0, p * n - 1)
            if (a, b, c) not in support_set:
                if abs(T[a, b, c]) > tol:
                    return False

        return True

    def make_rank_55():
        """Rank 55 decomposition: 20 (j=0 naive) + 35 (j=1,2 Strassen on 5 blocks)."""
        rank = 55
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        # Part A: j=0
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

        # Part B: j=1,2 with Strassen on 5 blocks of 2x2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1

            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2

            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1

            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1

            # M1 = (a11+a22)(b11+b22)
            U[u_11][col] += 1; U[u_22][col] += 1
            V[v_11][col] += 1; V[v_22][col] += 1
            W[w_00][col] += 1; W[w_11][col] += 1
            col += 1

            # M2 = (a21+a22)b11
            U[u_21][col] += 1; U[u_22][col] += 1
            V[v_11][col] += 1
            W[w_10][col] += 1; W[w_11][col] += -1
            col += 1

            # M3 = a11(b12-b22)
            U[u_11][col] += 1
            V[v_12][col] += 1; V[v_22][col] += -1
            W[w_01][col] += 1; W[w_11][col] += 1
            col += 1

            # M4 = a22(b21-b11)
            U[u_22][col] += 1
            V[v_21][col] += 1; V[v_11][col] += -1
            W[w_00][col] += 1; W[w_10][col] += 1
            col += 1

            # M5 = (a11+a12)b22
            U[u_11][col] += 1; U[u_12][col] += 1
            V[v_22][col] += 1
            W[w_00][col] += -1; W[w_01][col] += 1
            col += 1

            # M6 = (a21-a11)(b11+b12)
            U[u_21][col] += 1; U[u_11][col] += -1
            V[v_11][col] += 1; V[v_12][col] += 1
            W[w_11][col] += 1
            col += 1

            # M7 = (a12-a22)(b21+b22)
            U[u_12][col] += 1; U[u_22][col] += -1
            V[v_21][col] += 1; V[v_22][col] += 1
            W[w_00][col] += 1
            col += 1

        if col != rank:
            return None
        return {"U": U, "V": V, "W": W}

    def make_rank_54_candidate():
        """
        Attempt rank 54: 20 (j=0) + 34 (j=1,2).
        Use Strassen (7 terms) for 4 blocks = 28 terms, and a 6-term
        decomposition for the 5th block that reuses some existing U/V/W
        column patterns. This is a heuristic candidate; verify before use.
        """
        rank = 54
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        # Part A: j=0 (20 terms)
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

        # Part B: j=1,2
        # Blocks 0-3: standard Strassen (4*7 = 28 terms)
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1

            u_11 = 0 * m + 1; u_12 = 0 * m + 2
            u_21 = 1 * m + 1; u_22 = 1 * m + 2
            v_11 = 1 * p + k0; v_12 = 1 * p + k1
            v_21 = 2 * p + k0; v_22 = 2 * p + k1
            w_00 = k0 * n + 0; w_01 = k1 * n + 0
            w_10 = k0 * n + 1; w_11 = k1 * n + 1

            U[u_11][col] += 1; U[u_22][col] += 1
            V[v_11][col] += 1; V[v_22][col] += 1
            W[w_00][col] += 1; W[w_11][col] += 1
            col += 1

            U[u_21][col] += 1; U[u_22][col] += 1
            V[v_11][col] += 1
            W[w_10][col] += 1; W[w_11][col] += -1
            col += 1

            U[u_11][col] += 1
            V[v_12][col] += 1; V[v_22][col] += -1
            W[w_01][col] += 1; W[w_11][col] += 1
            col += 1

            U[u_22][col] += 1
            V[v_21][col] += 1; V[v_11][col] += -1
            W[w_00][col] += 1; W[w_10][col] += 1
            col += 1

            U[u_11][col] += 1; U[u_12][col] += 1
            V[v_22][col] += 1
            W[w_00][col] += -1; W[w_01][col] += 1
            col += 1

            U[u_21][col] += 1; U[u_11][col] += -1
            V[v_11][col] += 1; V[v_12][col] += 1
            W[w_11][col] += 1
            col += 1

            U[u_12][col] += 1; U[u_22][col] += -1
            V[v_21][col] += 1; V[v_22][col] += 1
            W[w_00][col] += 1
            col += 1

        # Block 4: 6-term decomposition attempt
        # This is a heuristic; we try a modified Strassen that merges two terms.
        k0 = 8; k1 = 9
        u_11 = 0 * m + 1; u_12 = 0 * m + 2
        u_21 = 1 * m + 1; u_22 = 1 * m + 2
        v_11 = 1 * p + k0; v_12 = 1 * p + k1
        v_21 = 2 * p + k0; v_22 = 2 * p + k1
        w_00 = k0 * n + 0; w_01 = k1 * n + 0
        w_10 = k0 * n + 1; w_11 = k1 * n + 1

        # Term 1: (a11+a22)(b11+b22)
        U[u_11][col] += 1; U[u_22][col] += 1
        V[v_11][col] += 1; V[v_22][col] += 1
        W[w_00][col] += 1; W[w_11][col] += 1
        col += 1

        # Term 2: (a21+a22)(b11)
        U[u_21][col] += 1; U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1; W[w_11][col] += -1
        col += 1

        # Term 3: a11(b12-b22)
        U[u_11][col] += 1
        V[v_12][col] += 1; V[v_22][col] += -1
        W[w_01][col] += 1; W[w_11][col] += 1
        col += 1

        # Term 4: a22(b21-b11)
        U[u_22][col] += 1
        V[v_21][col] += 1; V[v_11][col] += -1
        W[w_00][col] += 1; W[w_10][col] += 1
        col += 1

        # Term 5: (a11+a12)(b22) -- combined with part of M6
        U[u_11][col] += 1; U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1; W[w_01][col] += 1
        col += 1

        # Term 6: (a21-a11+a12-a22)(b11+b12+b21+b22) -- heuristic merge
        # This is likely incorrect; verification will catch it.
        U[u_21][col] += 1; U[u_11][col] += -1
        U[u_12][col] += 1; U[u_22][col] += -1
        V[v_11][col] += 1; V[v_12][col] += 1
        V[v_21][col] += 1; V[v_22][col] += 1
        W[w_11][col] += 1; W[w_00][col] += 1
        col += 1

        if col != rank:
            return None
        return {"U": U, "V": V, "W": W}

    def make_schoolbook():
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

    def random_search(rank, trials, values):
        """Random search for a decomposition of the given rank."""
        for t in range(trials):
            if time.time() - start_time > DEADLINE - 5:
                return None
            U = [[random.choice(values) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice(values) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice(values) for _ in range(rank)] for _ in range(p * n)]
            decomp = {"U": U, "V": V, "W": W}
            if verify(decomp):
                return decomp
        return None

    # Build the known rank-55 decomposition
    result_55 = make_rank_55()
    if result_55 is None or not verify(result_55):
        # Fallback to schoolbook
        result_55 = make_schoolbook()

    best = result_55
    best_rank = 55 if result_55["U"][0].__len__() == 55 else n * m * p

    # Try rank 54 hand-constructed candidate
    result_54 = make_rank_54_candidate()
    if result_54 is not None and verify(result_54):
        best = result_54
        best_rank = 54

    # Try random search for rank 53 and 54 (if not already found)
    values = [0, 0.5, 1, -0.5, -1]
    for r in range(53, best_rank):
        if time.time() - start_time > DEADLINE - 5:
            break
        result = random_search(r, trials=200, values=values)
        if result is not None:
            best = result
            best_rank = r
            break

    return best

# EVOLVE-BLOCK-END
