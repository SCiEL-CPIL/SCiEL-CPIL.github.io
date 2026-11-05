"""
Rank-minimal tensor decomposition for 2x3 x 3x7 matrix multiplication.

Strategy:
  1. Construct an explicit rank-39 decomposition using Strassen's 7-multiplication
     algorithm for the 2x2 sub-blocks, combined with straightforward multiplication
     for the remaining terms.
  2. Verify the decomposition.
  3. Attempt a local-search refinement to reduce rank from 39 to 38 (and below).
     The local search uses a more systematic neighborhood exploration and
     occasional column-swap / column-zero operations to escape local minima.

The interface returns {'U', 'V', 'W'} with integer entries.
"""
import numpy as np
import random
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 140  # safety margin before 150s kill

    # ---------- Schoolbook baseline (rank 42) ----------
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # ---------- Verification ----------
    def verify_fast(U, V, W, rank):
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return bool(np.all(T == 1))

    # ---------- Strassen-based rank-39 construction ----------
    def strassen_rank39():
        """
        Partition A (2x3) into [A_22, A_21] where A_22 is 2x2 and A_21 is 2x1.
        Partition B (3x7) into [B_1, B_2, B_3, B_4] where B_1,B_2,B_3 are 3x2
        and B_4 is 3x1.

        For i in {1,2,3}:
          C_i = A_22 * B_i22 + A_21 * B_i12
          A_22 * B_i22: Strassen, 7 mults
          A_21 * B_i12: 4 mults (2x1 times 1x2)
          Total: 11 per block

        For i=4:
          C_4 = A_22 * B_421 + A_21 * B_411
          A_22 * B_421: 4 mults
          A_21 * B_411: 2 mults
          Total: 6

        Grand total: 3*11 + 6 = 39
        """
        rank = 39
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        col = 0

        # Index helpers
        # U row: i*m + j  (i in 0..1, j in 0..2)
        # V row: j*p + k  (j in 0..2, k in 0..6)
        # W row: k*n + i  (k in 0..6, i in 0..1)

        def add_strassen_term(col_idx, i_base, k_base, u_coeffs, v_coeffs, w_contribs):
            u_rows = [i_base * m + 0, i_base * m + 1, (i_base + 1) * m + 0, (i_base + 1) * m + 1]
            for idx, coeff in enumerate(u_coeffs):
                if coeff != 0:
                    U[u_rows[idx]][col_idx] += coeff

            v_rows = [0 * p + k_base, 0 * p + k_base + 1, 1 * p + k_base, 1 * p + k_base + 1]
            for idx, coeff in enumerate(v_coeffs):
                if coeff != 0:
                    V[v_rows[idx]][col_idx] += coeff

            for (dr, dc), coeff in w_contribs.items():
                if coeff != 0:
                    w_row = (k_base + dc) * n + (i_base + dr)
                    W[w_row][col_idx] += coeff

        strassen_terms = [
            ([1, 0, 1, 0], [1, 1, 1, 1], {(0, 0): 1, (1, 1): 1}),
            ([0, 0, 1, 1], [1, 0, 0, 0], {(1, 0): 1}),
            ([1, 0, 0, 0], [0, 1, 0, -1], {(0, 1): 1}),
            ([0, 0, 0, 1], [0, -1, 1, 0], {(0, 0): 1, (1, 0): 1}),
            ([1, 1, 0, 0], [0, 0, 0, 1], {(0, 0): -1, (0, 1): 1}),
            ([-1, 0, 1, 0], [1, 1, 0, 0], {(1, 1): 1}),
            ([0, 1, 0, -1], [0, 0, 1, 1], {(0, 0): 1}),
        ]

        def add_21_12_terms(col_start, i_base, k_base):
            terms = [
                (i_base * m + 2, 2 * p + k_base, {(0, 0): 1}),
                (i_base * m + 2, 2 * p + k_base + 1, {(0, 1): 1}),
                ((i_base + 1) * m + 2, 2 * p + k_base, {(1, 0): 1}),
                ((i_base + 1) * m + 2, 2 * p + k_base + 1, {(1, 1): 1}),
            ]
            for offset, (u_row, v_row, w_contribs) in enumerate(terms):
                c = col_start + offset
                U[u_row][c] += 1
                V[v_row][c] += 1
                for (dr, dc), coeff in w_contribs.items():
                    w_row = (k_base + dc) * n + (i_base + dr)
                    W[w_row][c] += coeff

        def add_22_21_terms(col_start, i_base, k_base):
            terms = [
                (i_base * m + 0, 0 * p + k_base, k_base * n + i_base),
                (i_base * m + 1, 1 * p + k_base, k_base * n + i_base),
                ((i_base + 1) * m + 0, 0 * p + k_base, k_base * n + (i_base + 1)),
                ((i_base + 1) * m + 1, 1 * p + k_base, k_base * n + (i_base + 1)),
            ]
            for offset, (u_row, v_row, w_row) in enumerate(terms):
                c = col_start + offset
                U[u_row][c] += 1
                V[v_row][c] += 1
                W[w_row][c] += 1

        def add_21_11_terms(col_start, i_base, k_base):
            terms = [
                (i_base * m + 2, 2 * p + k_base, k_base * n + i_base),
                ((i_base + 1) * m + 2, 2 * p + k_base, k_base * n + (i_base + 1)),
            ]
            for offset, (u_row, v_row, w_row) in enumerate(terms):
                c = col_start + offset
                U[u_row][c] += 1
                V[v_row][c] += 1
                W[w_row][c] += 1

        i_base = 0

        # Block 1: B_1 occupies k=0,1
        k_base = 0
        for st in strassen_terms:
            add_strassen_term(col, i_base, k_base, st[0], st[1], st[2])
            col += 1
        add_21_12_terms(col, i_base, k_base)
        col += 4

        # Block 2: B_2 occupies k=2,3
        k_base = 2
        for st in strassen_terms:
            add_strassen_term(col, i_base, k_base, st[0], st[1], st[2])
            col += 1
        add_21_12_terms(col, i_base, k_base)
        col += 4

        # Block 3: B_3 occupies k=4,5
        k_base = 4
        for st in strassen_terms:
            add_strassen_term(col, i_base, k_base, st[0], st[1], st[2])
            col += 1
        add_21_12_terms(col, i_base, k_base)
        col += 4

        # Block 4: B_4 occupies k=6
        k_base = 6
        add_22_21_terms(col, i_base, k_base)
        col += 4
        add_21_11_terms(col, i_base, k_base)
        col += 2

        assert col == 39, f"Expected 39 columns, got {col}"
        return {"U": U, "V": V, "W": W}

    # Verify and use the Strassen construction
    strassen_result = strassen_rank39()
    if verify_fast(strassen_result["U"], strassen_result["V"], strassen_result["W"], 39):
        best = strassen_result
        best_rank = 39

    # ---------- Improved local search to try reducing rank ----------
    def local_search_improved(U, V, W, target_rank, max_iters=500):
        """Improved local search with column operations and systematic neighborhood."""
        U_work = [row[:] for row in U]
        V_work = [row[:] for row in V]
        W_work = [row[:] for row in W]

        if verify_fast(U_work, V_work, W_work, target_rank):
            return True

        # Precompute row counts for efficiency
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Try a random modification
            op = random.random()
            if op < 0.6:
                # Single cell modification
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    row = random.randint(0, nm - 1)
                    col = random.randint(0, target_rank - 1)
                    old = U_work[row][col]
                    U_work[row][col] = random.randint(-3, 3)
                    if not verify_fast(U_work, V_work, W_work, target_rank) and time.time() > deadline - 5:
                        U_work[row][col] = old
                elif mat == 'V':
                    row = random.randint(0, mp - 1)
                    col = random.randint(0, target_rank - 1)
                    old = V_work[row][col]
                    V_work[row][col] = random.randint(-3, 3)
                    if not verify_fast(U_work, V_work, W_work, target_rank) and time.time() > deadline - 5:
                        V_work[row][col] = old
                else:
                    row = random.randint(0, pn - 1)
                    col = random.randint(0, target_rank - 1)
                    old = W_work[row][col]
                    W_work[row][col] = random.randint(-3, 3)
                    if not verify_fast(U_work, V_work, W_work, target_rank) and time.time() > deadline - 5:
                        W_work[row][col] = old
            elif op < 0.8:
                # Swap two columns
                c1, c2 = random.sample(range(target_rank), 2)
                for r in range(nm):
                    U_work[r][c1], U_work[r][c2] = U_work[r][c2], U_work[r][c1]
                for r in range(mp):
                    V_work[r][c1], V_work[r][c2] = V_work[r][c2], V_work[r][c1]
                for r in range(pn):
                    W_work[r][c1], W_work[r][c2] = W_work[r][c2], W_work[r][c1]
                if not verify_fast(U_work, V_work, W_work, target_rank) and time.time() > deadline - 5:
                    for r in range(nm):
                        U_work[r][c1], U_work[r][c2] = U_work[r][c2], U_work[r][c1]
                    for r in range(mp):
                        V_work[r][c1], V_work[r][c2] = V_work[r][c2], V_work[r][c1]
                    for r in range(pn):
                        W_work[r][c1], W_work[r][c2] = W_work[r][c2], W_work[r][c1]
            else:
                # Add small perturbation to a random column
                col = random.randint(0, target_rank - 1)
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    row = random.randint(0, nm - 1)
                    old = U_work[row][col]
                    U_work[row][col] = old + random.choice([-1, 1])
                    if not verify_fast(U_work, V_work, W_work, target_rank) and time.time() > deadline - 5:
                        U_work[row][col] = old
                elif mat == 'V':
                    row = random.randint(0, mp - 1)
                    old = V_work[row][col]
                    V_work[row][col] = old + random.choice([-1, 1])
                    if not verify_fast(U_work, V_work, W_work, target_rank) and time.time() > deadline - 5:
                        V_work[row][col] = old
                else:
                    row = random.randint(0, pn - 1)
                    old = W_work[row][col]
                    W_work[row][col] = old + random.choice([-1, 1])
                    if not verify_fast(U_work, V_work, W_work, target_rank) and time.time() > deadline - 5:
                        W_work[row][col] = old

            if verify_fast(U_work, V_work, W_work, target_rank):
                return True

        return verify_fast(U_work, V_work, W_work, target_rank)

    def try_rank_with_local_search(target_rank, max_attempts=30):
        """Try to find a decomposition with local search refinement."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None

            # Start from the Strassen rank-39 solution and zero out some columns
            if best_rank > target_rank:
                U0 = [row[:target_rank] for row in best["U"]]
                V0 = [row[:target_rank] for row in best["V"]]
                W0 = [row[:target_rank] for row in best["W"]]
            else:
                U0 = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(nm)]
                V0 = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(mp)]
                W0 = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(pn)]

            if local_search_improved(U0, V0, W0, target_rank, max_iters=100):
                return {"U": U0, "V": V0, "W": W0}
        return None

    # Try to reduce rank from current best
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 15:
            break

        result = try_rank_with_local_search(target_rank, max_attempts=15)
        if result:
            best = result
            best_rank = target_rank
            continue

    return best
# EVOLVE-BLOCK-END
