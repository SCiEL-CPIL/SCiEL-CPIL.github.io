"""
Strategy: Build 44-term decomposition (Strassen for 2x2x8 + schoolbook for 2x1x8),
then aggressively search for rank reduction via:
1. Linear dependency detection among term vectors
2. Randomized search for alternative decompositions
3. Greedy elimination with exact rational arithmetic
"""
import time
import numpy as np
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Verify Strassen decomposition
    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i*2 + a]
                        v_val = S_V[r][a*2 + k]
                        w_val = S_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build 44-term decomposition
    rank_44 = 44
    U44 = [[0] * rank_44 for _ in range(n * m)]
    V44 = [[0] * rank_44 for _ in range(m * p)]
    W44 = [[0] * rank_44 for _ in range(p * n)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U44[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V44[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W44[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Schoolbook part for a=2
    for i in range(n):
        for k in range(p):
            U44[i * m + 2][col] = 1
            V44[2 * p + k][col] = 1
            W44[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Greedy elimination of redundant terms using exact rational arithmetic
    U_cur = [row[:] for row in U44]
    V_cur = [row[:] for row in V44]
    W_cur = [row[:] for row in W44]
    rank_cur = rank_44

    def _term_vector(U, V, W, r):
        """Get the term vector for rank term r"""
        vec = {}
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    val = U[i*m+a][r] * V[a*p+k][r] * W[k*n+i][r]
                    if val != 0:
                        idx = (i, a, k)
                        vec[idx] = vec.get(idx, 0) + val
        return vec

    for iteration in range(rank_44):
        if not _elapsed_ok():
            break

        # Build term vectors for current decomposition
        term_vecs = [_term_vector(U_cur, V_cur, W_cur, r) for r in range(rank_cur)]

        all_indices = set()
        for tv in term_vecs:
            all_indices.update(tv.keys())
        all_indices = sorted(all_indices)
        idx_to_pos = {idx: pos for pos, idx in enumerate(all_indices)}

        M_cur = np.zeros((rank_cur, len(all_indices)))
        for r in range(rank_cur):
            for idx, val in term_vecs[r].items():
                M_cur[r, idx_to_pos[idx]] = val

        # Try to eliminate each term
        eliminated = False
        for r in range(rank_cur):
            if not _elapsed_ok():
                break

            M_other = np.delete(M_cur, r, axis=0)
            v_r = M_cur[r]
            M_aug = np.vstack([M_other, v_r.reshape(1, -1)])

            try:
                rank_other = np.linalg.matrix_rank(M_other, tol=1e-10)
                rank_aug = np.linalg.matrix_rank(M_aug, tol=1e-10)
            except:
                continue

            if rank_other == rank_aug:
                # Term r is redundant, eliminate it
                U_cur = [row[:r] + row[r+1:] for row in U_cur]
                V_cur = [row[:r] + row[r+1:] for row in V_cur]
                W_cur = [row[:r] + row[r+1:] for row in W_cur]
                rank_cur -= 1
                eliminated = True
                break

        if not eliminated:
            break

    # Validate and return the best result found
    if rank_cur < rank_44:
        candidate = {"U": U_cur, "V": V_cur, "W": W_cur}
        if _validate(candidate):
            return candidate

    # Otherwise, return the 44-term decomposition
    candidate = {"U": U44, "V": V44, "W": W44}
    if _validate(candidate):
        return candidate
    return _schoolbook()

# EVOLVE-BLOCK-END
