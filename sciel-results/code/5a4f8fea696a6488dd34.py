"""
Strategy: Use Strassen 2x2x2 blocks for the 2x2x8 part (rank 28),
then attempt to absorb the 2x1x8 schoolbook part (rank 16) into the
existing structure by solving for extended U, V, W matrices.
If absorption fails, fall back to the 44-term decomposition.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0

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

    # Build 44-term decomposition (Strassen 28 + schoolbook 16)
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

    # Try to reduce rank by finding linear dependencies among the 44 terms
    # Each term r defines a vector in the tensor space via:
    # vec_r[(i,a,k)] = U[i*m+a][r] * V[a*p+k][r] * W[k*n+i][r]
    # We want to find if any linear combination of these vectors is zero.

    def _term_vector(U, V, W, r):
        vec = {}
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    val = U[i*m+a][r] * V[a*p+k][r] * W[k*n+i][r]
                    if val != 0:
                        idx = (i, a, k)
                        vec[idx] = vec.get(idx, 0) + val
        return vec

    num_terms = rank_44
    all_indices = set()
    term_vecs = []
    for r in range(num_terms):
        tv = _term_vector(U44, V44, W44, r)
        term_vecs.append(tv)
        all_indices.update(tv.keys())

    all_indices = sorted(all_indices)
    dim = len(all_indices)
    idx_to_pos = {idx: pos for pos, idx in enumerate(all_indices)}

    # Build matrix M where M[r, pos] = coefficient of term r at position pos
    M = np.zeros((num_terms, dim))
    for r in range(num_terms):
        for idx, val in term_vecs[r].items():
            M[r, idx_to_pos[idx]] = val

    # Find null space of M
    try:
        U_mat, s, Vt = np.linalg.svd(M, full_matrices=False)
        rank_M = np.sum(s > 1e-10)
        nullity = num_terms - rank_M
    except:
        nullity = 0

    best_rank = rank_44
    best_candidate = {"U": U44, "V": V44, "W": W44}

    if nullity > 0 and _elapsed_ok():
        # Get null space vectors
        null_space = Vt[rank_M:]

        # Try to eliminate terms using null space vectors
        # For each null vector, find the term with largest coefficient and try to eliminate it
        eliminated = set()
        new_U = [row[:] for row in U44]
        new_V = [row[:] for row in V44]
        new_W = [row[:] for row in W44]

        for ns_vec in null_space:
            if not _elapsed_ok():
                break
            abs_ns = np.abs(ns_vec)
            # Find terms not yet eliminated with significant coefficient
            candidates = [r for r in range(num_terms) if r not in eliminated and abs_ns[r] > 1e-6]
            if not candidates:
                continue
            max_idx = max(candidates, key=lambda r: abs_ns[r])

            # Try to eliminate term max_idx
            # The null vector says: sum_r ns_vec[r] * term_r = 0
            # So term_max_idx = -sum_{r != max_idx} (ns_vec[r]/ns_vec[max_idx]) * term_r
            # This means we can replace term_max_idx with a combination of others
            # But this doesn't directly reduce rank unless we can merge terms

            # Alternative: check if we can set one of the terms to zero
            # and adjust others to compensate
            # This is complex, so use a simpler heuristic:
            # Just mark this term as potentially eliminable
            pass

    # Try a different approach: use Gaussian elimination to find minimal rank
    # by trying to eliminate terms one by one

    if _elapsed_ok():
        # Try to find a lower rank decomposition by solving for it
        # Start with the 44-term decomposition and try to reduce

        # For each term, check if it can be expressed as a combination of others
        # by solving a linear system

        # Build the full tensor matrix
        # T[i,a,k] = sum_r U[i,a,r] * V[a,k,r] * W[k,i,r] = 1 for all i,a,k

        # This is a bilinear system, which is hard to solve directly.
        # Instead, try a heuristic: randomly perturb and see if rank drops

        # Try to find if any 43-term subset can span the same tensor space
        # This is equivalent to checking if any term is in the span of the others

        # For each term r, check if term_r is in the span of {term_s : s != r}
        # This is a linear algebra problem

        for r_elim in range(num_terms):
            if not _elapsed_ok():
                break

            # Build matrix of all terms except r_elim
            M_reduced = np.delete(M, r_elim, axis=0)
            v_elim = M[r_elim]

            # Check if v_elim is in the row space of M_reduced
            # This is equivalent to checking if rank(M_reduced) == rank(M_reduced with v_elim)
            M_aug = np.vstack([M_reduced, v_elim.reshape(1, -1)])

            try:
                rank_red = np.linalg.matrix_rank(M_reduced, tol=1e-10)
                rank_aug = np.linalg.matrix_rank(M_aug, tol=1e-10)
            except:
                continue

            if rank_red == rank_aug:
                # v_elim is in the span of M_reduced, so term r_elim can be eliminated
                # Create a new decomposition with rank 43
                new_rank = num_terms - 1
                new_U = [row[:new_rank] for row in U44]
                new_V = [row[:new_rank] for row in V44]
                new_W = [row[:new_rank] for row in W44]

                # Remove column r_elim
                for i in range(n * m):
                    new_U[i] = U44[i][:r_elim] + U44[i][r_elim+1:]
                for a in range(m * p):
                    new_V[a] = V44[a][:r_elim] + V44[a][r_elim+1:]
                for k in range(p * n):
                    new_W[k] = W44[k][:r_elim] + W44[k][r_elim+1:]

                candidate = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(candidate):
                    best_rank = new_rank
                    best_candidate = candidate
                    break  # Found a lower rank, stop searching

    # If we found a better rank, return it
    if best_rank < rank_44:
        return best_candidate

    # Otherwise, return the 44-term decomposition
    candidate = {"U": U44, "V": V44, "W": W44}
    if _validate(candidate):
        return candidate
    return _schoolbook()

# EVOLVE-BLOCK-END
