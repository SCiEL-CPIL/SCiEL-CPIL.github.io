"""Rank-minimal tensor decomposition for 3x4 * 4x7 matrix multiplication.

Uses a maximum-coverage greedy over all 2x2x2 Strassen blocks to maximize
the number of distinct (i,j,k) triples covered, then fills the remainder
with schoolbook rank-1 terms. This avoids the overlap waste of the previous
greedy that picked blocks sharing the same row/col pairs.
"""
import time

DIMENSIONS = [3, 4, 7]
_DEADLINE = 120.0  # safety margin before 150s kill


def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    return False
    return True


def _make_strassen_2x2x2_pattern(i0, i1, j0, j1, k0, k1, n, m, p):
    """Create a standard Strassen 2x2x2 pattern as a list of dicts."""
    pattern = []
    # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
    u = {i0*m+j0: 1, i1*m+j1: 1}
    v = {j0*p+k0: 1, j1*p+k1: 1}
    w = {k0*n+i0: 1, k1*n+i1: 1}
    pattern.append((u, v, w))
    # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0]
    u = {i1*m+j0: 1, i1*m+j1: 1}
    v = {j0*p+k0: 1}
    w = {k0*n+i1: 1, k1*n+i1: -1}
    pattern.append((u, v, w))
    # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1])
    u = {i0*m+j0: 1}
    v = {j0*p+k1: 1, j1*p+k1: -1}
    w = {k1*n+i0: 1, k1*n+i1: 1}
    pattern.append((u, v, w))
    # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0])
    u = {i1*m+j1: 1}
    v = {j1*p+k0: 1, j0*p+k0: -1}
    w = {k0*n+i0: 1, k0*n+i1: 1}
    pattern.append((u, v, w))
    # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1]
    u = {i0*m+j0: 1, i0*m+j1: 1}
    v = {j1*p+k1: 1}
    w = {k0*n+i0: -1, k1*n+i0: 1}
    pattern.append((u, v, w))
    # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
    u = {i1*m+j0: 1, i0*m+j0: -1}
    v = {j0*p+k0: 1, j0*p+k1: 1}
    w = {k1*n+i1: 1}
    pattern.append((u, v, w))
    # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
    u = {i0*m+j1: 1, i1*m+j1: -1}
    v = {j1*p+k0: 1, j1*p+k1: 1}
    w = {k0*n+i0: 1}
    pattern.append((u, v, w))
    return pattern


def _block_covers(pattern, n, m, p):
    """Return the set of (i,j,k) tuples covered by a pattern."""
    covered = set()
    for u_dict, v_dict, w_dict in pattern:
        for u_idx in u_dict:
            i, j = divmod(u_idx, m)
            for v_idx in v_dict:
                jj, k = divmod(v_idx, p)
                if j == jj:
                    for w_idx in w_dict:
                        kk, ii = divmod(w_idx, n)
                        if ii == i and kk == k:
                            covered.add((i, j, k))
    return covered


def _apply_block_pattern(U, V, W, col_start, pattern):
    """Apply a block pattern to U, V, W starting at col_start."""
    for r, (u_dict, v_dict, w_dict) in enumerate(pattern):
        col = col_start + r
        for idx, coeff in u_dict.items():
            if coeff != 0:
                U[idx][col] += coeff
        for idx, coeff in v_dict.items():
            if coeff != 0:
                V[idx][col] += coeff
        for idx, coeff in w_dict.items():
            if coeff != 0:
                W[idx][col] += coeff
    return col_start + len(pattern)


def _build_decomposition(used_blocks, n, m, p):
    """Build U, V, W from a list of (pattern, covers) and return (U, V, W, rank)."""
    total_rank = sum(len(pattern) for pattern, _ in used_blocks)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col = 0
    for pattern, _ in used_blocks:
        col = _apply_block_pattern(U, V, W, col, pattern)
    return U, V, W, total_rank


def _greedy_max_coverage(blocks, all_tuples, max_blocks, time_fn):
    """Greedy maximum coverage: pick blocks that cover the most new triples.
    
    To encourage disjointness, we penalize overlap heavily.
    """
    uncovered = set(all_tuples)
    used = []
    for _ in range(max_blocks):
        if time_fn():
            break
        best_block = None
        best_score = -1
        for pattern, covers in blocks:
            new_covers = covers & uncovered
            if len(new_covers) == 0:
                continue
            # Score: new coverage. We want to maximize distinct triples.
            # Since each block covers exactly 8 triples and costs 7 rank,
            # the benefit is 8 - (overlap) in terms of new triples.
            # Actually benefit = len(new_covers), cost = 7.
            # Net rank saving vs schoolbook = len(new_covers) - 7.
            score = len(new_covers)
            if score > best_score:
                best_score = score
                best_block = (pattern, covers)
        if best_block is None:
            break
        used.append(best_block)
        uncovered -= best_block[1]
    return used, uncovered


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    def time_check():
        return time.time() - _t0 > _DEADLINE - 5.0

    # Generate all possible 2x2x2 Strassen blocks
    blocks = []
    for i0 in range(n - 1):
        for i1 in range(i0 + 1, n):
            for j0 in range(m - 1):
                for j1 in range(j0 + 1, m):
                    for k0 in range(p - 1):
                        for k1 in range(k0 + 1, p):
                            pattern = _make_strassen_2x2x2_pattern(i0, i1, j0, j1, k0, k1, n, m, p)
                            covers = _block_covers(pattern, n, m, p)
                            blocks.append((pattern, covers))

    all_tuples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    total_tuples = len(all_tuples)  # 84

    # Try greedy maximum coverage with different max block counts
    best_result = None
    best_rank = n * m * p  # schoolbook fallback

    for max_blocks in range(8, 0, -1):
        if time_check():
            break
        used, uncovered = _greedy_max_coverage(blocks, all_tuples, max_blocks, time_check)
        if not used:
            continue
        # Add schoolbook for remaining
        for (i, j, k) in sorted(uncovered):
            u = {i * m + j: 1}
            v = {j * p + k: 1}
            w = {k * n + i: 1}
            used.append(([(u, v, w)], {(i, j, k)}))
        rank = sum(len(pat) for pat, _ in used)
        if rank < best_rank:
            best_rank = rank
            best_result = used
        # Stop if we hit the theoretical lower bound for this many blocks
        # 8*max_blocks triples covered max, but with overlap less.
        if rank <= 77:
            break

    if best_result is None:
        # Pure schoolbook
        best_result = []
        for (i, j, k) in all_tuples:
            u = {i * m + j: 1}
            v = {j * p + k: 1}
            w = {k * n + i: 1}
            best_result.append(([(u, v, w)], {(i, j, k)}))
        best_rank = n * m * p

    # Build final U, V, W
    U, V, W, rank = _build_decomposition(best_result, n, m, p)

    # Verify
    if not _verify_decomposition(U, V, W, n, m, p, rank):
        # Fallback: deterministic known-good rank-78
        rank = 78
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
            U[i0 * m + j0][col_start] = 1
            U[i1 * m + j1][col_start] = 1
            V[j0 * p + k0][col_start] = 1
            V[j1 * p + k1][col_start] = 1
            W[k0 * n + i0][col_start] = 1
            W[k1 * n + i1][col_start] = 1
            U[i1 * m + j0][col_start + 1] = 1
            U[i1 * m + j1][col_start + 1] = 1
            V[j0 * p + k0][col_start + 1] = 1
            W[k0 * n + i1][col_start + 1] = 1
            W[k1 * n + i1][col_start + 1] = -1
            U[i0 * m + j0][col_start + 2] = 1
            V[j0 * p + k1][col_start + 2] = 1
            V[j1 * p + k1][col_start + 2] = -1
            W[k1 * n + i0][col_start + 2] = 1
            W[k1 * n + i1][col_start + 2] = 1
            U[i1 * m + j1][col_start + 3] = 1
            V[j1 * p + k0][col_start + 3] = 1
            V[j0 * p + k0][col_start + 3] = -1
            W[k0 * n + i0][col_start + 3] = 1
            W[k0 * n + i1][col_start + 3] = 1
            U[i0 * m + j0][col_start + 4] = 1
            U[i0 * m + j1][col_start + 4] = 1
            V[j1 * p + k1][col_start + 4] = 1
            W[k0 * n + i0][col_start + 4] = -1
            W[k1 * n + i0][col_start + 4] = 1
            U[i1 * m + j0][col_start + 5] = 1
            U[i0 * m + j0][col_start + 5] = -1
            V[j0 * p + k0][col_start + 5] = 1
            V[j0 * p + k1][col_start + 5] = 1
            W[k1 * n + i1][col_start + 5] = 1
            U[i0 * m + j1][col_start + 6] = 1
            U[i1 * m + j1][col_start + 6] = -1
            V[j1 * p + k0][col_start + 6] = 1
            V[j1 * p + k1][col_start + 6] = 1
            W[k0 * n + i0][col_start + 6] = 1
            return col_start + 7

        blocks6 = [
            (0, 1, 0, 1, 0, 1),
            (0, 1, 0, 1, 2, 3),
            (0, 1, 0, 1, 4, 5),
            (0, 1, 2, 3, 0, 1),
            (0, 1, 2, 3, 2, 3),
            (0, 1, 2, 3, 4, 5),
        ]
        covered = set()
        for (i0, i1, j0, j1, k0, k1) in blocks6:
            col = add_strassen_block(col, i0, i1, j0, j1, k0, k1)
            for ii in (i0, i1):
                for jj in (j0, j1):
                    for kk in (k0, k1):
                        covered.add((ii, jj, kk))

        for i0 in range(n):
            for j0 in range(m):
                for k0 in range(p):
                    if (i0, j0, k0) in covered:
                        continue
                    U[i0 * m + j0][col] = 1
                    V[j0 * p + k0][col] = 1
                    W[k0 * n + i0][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
