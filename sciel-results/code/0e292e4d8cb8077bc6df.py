import time
import random
import itertools

# EVOLVE-BLOCK-START

def _check_decomposition(U, V, W, n, m, p, trials=8):
    """Check that the decomposition is exact for random inputs."""
    rank = len(U[0])
    for seed in range(trials):
        random.seed(seed)
        A = [[random.randint(-4, 4) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-4, 4) for _ in range(p)] for _ in range(m)]
        AB = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]

        X = [[0] * n for _ in range(rank)]
        for r in range(rank):
            for i in range(n):
                s = 0
                for j in range(m):
                    s += U[i * m + j][r] * A[i][j]
                X[r][i] = s

        Y = [[0] * p for _ in range(rank)]
        for r in range(rank):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += V[j * p + k][r] * B[j][k]
                Y[r][k] = s

        C = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += W[k * n + i][r] * X[r][i] * Y[r][k]
                C[i][k] = s

        for i in range(n):
            for k in range(p):
                if C[i][k] != AB[i][k]:
                    return False
    return True

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition."""
    U_s = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    V_s = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    W_s = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    return U_s, V_s, W_s

def _strassen_2x2x6():
    """Rank-21 baseline: 3 independent Strassen blocks over the 6 columns."""
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    U_s, V_s, W_s = _strassen_2x2x2()

    for block in range(3):
        for s in range(7):
            r = block * 7 + s
            for ij in range(4):
                U[ij][r] = U_s[s][ij]
            for j in range(2):
                for k in range(2):
                    v_idx = j * 6 + (block * 2 + k)
                    V[v_idx][r] = V_s[s][j * 2 + k]
            for k in range(2):
                for i in range(2):
                    w_idx = (block * 2 + k) * 2 + i
                    W[w_idx][r] = W_s[s][k * 2 + i]

    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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

def _round_half(x):
    """Round to nearest half-integer."""
    return round(x * 2) / 2.0

def _try_reduce_rank(base, n, m, p, deadline, start_rank=None):
    """
    Structural rank-reduction search.
    Starting from a known valid decomposition, try to lower the rank by
    dropping a rank-1 term and re-solving the affected linear system with
    exact half-integer rounding, then verifying.
    Returns the best (lowest-rank) valid decomposition found, or None.
    """
    U = [row[:] for row in base["U"]]
    V = [row[:] for row in base["V"]]
    W = [row[:] for row in base["W"]]
    current_rank = len(U[0])
    best = {"U": [row[:] for row in U], "V": [row[:] for row in V], "W": [row[:] for row in W], "rank": current_rank}

    if start_rank is None:
        start_rank = current_rank

    # Attempt to reduce from current_rank down to a floor.
    # We try removing one rank-1 term at a time and re-solving.
    for target_rank in range(current_rank - 1, max(10, start_rank - 6), -1):
        if time.time() > deadline:
            break
        success = False
        # Try removing each possible term
        for drop_r in range(current_rank):
            if time.time() > deadline:
                break
            # Build a candidate with term drop_r removed.
            # Keep all other terms; the removed term's contribution must be
            # absorbed by re-solving the remaining free entries.
            # This is a heuristic: we re-solve by least squares over the
            # remaining rank-1 terms for the entries that the dropped term
            # covered.
            cand_U = [row[:] for row in U]
            cand_V = [row[:] for row in V]
            cand_W = [row[:] for row in W]

            # Zero out the dropped column
            for i in range(len(cand_U)):
                cand_U[i][drop_r] = 0
            for i in range(len(cand_V)):
                cand_V[i][drop_r] = 0
            for i in range(len(cand_W)):
                cand_W[i][drop_r] = 0

            # Now we have target_rank = current_rank - 1 columns, one of which
            # (drop_r) is all-zero. We can drop that column to get a cleaner
            # representation, but to keep indexing simple we keep the zero column
            # and just verify. However, a zero column is wasteful; instead we
            # compact by removing the zero column.
            keep = [r for r in range(current_rank) if r != drop_r]
            n_keep = len(keep)
            cU = [[cand_U[i][r] for r in keep] for i in range(len(cand_U))]
            cV = [[cand_V[i][r] for r in keep] for i in range(len(cand_V))]
            cW = [[cand_W[i][r] for r in keep] for i in range(len(cand_W))]

            # Verify exactness of the compacted candidate.
            if _check_decomposition(cU, cV, cW, n, m, p, trials=6):
                best = {"U": cU, "V": cV, "W": cW, "rank": n_keep}
                U, V, W = cU, cV, cW
                current_rank = n_keep
                success = True
                break

        if not success:
            # Could not reduce further from this rank; stop.
            break

    if best["rank"] < start_rank:
        return best
    return None

def _local_search_improve(base, n, m, p, deadline):
    """
    Structural local search: perturb a random rank-1 term's entries (within a
    small set of half-integer values) and re-solve the rest by exact matching,
    keeping only exact valid decompositions. Attempts to find a lower-rank
    decomposition by merging terms.
    """
    U = [row[:] for row in base["U"]]
    V = [row[:] for row in base["V"]]
    W = [row[:] for row in base["W"]]
    rank = len(U[0])
    best_rank = rank

    values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

    for _ in range(200):
        if time.time() > deadline:
            break
        r = random.randrange(rank)
        # Perturb a few entries in column r of U, V, W.
        cU = [row[:] for row in U]
        cV = [row[:] for row in V]
        cW = [row[:] for row in W]
        for _ in range(3):
            cU[random.randrange(len(cU))][r] = random.choice(values)
            cV[random.randrange(len(cV))][r] = random.choice(values)
            cW[random.randrange(len(cW))][r] = random.choice(values)
        # Re-solve: for each (i,k) output, the sum over r of W[k*n+i][r]*X[r]*Y[r]
        # must equal the true product. This is a bilinear constraint; we cannot
        # trivially re-solve, so we just verify.
        if _check_decomposition(cU, cV, cW, n, m, p, trials=4):
            # Try to drop a zero or redundant column
            keep = []
            for rr in range(rank):
                if any(cU[i][rr] != 0 for i in range(len(cU))) or \
                   any(cV[i][rr] != 0 for i in range(len(cV))) or \
                   any(cW[i][rr] != 0 for i in range(len(cW))):
                    keep.append(rr)
            if len(keep) < best_rank:
                cU2 = [[cU[i][rr] for rr in keep] for i in range(len(cU))]
                cV2 = [[cV[i][rr] for rr in keep] for i in range(len(cV))]
                cW2 = [[cW[i][rr] for rr in keep] for i in range(len(cW))]
                if _check_decomposition(cU2, cV2, cW2, n, m, p, trials=6):
                    U, V, W = cU2, cV2, cW2
                    rank = len(keep)
                    best_rank = rank
    if best_rank < rank:
        return {"U": U, "V": V, "W": W, "rank": best_rank}
    return None

def construct():
    n, m, p = 2, 2, 6
    start_time = time.time()
    deadline = start_time + 140

    # Baseline: rank-21 Strassen.
    base = _strassen_2x2x6()
    if not _check_decomposition(base["U"], base["V"], base["W"], n, m, p):
        return _schoolbook(n, m, p)

    best = base
    best_rank = len(base["U"][0])

    # Structural change: rank-reduction search from the baseline.
    reduced = _try_reduce_rank(base, n, m, p, deadline)
    if reduced is not None and reduced["rank"] < best_rank:
        best = reduced
        best_rank = reduced["rank"]

    # Local search to try to improve further.
    if time.time() < deadline:
        improved = _local_search_improve(best, n, m, p, deadline)
        if improved is not None and improved["rank"] < best_rank:
            best = improved
            best_rank = improved["rank"]

    # Final verification.
    if _check_decomposition(best["U"], best["V"], best["W"], n, m, p):
        return {"U": best["U"], "V": best["V"], "W": best["W"]}

    # Fallback to a guaranteed valid decomposition.
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
