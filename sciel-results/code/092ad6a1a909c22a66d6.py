"""Structural change: targeted-swap local search using neighbor-list intersections
of risky (uncovered) points, multi-start greedy, and restart loop."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def _word_to_idx(word, q, n):
    idx = 0
    for d in word:
        idx = idx * q + d
    return idx


def _tetracode_solution(q, n, radius):
    """Direct sum construction: 9 * q^(n-4) words (length-4 tetracode core)."""
    if q != 3 or n < 4:
        return None
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]


def _build_neighbors(q, n, radius):
    """Return list of neighbor index lists (words within Hamming distance <= radius)."""
    total = q ** n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int64)
    neighbors = [None] * total
    for idx in range(total):
        w = all_words[idx].copy()
        nbrs = [idx]
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val != orig:
                        w[pos] = val
                        nbrs.append(_word_to_idx(w, q, n))
                w[pos] = orig
        neighbors[idx] = nbrs
    # Also build a coverage matrix for fast greedy
    cov = np.zeros((total, total), dtype=np.int8)
    for i in range(total):
        for nbr in neighbors[i]:
            cov[i, nbr] = 1
    return neighbors, all_words, cov


def _verify_cover_indices(code_indices, cov, total):
    """Return True if code_indices cover the whole space."""
    covered = np.zeros(total, dtype=bool)
    for c in code_indices:
        covered[cov[c] == 1] = True
    return bool(covered.all())


def _greedy_init(cov, total, deadline, rng=None):
    """Fast vectorized greedy: pick candidate covering most uncovered words."""
    uncovered = np.ones(total, dtype=np.int8)
    code_list = []
    while True:
        if time.time() > deadline:
            break
        counts = cov.dot(uncovered)
        best_idx = int(np.argmax(counts))
        if counts[best_idx] == 0:
            break
        code_list.append(best_idx)
        uncovered[cov[best_idx] == 1] = 0
        if uncovered.sum() == 0:
            break
    return code_list


def _greedy_init_random(cov, total, deadline, rng, alpha=0.3):
    """Greedy with randomization: pick from top-k candidates with probability."""
    uncovered = np.ones(total, dtype=np.int8)
    code_list = []
    while True:
        if time.time() > deadline:
            break
        counts = cov.dot(uncovered)
        max_count = counts.max()
        if max_count == 0:
            break
        # Softmax-like: pick from candidates with count >= (1-alpha)*max_count
        threshold = (1 - alpha) * max_count
        candidates = np.where(counts >= threshold)[0]
        if len(candidates) == 0:
            candidates = np.array([int(np.argmax(counts))])
        # Weighted random pick among candidates
        weights = counts[candidates].astype(np.float64)
        probs = weights / weights.sum()
        chosen = rng.choice(candidates, p=probs)
        code_list.append(int(chosen))
        uncovered[cov[chosen] == 1] = 0
        if uncovered.sum() == 0:
            break
    return code_list


def _remove_redundant(code_list, neighbors, total, deadline):
    """Remove any codeword whose neighbors are all covered by others."""
    code_set = set(code_list)
    cover_count = np.zeros(total, dtype=np.int32)
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for c in list(code_list):
            if c not in code_set:
                continue
            removable = True
            for nbr in neighbors[c]:
                if cover_count[nbr] <= 1:
                    removable = False
                    break
            if removable:
                code_list.remove(c)
                code_set.discard(c)
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1
                changed = True
    return code_list, cover_count


def _targeted_swap(code_list, neighbors, cov, total, deadline, rng):
    """
    Structural improvement: when removing codeword c creates uncovered points,
    find the best replacement by intersecting neighbor lists of those points.
    A candidate d covers uncovered point u iff u is in neighbors[d], i.e.,
    d is in the "reverse neighbor" set of u. We precompute this via cov.
    """
    code_set = set(code_list)
    cover_count = np.zeros(total, dtype=np.int32)
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1

    best_size = len(code_list)
    best_code = list(code_list)

    # Precompute reverse coverage: for each word u, which codeword positions cover it?
    # Actually cov[i][j]=1 means candidate i covers word j.
    # For a word u, the set of candidates covering it is {i : cov[i][u]==1}.
    # We can precompute this as a list for faster lookup.
    # But that's total x total memory. Instead, use cov column access.

    while time.time() < deadline:
        # Phase 1: removal pass
        improved = False
        order = list(code_list)
        rng.shuffle(order)
        for c in order:
            if time.time() > deadline:
                break
            if c not in code_set:
                continue
            removable = True
            for nbr in neighbors[c]:
                if cover_count[nbr] <= 1:
                    removable = False
                    break
            if removable:
                code_list.remove(c)
                code_set.discard(c)
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1
                improved = True
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)

        if improved:
            continue

        # Phase 2: targeted swap
        if len(code_list) <= 403:
            break

        c = rng.choice(code_list)
        # Temporarily remove c
        for nbr in neighbors[c]:
            cover_count[nbr] -= 1
        code_list.remove(c)
        code_set.discard(c)

        # Find uncovered words (those with cover_count == 0 in neighbors[c])
        uncovered_words = []
        for nbr in neighbors[c]:
            if cover_count[nbr] == 0:
                uncovered_words.append(nbr)

        if not uncovered_words:
            # Removal was valid
            if len(code_list) < best_size:
                best_size = len(code_list)
                best_code = list(code_list)
            continue

        # TARGETED CANDIDATE SELECTION:
        # For each uncovered word u, the candidates that cover u are those i with cov[i][u]==1.
        # We want candidates that cover the MOST uncovered words.
        # Build a count array: for each candidate i, count how many uncovered words it covers.
        # Use cov rows: cov[i][u] for u in uncovered_words.
        # Efficient approach: create a mask of uncovered words, then for each candidate,
        # sum cov[i, mask]. But that's O(total * |uncovered|) which can be large.
        # Instead, use the fact that cov is sparse-ish (each row has ~1 + n*(q-1) = 25 entries for q=3,n=8,r=1).
        # So for each uncovered word u, find all i such that cov[i][u]==1, and increment a counter.

        # Build reverse index: for each word u, list of candidates covering it.
        # Precompute once outside the loop for efficiency.
        # Actually let's precompute cov_col: cov_col[u] = np.where(cov[:, u] == 1)[0]
        # This is expensive to build but done once.

        # For now, do it on the fly with a counter array
        cand_count = np.zeros(total, dtype=np.int32)
        for u in uncovered_words:
            # Find all i with cov[i][u] == 1
            # cov[:, u] is a column; use the fact that cov[i][u]=1 iff u in neighbors[i]
            # So we can iterate over neighbors and check, but that's what cov gives us.
            # Use numpy: np.where(cov[:, u] == 1)[0]
            covering = np.where(cov[:, u] == 1)[0]
            cand_count[covering] += 1

        # Exclude c itself
        cand_count[c] = 0

        # Pick the candidate with max count
        best_cand = int(np.argmax(cand_count))
        best_gain = int(cand_count[best_cand])

        if best_gain > 0:
            code_list.append(best_cand)
            code_set.add(best_cand)
            for nbr in neighbors[best_cand]:
                cover_count[nbr] += 1
            if len(code_list) < best_size:
                best_size = len(code_list)
                best_code = list(code_list)
        else:
            # Restore c
            code_list.append(c)
            code_set.add(c)
            for nbr in neighbors[c]:
                cover_count[nbr] += 1

    return best_code


def _polish(code_list, neighbors, total, deadline):
    """Deterministic removal polish."""
    code_set = set(code_list)
    cover_count = np.zeros(total, dtype=np.int32)
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1
    for _ in range(5):
        if time.time() > deadline:
            break
        changed = False
        for c in sorted(code_list):
            if c not in code_set:
                continue
            removable = True
            for nbr in neighbors[c]:
                if cover_count[nbr] <= 1:
                    removable = False
                    break
            if removable:
                code_list.remove(c)
                code_set.discard(c)
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1
                changed = True
        if not changed:
            break
    return code_list


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135.0

    # Baseline
    tc = _tetracode_solution(q, n, radius)
    if tc is not None:
        best = tc
        best_idx = [_word_to_idx(w, q, n) for w in tc]
    else:
        best = []
        best_idx = []

    total = q ** n
    neighbors, all_words, cov = _build_neighbors(q, n, radius)

    try:
        # Multi-start greedy + targeted local search
        rng = random.Random(42)
        best_found = best_idx[:]
        best_found_size = len(best_found)

        # Number of restarts based on time budget
        # Each restart: greedy (~5-10s) + local search (variable)
        # Try at least 2-3 restarts if time permits
        max_restarts = 3
        for restart in range(max_restarts):
            if time.time() > deadline - 20.0:
                break

            # Greedy init (randomized for diversity)
            greedy_deadline = min(deadline - 30.0, time.time() + 15.0)
            if restart == 0:
                code_list = _greedy_init(cov, total, greedy_deadline)
            else:
                code_list = _greedy_init_random(cov, total, greedy_deadline, rng, alpha=0.4)

            if not code_list:
                continue

            # Verify
            if not _verify_cover_indices(code_list, cov, total):
                continue

            # Removal polish first
            polish_deadline = min(deadline - 15.0, time.time() + 5.0)
            code_list, _ = _remove_redundant(code_list, neighbors, total, polish_deadline)

            # Targeted local search
            ls_deadline = deadline - 10.0
            code_list = _targeted_swap(code_list, neighbors, cov, total, ls_deadline, rng)

            # Final polish
            polish_deadline2 = deadline - 3.0
            code_list = _polish(code_list, neighbors, total, polish_deadline2)

            # Verify final
            if _verify_cover_indices(code_list, cov, total):
                if len(code_list) < best_found_size:
                    best_found_size = len(code_list)
                    best_found = list(code_list)

        # Convert best to word lists
        if best_found:
            candidate = [list(all_words[i]) for i in best_found]
            if len(candidate) < len(best):
                best = candidate
    except Exception:
        pass

    return best
# EVOLVE-BLOCK-END
