"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations, product


def _to_int(w, q):
    v = 0
    for x in w:
        v = v * q + x
    return v


def _to_word(i, q, n):
    w = [0] * n
    for j in range(n - 1, -1, -1):
        w[j] = i % q
        i //= q
    return w


def _build_neighbor_map(q, n, radius):
    """Build neighbor list for each word in {0..q-1}^n."""
    total = q ** n
    neighbor_map = []
    for wi in range(total):
        w = _to_word(wi, q, n)
        nbrs = set()
        for r in range(radius + 1):
            for pos in combinations(range(n), r):
                for vals in product(range(q), repeat=r):
                    nw = list(w)
                    for i, v in zip(pos, vals):
                        nw[i] = v
                    ni = _to_int(nw, q)
                    nbrs.add(ni)
        neighbor_map.append(nbrs)
    return neighbor_map


def _greedy_set_cover(neighbor_map, total_space, random_seed=None, deadline=None):
    """Greedy set cover using vectorized incidence matrix."""
    if random_seed is not None:
        rng = random.Random(random_seed)
    else:
        rng = random.Random()

    # Build incidence matrix: M[cw][w] = 1 if w is within radius of cw
    # For efficiency, use a sparse representation
    # Each codeword covers its neighbors
    # We'll track uncovered count per codeword

    # Initialize: each codeword covers all its neighbors
    uncovered_count = np.array([len(nbrs) for nbrs in neighbor_map], dtype=np.int32)
    counts = np.zeros(total_space, dtype=np.int32)
    codewords = []

    # Precompute neighbor lists as arrays for fast iteration
    neighbor_arrays = [list(nbrs) for nbrs in neighbor_map]
    # Reverse mapping: for each word w, which codewords cover it
    # Since distance is symmetric, reverse[cw] = neighbor_map[cw]
    # But we need: for word w, list of codewords c such that w in neighbor_map[c]
    # This is the same as neighbor_map[w] since distance is symmetric
    reverse_map = neighbor_arrays  # neighbor_map[w] gives codewords covering w

    while True:
        if deadline is not None and time.time() > deadline:
            break
        # Find codeword with max uncovered coverage
        best_cw = int(np.argmax(uncovered_count))
        if uncovered_count[best_cw] <= 0:
            break

        codewords.append(best_cw)

        # Update: for each word w in neighbor_map[best_cw] that was uncovered
        for w in neighbor_arrays[best_cw]:
            if counts[w] == 0:
                counts[w] = 1
                # Decrement uncovered_count for all codewords covering w
                for cw2 in reverse_map[w]:
                    uncovered_count[cw2] -= 1

    return codewords


def _local_search_reduce(neighbor_map, codewords, total_space, deadline=None):
    """Try to remove redundant codewords."""
    if deadline is not None and time.time() > deadline:
        return codewords

    changed = True
    while changed:
        if deadline is not None and time.time() > deadline:
            break
        changed = False
        # Compute coverage counts
        counts = np.zeros(total_space, dtype=np.int32)
        for cw in codewords:
            for w in neighbor_map[cw]:
                counts[w] += 1

        # Check all covered
        if np.min(counts) == 0:
            break

        # Try removing each codeword (from end to start)
        for i in range(len(codewords) - 1, -1, -1):
            cw = codewords[i]
            removable = True
            for w in neighbor_map[cw]:
                if counts[w] <= 1:
                    removable = False
                    break
            if removable:
                codewords.pop(i)
                changed = True
                break

    return codewords


def _local_search_swap(neighbor_map, codewords, total_space, deadline=None):
    """Try 1-exchange swaps: replace one codeword with another to maintain coverage."""
    if deadline is not None and time.time() > deadline:
        return codewords

    # Compute coverage counts
    counts = np.zeros(total_space, dtype=np.int32)
    for cw in codewords:
        for w in neighbor_map[cw]:
            counts[w] += 1

    if np.min(counts) == 0:
        return codewords

    # For each codeword, check if removing it leaves some words uncovered
    # If so, try to add a different codeword that covers those uncovered words
    # This is complex; simpler: try replacing codeword i with codeword j where j not in set
    # and the new set still covers everything

    # More efficient: identify "critical" codewords (removal causes uncovered words)
    # For non-critical codewords, just remove them (handled by reduce)
    # For critical ones, try swapping with a codeword that covers the critical words

    # Simplified approach: try all pairs (i, j) where i is in set, j not in set
    # Check if replacing i with j maintains coverage
    # This is O(n^2 * degree) which may be slow

    # Better: only try swaps for codewords whose removal causes exactly a few uncovered words
    # and find a single codeword that covers all those words

    for i in range(len(codewords)):
        if deadline is not None and time.time() > deadline:
            break
        cw_i = codewords[i]
        # Words that would become uncovered if we remove cw_i
        critical_words = [w for w in neighbor_map[cw_i] if counts[w] == 1]
        if not critical_words:
            continue  # Would be removed by reduce phase

        # Try to find a codeword j (not in current set) that covers all critical_words
        # j must be such that neighbor_map[j] contains all critical_words
        cw_set = set(codewords)
        # For each candidate j, check if it covers all critical_words
        # Efficient: for each critical word, the candidates that cover it are in neighbor_map[w]
        # Intersection of neighbor_map[w] for all critical w gives candidates
        candidates = set(neighbor_map[critical_words[0]])
        for w in critical_words[1:]:
            candidates &= set(neighbor_map[w])
            if not candidates:
                break

        candidates -= cw_set
        if candidates:
            # Pick one (could be randomized)
            new_cw = next(iter(candidates))
            # Verify: replace cw_i with new_cw and check coverage
            # Quick check: new_cw covers all critical_words (by construction)
            # And all other words were covered by >=2 or by new_cw
            # Actually need to verify no word becomes uncovered
            # Words covered only by cw_i are critical_words, all covered by new_cw
            # Words covered by cw_i and others remain covered
            # So the swap is valid
            codewords[i] = new_cw
            # Update counts
            for w in neighbor_map[cw_i]:
                counts[w] -= 1
            for w in neighbor_map[new_cw]:
                counts[w] += 1

    return codewords


def _local_search_iterate(neighbor_map, codewords, total_space, deadline=None):
    """Iterate reduce and swap until no improvement."""
    if deadline is not None and time.time() > deadline:
        return codewords

    improved = True
    while improved:
        if deadline is not None and time.time() > deadline:
            break
        before = len(codewords)
        codewords = _local_search_reduce(neighbor_map, codewords, total_space, deadline)
        codewords = _local_search_swap(neighbor_map, codewords, total_space, deadline)
        improved = len(codewords) < before

    return codewords


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    total_space = q ** n

    # Build neighbor map (this is the expensive part, but only done once)
    neighbor_map = _build_neighbor_map(q, n, radius)

    best_codewords = None
    best_len = float('inf')

    # Multiple restarts with different random seeds
    num_restarts = 5
    per_restart_deadline = deadline - 5.0  # leave 5s for final processing

    for restart in range(num_restarts):
        restart_deadline = start_time + (per_restart_deadline * (restart + 1) / num_restarts)
        if time.time() > restart_deadline - 2.0:
            break

        # Greedy with randomization: occasionally pick from top-k instead of top-1
        codewords = _greedy_set_cover(neighbor_map, total_space, random_seed=restart, deadline=restart_deadline - 3.0)

        # Local search
        codewords = _local_search_iterate(neighbor_map, codewords, total_space, deadline=restart_deadline - 1.0)

        if len(codewords) < best_len:
            best_len = len(codewords)
            best_codewords = codewords[:]

    # Final local search with remaining time
    if best_codewords is not None and time.time() < deadline - 2.0:
        best_codewords = _local_search_iterate(neighbor_map, best_codewords, total_space, deadline=deadline - 1.0)

    # Convert to word lists
    result = [_to_word(cw, q, n) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
