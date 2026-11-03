import time
import random
import numpy as np

def construct(q, n, radius):
    start = time.time()
    deadline = start + 120.0

    # Build the full space as integers 0..q^n-1
    total = q ** n
    # Convert an integer to a word of length n (base q, most significant digit first)
    def int_to_word(x):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = x % q
            x //= q
        return w

    # Generate all neighbors (within Hamming distance radius) of a word, as a list of ints
    # Precompute single-position variations for speed
    def neighbors_of_int(x):
        w = int_to_word(x)
        seen = {x}
        # distance 0
        # distance 1
        for pos in range(n):
            orig = w[pos]
            for v in range(q):
                if v == orig:
                    continue
                w[pos] = v
                # encode
                val = 0
                for i in range(n):
                    val = val * q + w[i]
                seen.add(val)
            w[pos] = orig
        # distance 2 if radius >= 2
        if radius >= 2:
            for pos1 in range(n):
                orig1 = w[pos1]
                for v1 in range(q):
                    if v1 == orig1:
                        continue
                    w[pos1] = v1
                    for pos2 in range(pos1 + 1, n):
                        orig2 = w[pos2]
                        for v2 in range(q):
                            if v2 == orig2:
                                continue
                            w[pos2] = v2
                            val = 0
                            for i in range(n):
                                val = val * q + w[i]
                            seen.add(val)
                        w[pos2] = orig2
                w[pos1] = orig1
        return list(seen)

    # Greedy set cover: repeatedly pick the codeword covering the most uncovered words
    # Precompute neighbors for all possible codewords (total of them) - this is q^n which is 6561
    # For q=3,n=8 that's 6561, each with up to 1 + 8*2 + 28*2 = 1+16+56 = 73 neighbors -> manageable
    all_neighbors = [None] * total
    for x in range(total):
        all_neighbors[x] = neighbors_of_int(x)

    # Greedy construction
    covered = np.zeros(total, dtype=bool)
    codewords = []
    # Work with a set of uncovered
    uncovered_count = total
    while uncovered_count > 0:
        if time.time() > deadline - 5.0:
            break
        best = -1
        best_gain = -1
        # To speed up, we can sample; but with 6561 candidates and 73 neighbors, full scan is ~475k ops per iteration
        # Greedy may need ~100 iterations -> 47M ops, might be slow in Python. Let's try.
        for x in range(total):
            gain = 0
            for nb in all_neighbors[x]:
                if not covered[nb]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best = x
        if best_gain <= 0:
            break
        codewords.append(best)
        for nb in all_neighbors[best]:
            if not covered[nb]:
                covered[nb] = True
                uncovered_count -= 1
        if time.time() > deadline - 5.0:
            break

    # Convert to list of words
    result = [int_to_word(x) for x in codewords]

    # Randomized local search: swap codeword i with candidate c to maximize uniquely covered words
    # We track coverage counts: for each word in space, how many codewords cover it
    coverage = np.zeros(total, dtype=np.int32)
    for x in codewords:
        for nb in all_neighbors[x]:
            coverage[nb] += 1

    # A word is "uniquely covered" if coverage == 1; "covered" if coverage >= 1
    # Objective to maximize: number of words with coverage >= 1 (full cover) is constant if we maintain cover.
    # Actually we want to minimize number of codewords while keeping full cover.
    # So we try to remove a codeword (swap with none) if it doesn't break coverage, OR swap to reduce redundancy.
    # Better objective: minimize sum of codewords, but we can only remove if coverage stays >=1 everywhere.
    # Alternative: minimize total "waste" = sum over codewords of (neighbors that are covered by >=2 codewords).
    # Let's define: for each codeword, its "unique contribution" = number of neighbors with coverage == 1 that it covers.
    # We want to remove codewords with 0 unique contribution.

    def coverage_counts_for(codeword_list):
        cov = np.zeros(total, dtype=np.int32)
        for x in codeword_list:
            for nb in all_neighbors[x]:
                cov[nb] += 1
        return cov

    # Local search: repeatedly try to remove a codeword that has no unique contribution, or swap.
    # We'll do: for each codeword, compute unique contribution; if 0, remove it.
    # Then try swaps: replace codeword i with a random candidate c; accept if coverage stays full and unique contributions improve.

    def is_full_cover(cov):
        return np.all(cov >= 1)

    def unique_contributions(codeword_list, cov):
        # For each codeword, count neighbors where cov == 1
        uc = [0] * len(codeword_list)
        for i, x in enumerate(codeword_list):
            for nb in all_neighbors[x]:
                if cov[nb] == 1:
                    uc[i] += 1
        return uc

    # We'll iterate:
    # 1. Remove any codeword with 0 unique contribution (recompute coverage each time for simplicity, but that's slow).
    # Instead, recompute coverage after each removal.

    # To keep it fast, we'll do a limited number of passes.
    # Use a deadline-aware loop.
    max_swaps = 200000
    rng = random.Random(42)
    # Precompute a list of all possible codeword indices for random sampling
    all_indices = list(range(total))

    # We'll maintain codewords as a list of ints.
    cw = list(codewords)
    cov = coverage_counts_for(cw)
    # Ensure full cover
    if not is_full_cover(cov):
        # Greedy didn't fully cover (deadline), try to finish greedily quickly
        while not is_full_cover(cov):
            # Find an uncovered word, pick a codeword covering it with max gain
            uncovered = np.where(cov == 0)[0]
            if len(uncovered) == 0:
                break
            # Pick a random uncovered word
            target = uncovered[rng.randrange(len(uncovered))]
            # Find all codewords covering target
            candidates = [x for x in all_indices if not covered[target] or True]
            # Actually, find codewords whose neighbors include target
            # We can precompute for each word, which codewords cover it. But that's heavy.
            # Instead, just pick a random codeword and add it; greedy-ish.
            # Better: find codeword with max gain among those covering target.
            best = -1
            best_gain = -1
            for x in all_indices:
                if target in all_neighbors[x]:
                    gain = sum(1 for nb in all_neighbors[x] if cov[nb] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best = x
            if best >= 0:
                cw.append(best)
                for nb in all_neighbors[best]:
                    cov[nb] += 1
            else:
                break
            if time.time() > deadline - 3.0:
                break

    # Now local search
    # We'll do:
    # - Removal pass: find codeword with min unique contribution; if 0, remove it.
    # - Swap pass: pick random codeword, try to replace with random candidate; accept if full cover maintained and total unique contributions decrease (or same, to allow exploration).
    # Objective: minimize number of codewords. So removal is best.
    # We'll prioritize removal, then swaps to enable future removals.

    # Compute unique contributions
    uc = unique_contributions(cw, cov)
    # Find index with 0 unique contribution
    removable = [i for i, u in enumerate(uc) if u == 0]

    # We'll do a loop with deadline
    iterations = 0
    while time.time() < deadline - 3.0:
        iterations += 1
        # Recompute uc periodically (it's O(len(cw) * 73) ~ 486*73 ~ 35k, fine)
        if iterations % 10 == 0 or not removable:
            uc = unique_contributions(cw, cov)
            removable = [i for i, u in enumerate(uc) if u == 0]

        # Try removal
        if removable:
            i = removable[rng.randrange(len(removable))]
            x = cw[i]
            # Remove
            for nb in all_neighbors[x]:
                cov[nb] -= 1
            if is_full_cover(cov):
                # Accept removal
                cw.pop(i)
                removable = []
                uc = None
                continue
            else:
                # Revert
                for nb in all_neighbors[x]:
                    cov[nb] += 1
                # Mark this as not removable for now; try swap
                removable = [i for i, u in enumerate(uc) if u == 0 and i != i] if uc else []

        # Try swap
        # Pick a random codeword to swap
        if len(cw) == 0:
            break
        i = rng.randrange(len(cw))
        old_x = cw[i]
        # Pick a random candidate
        new_x = rng.choice(all_indices)
        if new_x == old_x:
            continue
        # Check if new_x is already in cw
        if new_x in cw:
            continue
        # Tentatively swap
        # Remove old_x
        for nb in all_neighbors[old_x]:
            cov[nb] -= 1
        # Add new_x
        for nb in all_neighbors[new_x]:
            cov[nb] += 1
        if is_full_cover(cov):
            # Accept swap (keeps size same, may enable removals)
            cw[i] = new_x
            uc = None
            removable = []
        else:
            # Revert
            for nb in all_neighbors[new_x]:
                cov[nb] -= 1
            for nb in all_neighbors[old_x]:
                cov[nb] += 1

    # Final: try one more removal pass
    cov = coverage_counts_for(cw)
    uc = unique_contributions(cw, cov)
    removable = [i for i, u in enumerate(uc) if u == 0]
    # Remove all removable in order
    for i in sorted(removable, reverse=True):
        x = cw[i]
        for nb in all_neighbors[x]:
            cov[nb] -= 1
        if is_full_cover(cov):
            cw.pop(i)
        else:
            for nb in all_neighbors[x]:
                cov[nb] += 1

    # Convert back to words
    result = [int_to_word(x) for x in cw]

    # Ensure distinct
    seen = set()
    unique_result = []
    for w in result:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            unique_result.append(w)

    return unique_result
