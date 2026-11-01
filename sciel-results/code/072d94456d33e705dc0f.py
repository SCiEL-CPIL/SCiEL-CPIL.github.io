import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start = time.time()
    deadline = start + 135.0  # Leave safety margin

    space = q ** n

    def w2i(w):
        i = 0
        for v in w:
            i = i * q + v
        return i

    def i2w(i):
        w = [0] * n
        for j in range(n - 1, -1, -1):
            w[j] = i % q
            i //= q
        return w

    # Precompute neighbor index lists for every word in the space (distance <= radius).
    # For radius 1: the word itself plus all single-position changes.
    if radius >= 1:
        nbrs = []
        for i in range(space):
            w = i2w(i)
            lst = [i]
            for p in range(n):
                ov = w[p]
                for nv in range(q):
                    if nv != ov:
                        w[p] = nv
                        lst.append(w2i(w))
                w[p] = ov
            nbrs.append(lst)
    else:
        nbrs = [[i] for i in range(space)]

    # Bitset representation for fast coverage operations
    # Each codeword covers a set of words (its neighbors)
    # We'll use Python integers as bitsets for O(1) set operations
    def make_bitset(indices):
        b = 0
        for i in indices:
            b |= (1 << i)
        return b

    # Precompute bitsets for each codeword's coverage
    cover_bitsets = [make_bitset(nbrs[i]) for i in range(space)]

    # Product construction start: tetracode (length 4, size 9) x full space (length n-4).
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tails = list(itertools.product(range(q), repeat=n - 4))
    base = [h + list(t) for h in tetracode for t in tails]
    base_idx = [w2i(c) for c in base]

    def check_valid_bitset(coverage_bits):
        """Check if all words are covered using bitset"""
        return coverage_bits == (1 << space) - 1

    def greedy_bitset(idx_set, order, rng):
        """Greedy deletion using bitset coverage tracking"""
        cur = list(idx_set)
        pos = {x: k for k, x in enumerate(cur)}
        coverage = 0
        for i in cur:
            coverage |= cover_bitsets[i]

        for c in order:
            if c not in pos:
                continue
            # Check if removing c leaves any word uncovered
            # A word is covered by c only if it's in c's bitset but not in others'
            # We need to check: is there a word covered ONLY by c?
            # This means: coverage_without_c has a gap where c had coverage
            coverage_without_c = 0
            for y in cur:
                if y != c:
                    coverage_without_c |= cover_bitsets[y]

            if coverage_without_c != coverage:
                continue  # c is essential

            k = pos[c]
            del cur[k]
            del pos[c]
            coverage = coverage_without_c

        return cur

    def swap_search_bitset(cur, rng, max_iters=2000):
        """Local swap search using bitset"""
        cur_set = set(cur)
        all_idx = list(range(space))
        improved = True
        it = 0

        def compute_coverage(codewords):
            cov = 0
            for i in codewords:
                cov |= cover_bitsets[i]
            return cov

        while improved and it < max_iters:
            improved = False
            it += 1
            if time.time() > deadline - 5.0:
                break

            coverage = compute_coverage(cur)

            for c in list(cur):
                if time.time() > deadline - 5.0:
                    break

                # Try removing c
                coverage_without_c = 0
                for y in cur:
                    if y != c:
                        coverage_without_c |= cover_bitsets[y]

                if coverage_without_c != coverage:
                    continue  # c is essential

                # Try moving c to a better location
                cand = [x for x in all_idx if x not in cur_set]
                rng.shuffle(cand)

                for x in cand[:30]:
                    # Simulate: remove c, add x
                    new_cur = [y for y in cur if y != c] + [x]
                    new_coverage = 0
                    for y in new_cur:
                        new_coverage |= cover_bitsets[y]

                    if new_coverage != (1 << space) - 1:
                        continue  # Not valid

                    # Try to delete more from new_cur
                    new_cur_set = set(new_cur)
                    for y in list(new_cur):
                        # Check if y is essential in new_cur
                        cov_without_y = 0
                        for z in new_cur:
                            if z != y:
                                cov_without_y |= cover_bitsets[z]
                        if cov_without_y != new_coverage:
                            continue  # y is essential
                        new_cur.remove(y)
                        new_cur_set.discard(y)
                        # Recompute coverage after deletion
                        new_coverage = 0
                        for z in new_cur:
                            new_coverage |= cover_bitsets[z]

                    if len(new_cur) < len(cur):
                        cur = new_cur
                        cur_set = set(cur)
                        improved = True
                        break

                if improved:
                    break

        return cur

    best = None
    best_len = 10 ** 9

    # Check if base is valid
    base_coverage = 0
    for i in base_idx:
        base_coverage |= cover_bitsets[i]
    if check_valid_bitset(base_coverage):
        best = list(base_idx)
        best_len = len(base_idx)

    # Multiple randomized restarts with more attempts due to faster bitset ops
    num_restarts = 50
    for s in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        rng = random.Random(s)
        order = base_idx[:]
        rng.shuffle(order)
        res = greedy_bitset(base_idx, order, rng)
        if len(res) < best_len:
            best = res
            best_len = len(res)

        # Swap polish on the current best
        res2 = swap_search_bitset(res, rng, max_iters=1000)
        # Verify res2 is valid
        res2_coverage = 0
        for i in res2:
            res2_coverage |= cover_bitsets[i]
        if check_valid_bitset(res2_coverage) and len(res2) < best_len:
            best = res2
            best_len = len(res2)

    if best is None:
        return [i2w(i) for i in base_idx]
    return [i2w(i) for i in best]
