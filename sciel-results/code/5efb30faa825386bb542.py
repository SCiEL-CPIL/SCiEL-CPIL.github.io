"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _tetracode_words():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _build_base_code(q, n, radius):
    # Start from the direct-sum construction: tetracode x full space on remaining coords.
    # This gives a valid covering code with 9 * 3**(n-4) words.
    tet = _tetracode_words()
    if n == 4:
        return [w[:] for w in tet]
    tails = list(itertools.product(range(q), repeat=n - 4))
    return [h + list(t) for h in tet for t in tails]


def _hamming_dist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _build_neighborhoods(codes, q, n, radius):
    """
    For each codeword, precompute the set of all words within Hamming distance <= radius.
    Represented as a set of integer encodings.
    """
    neighborhoods = []
    for cw in codes:
        nb = set()
        # All words within distance <= radius from cw
        for d in range(radius + 1):
            for pos in itertools.combinations(range(n), d):
                for vals in itertools.product(range(q), repeat=d):
                    w = list(cw)
                    for i, v in zip(pos, vals):
                        if v == cw[i]:
                            continue
                        w[i] = v
                    # encode
                    code = 0
                    for i in range(n - 1, -1, -1):
                        code = code * q + w[i]
                    nb.add(code)
        neighborhoods.append(nb)
    return neighborhoods


def _all_words_encoded(q, n):
    """Return a set of all integer encodings of words in {0..q-1}^n."""
    total = q ** n
    return set(range(total))


def _encode(word, q):
    code = 0
    for i in range(len(word) - 1, -1, -1):
        code = code * q + word[i]
    return code


def _decode(code, q, n):
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = code % q
        code //= q
    return w


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Build initial code
    codes = _build_base_code(q, n, radius)
    best_code = codes[:]
    best_size = len(codes)

    # For small instances, we can be more aggressive
    # Try a greedy reduction: iteratively remove codewords if coverage is maintained

    # Build coverage counts: for each word in the space, how many codewords cover it?
    # But the space is 3^8 = 6561 words, which is manageable.
    total_words = q ** n
    all_words = list(range(total_words))

    # For each codeword, compute which words it covers
    def compute_coverage(cw):
        covered = set()
        for d in range(radius + 1):
            for pos in itertools.combinations(range(n), d):
                for vals in itertools.product(range(q), repeat=d):
                    w = list(cw)
                    for i, v in zip(pos, vals):
                        if v == cw[i]:
                            continue
                        w[i] = v
                    code = 0
                    for i in range(n - 1, -1, -1):
                        code = code * q + w[i]
                    covered.add(code)
        return covered

    # Compute coverage sets for all codewords
    cov_sets = [compute_coverage(cw) for cw in codes]

    # Coverage count for each word
    cov_count = np.zeros(total_words, dtype=np.int32)
    for cs in cov_sets:
        for w in cs:
            cov_count[w] += 1

    # Greedy reduction: try to remove each codeword if no word becomes uncovered
    # We do multiple passes to reach a local minimum
    removed = [False] * len(codes)
    active = [True] * len(codes)

    def try_remove(idx):
        """Try to remove codeword idx. Return True if successful."""
        if not active[idx]:
            return False
        # Check if any word in cov_sets[idx] has cov_count == 1
        for w in cov_sets[idx]:
            if cov_count[w] == 1:
                return False
        # Safe to remove
        for w in cov_sets[idx]:
            cov_count[w] -= 1
        active[idx] = False
        removed[idx] = True
        return True

    # Multiple passes of greedy removal
    pass_count = 0
    while True:
        if time.time() > deadline - 10.0:
            break
        changed = False
        for idx in range(len(codes)):
            if time.time() > deadline - 10.0:
                break
            if active[idx]:
                if try_remove(idx):
                    changed = True
        pass_count += 1
        if not changed:
            break

    # Build reduced code
    reduced_codes = [codes[i] for i in range(len(codes)) if active[i]]
    if len(reduced_codes) < best_size:
        best_code = reduced_codes[:]
        best_size = len(reduced_codes)

    # Try local search: swap codewords to potentially reduce size further
    # This is complex, so let's try a simpler approach: random restarts with greedy
    if time.time() < deadline - 20.0:
        # Try a few random perturbations
        for attempt in range(3):
            if time.time() > deadline - 10.0:
                break

            # Start from current best
            cur_codes = best_code[:]
            cur_active = [True] * len(cur_codes)

            # Recompute coverage
            cov_sets_cur = [compute_coverage(cw) for cw in cur_codes]
            cov_count_cur = np.zeros(total_words, dtype=np.int32)
            for cs in cov_sets_cur:
                for w in cs:
                    cov_count_cur[w] += 1

            # Greedy removal again
            for _ in range(5):  # up to 5 passes
                changed = False
                for idx in range(len(cur_codes)):
                    if time.time() > deadline - 10.0:
                        break
                    if cur_active[idx]:
                        # Try remove
                        can_remove = True
                        for w in cov_sets_cur[idx]:
                            if cov_count_cur[w] == 1:
                                can_remove = False
                                break
                        if can_remove:
                            for w in cov_sets_cur[idx]:
                                cov_count_cur[w] -= 1
                            cur_active[idx] = False
                            changed = True
                if not changed:
                    break

            cur_reduced = [cur_codes[i] for i in range(len(cur_codes)) if cur_active[i]]
            if len(cur_reduced) < best_size:
                best_code = cur_reduced[:]
                best_size = len(cur_reduced)

    # Final verification: ensure best_code is a valid covering code
    # For safety, if verification fails, fall back to the original direct-sum code
    if time.time() < deadline - 5.0:
        # Verify coverage
        covered_all = set()
        for cw in best_code:
            cs = compute_coverage(cw)
            covered_all.update(cs)
        if len(covered_all) < total_words:
            # Invalid, use original
            best_code = _build_base_code(q, n, radius)

    return best_code
# EVOLVE-BLOCK-END
