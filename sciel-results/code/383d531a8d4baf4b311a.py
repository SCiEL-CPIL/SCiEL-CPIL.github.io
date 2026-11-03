"""Ternary covering code K_3(8,1) construction via greedy + local search."""
import time
import itertools
import random
import sys

# Deadline: must return well before 150s wall-clock
_DEADLINE = 140.0  # seconds from start of construct call


def _hamming_dist(a, b):
    """Hamming distance between two equal-length lists."""
    return sum(1 for x, y in zip(a, b) if x != y)


def _all_neighbors(word, q, n, radius):
    """Yield all words within Hamming distance <= radius of word."""
    yield tuple(word)
    if radius >= 1:
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    w = list(word)
                    w[pos] = val
                    yield tuple(w)
    if radius >= 2:
        for pos1 in range(n):
            for pos2 in range(pos1 + 1, n):
                for v1 in range(q):
                    if v1 == word[pos1]:
                        continue
                    for v2 in range(q):
                        if v2 == word[pos2]:
                            continue
                        w = list(word)
                        w[pos1] = v1
                        w[pos2] = v2
                        yield tuple(w)


def _precompute_balls(q, n, radius):
    """Precompute the ball (set of neighbors) for every word in {0..q-1}^n.
    Returns a list of frozensets indexed by the linear index of the word.
    """
    total = q ** n
    # Map word -> index
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    balls = [None] * total
    for idx in range(total):
        w = idx_to_word(idx)
        ball = set()
        for nb in _all_neighbors(w, q, n, radius):
            ball.add(word_to_idx(nb))
        balls[idx] = frozenset(ball)
    return balls, word_to_idx, idx_to_word


def _greedy_cover(q, n, radius, balls, word_to_idx, idx_to_word, rng, time_limit):
    """Greedy set cover: repeatedly pick the codeword covering the most uncovered words."""
    total = q ** n
    # uncovered: set of indices not yet covered
    uncovered = set(range(total))
    codewords = []

    # For each candidate codeword, count how many uncovered words it covers
    # We'll maintain a count array
    cover_count = [0] * total
    for idx in range(total):
        for nb in balls[idx]:
            if nb in uncovered:
                cover_count[idx] += 1

    start = time.time()
    while uncovered:
        if time.time() - start > time_limit:
            break
        # Find the codeword with max cover_count among those that cover at least 1 uncovered
        best_idx = -1
        best_count = 0
        # Randomized tie-breaking for diversity
        candidates = [i for i in range(total) if cover_count[i] > best_count]
        if candidates:
            best_count = max(cover_count[i] for i in candidates)
            ties = [i for i in range(total) if cover_count[i] == best_count]
            best_idx = rng.choice(ties)
        else:
            # Should not happen if uncovered is non-empty
            break

        if best_idx < 0:
            break

        codewords.append(best_idx)
        # Remove covered words from uncovered
        for nb in balls[best_idx]:
            if nb in uncovered:
                uncovered.discard(nb)
                # Decrease cover_count for all codewords that also cover nb
                # This is expensive; skip for now and recompute periodically
        # Recompute cover_count periodically or after each addition (expensive)
        # For efficiency, just recompute when uncovered shrinks significantly
        if len(uncovered) % 100 == 0 or not uncovered:
            for i in range(total):
                cover_count[i] = 0
            for nb in uncovered:
                for i in range(total):
                    if nb in balls[i]:
                        cover_count[i] += 1

    return codewords, uncovered


def _local_search(codewords, q, n, radius, balls, word_to_idx, idx_to_word, rng, time_limit):
    """Local search: try to remove a codeword if the set still covers everything,
    or swap one codeword for another to improve coverage."""
    total = q ** n
    start = time.time()
    improved = True
    while improved:
        improved = False
        if time.time() - start > time_limit:
            break

        # Compute coverage: for each word, how many codewords cover it
        coverage = [0] * total
        cw_set = set(codewords)
        for cw in codewords:
            for nb in balls[cw]:
                coverage[nb] += 1

        # Try removing each codeword
        for i in range(len(codewords)):
            if time.time() - start > time_limit:
                break
            cw = codewords[i]
            # Check if removing cw leaves any word uncovered
            can_remove = True
            for nb in balls[cw]:
                if coverage[nb] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove it
                for nb in balls[cw]:
                    coverage[nb] -= 1
                codewords.pop(i)
                improved = True
                break

        if not improved:
            # Try swaps: remove one, add another
            for i in range(len(codewords)):
                if time.time() - start > time_limit:
                    break
                cw = codewords[i]
                # Words that would become uncovered if we remove cw
                at_risk = [nb for nb in balls[cw] if coverage[nb] == 1]
                if not at_risk:
                    continue
                # Find a codeword that covers all at_risk words
                best_new = -1
                for j in range(total):
                    if j in cw_set:
                        continue
                    covers_all = True
                    for nb in at_risk:
                        if nb not in balls[j]:
                            covers_all = False
                            break
                    if covers_all:
                        # Check if adding j and removing i reduces total or keeps same
                        # Since we're removing 1 and adding 1, size stays same
                        # But maybe it enables further removals later
                        best_new = j
                        break
                if best_new >= 0:
                    # Perform swap
                    for nb in balls[cw]:
                        coverage[nb] -= 1
                    for nb in balls[best_new]:
                        coverage[nb] += 1
                    codewords[i] = best_new
                    improved = True
                    break

    return codewords


def _verify_coverage(codewords, q, n, radius, balls, word_to_idx):
    """Check that every word is within radius of some codeword."""
    total = q ** n
    covered = [False] * total
    for cw in codewords:
        for nb in balls[cw]:
            covered[nb] = True
    return all(covered)


def construct(q, n, radius):
    start_time = time.time()
    rng = random.Random(42)

    # For the specific case q=3, n=8, radius=1, we can use the known structure
    # But let's do a general greedy + local search approach

    if q ** n > 1000000:
        # Too large for full ball precomputation; use direct distance checks
        return _construct_large(q, n, radius, start_time, rng)

    balls, word_to_idx, idx_to_word = _precompute_balls(q, n, radius)
    total = q ** n

    # Initial solution: direct sum construction (same as seed)
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                w = h + list(tail)
                initial.append(word_to_idx(w))
    else:
        # Fallback: greedy from scratch
        initial, _ = _greedy_cover(q, n, radius, balls, word_to_idx, idx_to_word, rng, 30)

    # Ensure the initial solution is valid
    if not _verify_coverage(initial, q, n, radius, balls, word_to_idx):
        # Add more codewords to fix gaps
        uncovered = set()
        covered = [False] * total
        for cw in initial:
            for nb in balls[cw]:
                covered[nb] = True
        for i in range(total):
            if not covered[i]:
                uncovered.add(i)
        # Greedily add to cover remaining
        while uncovered and time.time() - start_time < _DEADLINE - 10:
            best_cw = -1
            best_count = 0
            for i in range(total):
                cnt = sum(1 for nb in balls[i] if nb in uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_cw = i
            if best_cw < 0 or best_count == 0:
                break
            initial.append(best_cw)
            for nb in balls[best_cw]:
                if nb in uncovered:
                    uncovered.discard(nb)

    best_solution = list(initial)
    best_size = len(best_solution)

    # Local search to reduce size
    remaining_time = _DEADLINE - 10 - (time.time() - start_time)
    if remaining_time > 5:
        improved = _local_search(list(initial), q, n, radius, balls, word_to_idx, idx_to_word, rng, remaining_time)
        if len(improved) < best_size and _verify_coverage(improved, q, n, radius, balls, word_to_idx):
            best_solution = improved
            best_size = len(improved)

    # Multi-start: try different random seeds for greedy
    while time.time() - start_time < _DEADLINE - 15:
        rng2 = random.Random(rng.randint(0, 2**31))
        greedy_sol, _ = _greedy_cover(q, n, radius, balls, word_to_idx, idx_to_word, rng2, 15)
        if len(greedy_sol) < best_size:
            ls_sol = _local_search(list(greedy_sol), q, n, radius, balls, word_to_idx, idx_to_word, rng2, 10)
            if _verify_coverage(ls_sol, q, n, radius, balls, word_to_idx):
                if len(ls_sol) < best_size:
                    best_solution = ls_sol
                    best_size = len(ls_sol)
        elif len(greedy_sol) == best_size:
            # Try local search even on same-size solutions for potential improvement
            ls_sol = _local_search(list(greedy_sol), q, n, radius, balls, word_to_idx, idx_to_word, rng2, 8)
            if _verify_coverage(ls_sol, q, n, radius, balls, word_to_idx) and len(ls_sol) < best_size:
                best_solution = ls_sol
                best_size = len(ls_sol)

    # Convert indices back to words
    result = [idx_to_word(i) for i in best_solution]

    # Final safety check
    if not _verify_coverage(best_solution, q, n, radius, balls, word_to_idx):
        # Fall back to the direct sum construction which is guaranteed valid
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            result = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Just return what we have (may be invalid, but better than nothing)
            pass

    return result


def _construct_large(q, n, radius, start_time, rng):
    """Fallback for large spaces: use the direct sum construction."""
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    # General fallback: random greedy
    codewords = []
    covered = set()
    total = q ** n
    while time.time() - start_time < _DEADLINE - 5:
        # Pick a random uncovered word as center
        candidates = list(range(total))
        rng.shuffle(candidates)
        best = None
        best_cover = 0
        for idx in candidates[:200]:  # Check a sample
            w = [0] * n
            tmp = idx
            for i in range(n - 1, -1, -1):
                w[i] = tmp % q
                tmp //= q
            # Count new coverage
            new_cov = 0
            for nb in _all_neighbors(tuple(w), q, n, radius):
                if nb not in covered:
                    new_cov += 1
            if new_cov > best_cover:
                best_cover = new_cov
                best = w
        if best is None or best_cover == 0:
            break
        codewords.append(best)
        for nb in _all_neighbors(tuple(best), q, n, radius):
            covered.add(nb)
    return codewords
