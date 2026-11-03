import time
import itertools
import random


def construct(q, n, radius):
    start = time.time()
    deadline = start + 115.0  # 5s safety margin before 120s limit

    def time_left():
        return deadline - time.time()

    # The reference construction: direct sum of ternary tetracode (9 words of length 4)
    # with every word on the remaining n-4 coordinates.
    # This gives 9 * 3^(n-4) codewords and covers radius 1.
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        codewords = [h + list(tail) for h in tetracode
                     for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # General greedy covering code construction
        if q <= 4 and n <= 10 and radius >= 1:
            codewords = _greedy_covering(q, n, radius, deadline)
        else:
            # Fallback: brute force for small cases, otherwise empty
            if q ** n <= 100000:
                codewords = [list(x) for x in itertools.product(range(q), repeat=n)]
            else:
                codewords = []

    # Try to improve by removing redundant codewords
    if codewords and time_left() > 10:
        codewords = _try_remove_redundant(codewords, q, n, radius, deadline)

    return codewords


def _greedy_covering(q, n, radius, deadline):
    """Greedy set cover: repeatedly pick the word covering the most uncovered words."""
    # For small q^n, do exact greedy. For larger, use randomized heuristic.
    if q ** n <= 50000:
        return _greedy_exact(q, n, radius, deadline)
    else:
        return _greedy_random(q, n, radius, deadline)


def _hamming_dist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _neighbors(word, q, n, radius):
    """Generate all words within Hamming distance radius of word."""
    # For small radius, enumerate all combinations of positions to change
    if radius == 0:
        yield list(word)
        return
    if radius == 1:
        yield list(word)
        for i in range(n):
            for v in range(q):
                if v != word[i]:
                    w = list(word)
                    w[i] = v
                    yield w
        return
    # For radius >= 2, use itertools for positions and values
    # This can be expensive, so only use for small n
    if n <= 8:
        for d in range(radius + 1):
            for positions in itertools.combinations(range(n), d):
                # For each position, choose a different value
                base = list(word)
                for deltas in itertools.product(
                    [(v for v in range(q) if v != word[p]) for p in positions],
                    repeat=1
                ):
                    # This is getting complicated; use a simpler approach
                    pass
        # Fallback: generate all words and filter (expensive but correct for small space)
        for w in itertools.product(range(q), repeat=n):
            if sum(1 for a, b in zip(w, word) if a != b) <= radius:
                yield list(w)


def _greedy_exact(q, n, radius, deadline):
    """Exact greedy set cover for small spaces."""
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    n_total = len(all_words)
    
    # Precompute coverage: for each candidate word, which words it covers
    # For radius 1, this is efficient
    covered = [False] * n_total
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Build coverage lists
    cover = [set() for _ in range(n_total)]
    if radius == 1:
        for i, w in enumerate(all_words):
            cover[i].add(i)
            for j in range(n):
                for v in range(q):
                    if v != w[j]:
                        nw = list(w)
                        nw[j] = v
                        idx = word_to_idx.get(tuple(nw))
                        if idx is not None:
                            cover[i].add(idx)
    else:
        # For radius > 1, use distance check
        for i in range(n_total):
            wi = all_words[i]
            for j in range(n_total):
                if _hamming_dist(wi, all_words[j]) <= radius:
                    cover[i].add(j)
    
    codewords = []
    uncovered_count = n_total
    uncovered = set(range(n_total))
    
    while uncovered and time_left_safe(deadline):
        # Find word covering most uncovered
        best_i = -1
        best_count = 0
        for i in uncovered:
            cnt = len(cover[i] & uncovered)
            if cnt > best_count:
                best_count = cnt
                best_i = i
            if cnt == len(uncovered):
                break
        if best_i < 0:
            break
        codewords.append(all_words[best_i])
        uncovered -= cover[best_i]
    
    return codewords


def _time_left_safe(deadline):
    return (deadline - time.time()) > 0.5


def _greedy_random(q, n, radius, deadline):
    """Randomized greedy for larger spaces."""
    # Start with the direct-sum construction if applicable
    codewords = []
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        codewords = [h + list(tail) for h in tetracode
                     for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Try to add random words to improve coverage
    rng = random.Random(42)
    for _ in range(1000):
        if time_left_safe(deadline):
            rw = [rng.randint(0, q - 1) for _ in range(n)]
            codewords.append(rw)
        else:
            break
    
    return codewords


def _try_remove_redundant(codewords, q, n, radius, deadline):
    """Try to remove codewords that are not needed for coverage."""
    if not codewords:
        return codewords
    
    # Build set of all codewords for quick lookup
    cw_set = set(tuple(cw) for cw in codewords)
    
    def is_covered(word, remaining):
        """Check if word is within radius of some word in remaining."""
        for cw in remaining:
            if _hamming_dist(word, cw) <= radius:
                return True
        return False
    
    # This is expensive; only try for small codes
    if len(codewords) > 2000:
        return codewords
    
    removed = []
    remaining = list(codewords)
    new_remaining = []
    
    for i, cw in enumerate(codewords):
        if time_left_safe(deadline) and (deadline - time.time()) < 2:
            break
        # Check if cw is redundant: every word within radius of cw is also within radius of another codeword
        redundant = True
        if radius == 1:
            # Check all neighbors of cw
            neighbors = [cw]
            for j in range(n):
                for v in range(q):
                    if v != cw[j]:
                        nw = list(cw)
                        nw[j] = v
                        neighbors.append(nw)
            for nb in neighbors:
                found = False
                for other_idx, other in enumerate(codewords):
                    if other_idx == i:
                        continue
                    if _hamming_dist(nb, other) <= radius:
                        found = True
                        break
                if not found:
                    redundant = False
                    break
        else:
            redundant = False  # Skip for radius > 1
        
        if redundant:
            removed.append(cw)
        else:
            new_remaining.append(cw)
    
    if removed:
        return new_remaining
    return codewords
