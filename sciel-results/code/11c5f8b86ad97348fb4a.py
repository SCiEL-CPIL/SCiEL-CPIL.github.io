"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import random

# EVOLVE-BLOCK-START

def _is_sidon4_fast(a, b, c, d):
    """Check if 4-element set {a,b,c,d} (sorted) has all pair sums unique."""
    if (2*a == b+c or 2*a == b+d or 2*a == c+d or
        a+b == 2*c or a+b == c+d or a+b == 2*d or
        a+c == 2*b or a+c == b+d or a+c == 2*d or
        a+d == 2*b or a+d == b+c or a+d == 2*c or
        2*b == c+d or b+c == 2*d or b+d == 2*c):
        return False
    return True

def _has_sidon4(arr):
    """Check if any 4-subset of arr is Sidon. Returns True if exists."""
    n = len(arr)
    for i in range(n - 3):
        a = arr[i]
        for j in range(i + 1, n - 2):
            b = arr[j]
            for k in range(j + 1, n - 1):
                c = arr[k]
                for l in range(k + 1, n):
                    d = arr[l]
                    if _is_sidon4_fast(a, b, c, d):
                        return True
    return False

def _max_sidon_size(candidates):
    """Find the largest Sidon subset size of the given 12-element set."""
    for size in range(12, 0, -1):
        for combo in itertools.combinations(candidates, size):
            n = len(combo)
            sums = set()
            ok = True
            for i in range(n):
                for j in range(i, n):
                    s = combo[i] + combo[j]
                    if s in sums:
                        ok = False
                        break
                    sums.add(s)
                if not ok:
                    break
            if ok:
                return size
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    best_set = list(range(12))
    best_size = _max_sidon_size(best_set)  # should be 4

    # Strategy 1: Systematically try all small consecutive ranges
    # range(12) gives max Sidon 4. Check range(13), range(14), etc.
    for end in range(13, 30):
        if time.time() > deadline:
            break
        candidates = list(range(end))
        # Only use first 12 or all if exactly 12
        if len(candidates) == 12:
            pass
        elif len(candidates) > 12:
            # Try all 12-subsets of this range? Too many. Just check if the full range has no Sidon-4
            # Actually we need exactly 12 elements. Try different 12-subsets of [0..end-1]
            # For now, check if [0..end-1] contains any Sidon-4 at all
            if not _has_sidon4(candidates):
                # Every 12-subset would have max Sidon <= 3
                # Pick the first 12
                sub = candidates[:12]
                ms = _max_sidon_size(sub)
                if ms < best_size:
                    best_size = ms
                    best_set = sub
                    if best_size <= 3:
                        return best_set
        else:
            continue

    # Strategy 2: Try all consecutive 12-blocks in [0,128]
    for start in range(0, 117):
        if time.time() > deadline:
            break
        candidates = list(range(start, start + 12))
        if not _has_sidon4(candidates):
            ms = _max_sidon_size(candidates)
            if ms < best_size:
                best_size = ms
                best_set = candidates
                if best_size <= 3:
                    return best_set

    # Strategy 3: Arithmetic progressions
    for d in range(1, 12):
        for a in range(0, 129 - 11 * d):
            if time.time() > deadline:
                break
            candidates = [a + i * d for i in range(12)]
            if not _has_sidon4(candidates):
                ms = _max_sidon_size(candidates)
                if ms < best_size:
                    best_size = ms
                    best_set = candidates
                    if best_size <= 3:
                        return best_set

    # Strategy 4: Dense random search biased toward small ranges
    random.seed(42)
    for attempt in range(200000):
        if time.time() > deadline:
            break
        # Bias heavily toward small ranges
        r = random.random()
        if r < 0.3:
            max_val = random.randint(12, 16)
        elif r < 0.5:
            max_val = random.randint(17, 22)
        elif r < 0.7:
            max_val = random.randint(23, 35)
        elif r < 0.85:
            max_val = random.randint(36, 60)
        else:
            max_val = random.randint(61, 128)
        candidates = sorted(random.sample(range(max_val + 1), 12))
        if not _has_sidon4(candidates):
            ms = _max_sidon_size(candidates)
            if ms < best_size:
                best_size = ms
                best_set = candidates
                if best_size <= 3:
                    return best_set

    # Strategy 5: Local search with aggressive small-range bias
    if best_size > 3:
        current_set = best_set[:]
        for iteration in range(50000):
            if time.time() > deadline:
                break
            idx = random.randint(0, 11)
            found = False
            for _ in range(40):
                if time.time() > deadline:
                    break
                # Bias toward small values
                if random.random() < 0.5:
                    new_val = random.randint(0, 20)
                else:
                    new_val = random.randint(0, 128)
                if new_val in current_set:
                    continue
                neighbor = current_set[:]
                neighbor[idx] = new_val
                neighbor.sort()
                if not _has_sidon4(neighbor):
                    ms = _max_sidon_size(neighbor)
                    if ms < best_size:
                        best_size = ms
                        best_set = neighbor[:]
                        if best_size <= 3:
                            return best_set
                    current_set = neighbor
                    found = True
                    break
            if not found:
                # Try swapping two elements
                idx2 = random.randint(0, 11)
                if idx2 != idx:
                    neighbor = current_set[:]
                    neighbor[idx], neighbor[idx2] = neighbor[idx2], neighbor[idx]
                    neighbor.sort()
                    if not _has_sidon4(neighbor):
                        ms = _max_sidon_size(neighbor)
                        if ms < best_size:
                            best_size = ms
                            best_set = neighbor[:]
                            if best_size <= 3:
                                return best_set
                        current_set = neighbor

    # Strategy 6: Structured candidates with many additive relations
    structured_candidates = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13],
    ]
    for cand in structured_candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) == 12 and max(cand) <= 128:
            if not _has_sidon4(cand):
                ms = _max_sidon_size(cand)
                if ms < best_size:
                    best_size = ms
                    best_set = cand[:]
                    if best_size <= 3:
                        return best_set

    # Strategy 7: Exhaustive search over 12-subsets of small ranges
    # For range [0, n-1], try all C(n,12) subsets if n is small enough
    for n in range(13, 22):
        if time.time() > deadline:
            break
        full_range = list(range(n))
        # C(n,12) combinations
        for combo in itertools.combinations(full_range, 12):
            if time.time() > deadline:
                break
            if not _has_sidon4(combo):
                ms = _max_sidon_size(combo)
                if ms < best_size:
                    best_size = ms
                    best_set = list(combo)
                    if best_size <= 3:
                        return best_set

    return best_set

# EVOLVE-BLOCK-END
