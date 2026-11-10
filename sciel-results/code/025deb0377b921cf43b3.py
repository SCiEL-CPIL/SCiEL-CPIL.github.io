import time
import itertools

def _validate(arr):
    """Check that arr is a valid set: all non-empty subset sums are distinct and
    no two distinct sums divide each other."""
    n = len(arr)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in sums:
            return False
        sums.add(s)
    sl = sorted(sums)
    for i in range(len(sl)):
        a = sl[i]
        for j in range(i + 1, len(sl)):
            b = sl[j]
            if b % a == 0:
                return False
    return True

def _try_add(arr, candidate):
    """Try adding candidate to arr; return new valid arr or None."""
    if candidate in arr:
        return None
    new_arr = arr + [candidate]
    if _validate(new_arr):
        return new_arr
    return None

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s

    # Known valid size-3
    best = [31, 32, 34]
    best_len = 3

    # Strategy: systematic search for size-4, then size-5, etc.
    # Use random restarts with greedy extension.
    import random

    # Pre-compute some candidate starting triples that are valid
    valid_triples = []
    for a in range(2, 40):
        for b in range(a + 1, 60):
            for c in range(b + 1, 80):
                tri = [a, b, c]
                if _validate(tri):
                    valid_triples.append(tri)
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break

    # Also add the known one
    if best not in valid_triples:
        valid_triples.append(best)

    # For each valid triple, try to extend to size 4, 5, ...
    candidates_pool = list(range(1, 257))

    for tri in valid_triples:
        if time.time() > deadline - 3:
            break
        current = list(tri)
        # Greedy extend: try to add elements from pool
        for _ in range(100):  # max 100 extension attempts per triple
            if time.time() > deadline - 3:
                break
            found = False
            # Try random candidates
            for _trial in range(200):
                cand = random.randint(1, 256)
                new_arr = _try_add(current, cand)
                if new_arr is not None:
                    current = new_arr
                    current.sort()
                    found = True
                    if len(current) > best_len:
                        best = current[:]
                        best_len = len(current)
                    break
            if not found:
                break

    # Also try a broader random search
    for trial in range(500):
        if time.time() > deadline - 2:
            break
        random.seed(trial * 7919 + 1)
        # Start with random valid triple
        tri = random.sample(range(5, 100), 3)
        tri.sort()
        if not _validate(tri):
            # Try a few more
            found_valid = False
            for _ in range(20):
                tri = random.sample(range(5, 100), 3)
                tri.sort()
                if _validate(tri):
                    found_valid = True
                    break
            if not found_valid:
                continue

        current = list(tri)
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            found = False
            for _trial in range(150):
                cand = random.randint(1, 256)
                new_arr = _try_add(current, cand)
                if new_arr is not None:
                    current = new_arr
                    current.sort()
                    found = True
                    if len(current) > best_len:
                        best = current[:]
                        best_len = len(current)
                    break
            if not found:
                break

    # Final: try to extend the best set found
    if time.time() < deadline - 1:
        current = list(best)
        for _ in range(200):
            if time.time() > deadline - 1:
                break
            found = False
            for _trial in range(300):
                cand = random.randint(1, 256)
                new_arr = _try_add(current, cand)
                if new_arr is not None:
                    current = new_arr
                    current.sort()
                    found = True
                    if len(current) > best_len:
                        best = current[:]
                        best_len = len(current)
                    break
            if not found:
                break

    # Safety: validate final result
    if not _validate(best):
        best = [31, 32, 34]

    return [int(x) for x in best]
