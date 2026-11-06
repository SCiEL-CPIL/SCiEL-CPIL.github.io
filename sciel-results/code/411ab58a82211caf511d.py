"""Find a 12-element ambient set in [0,128] minimizing the largest Sidon subset size."""
import time
import itertools
import random

_DEADLINE = 90.0
_T0 = time.time()


def _time_left():
    return _DEADLINE - 12.0 - (time.time() - _T0)


def _is_sidon(arr):
    """Check if all unordered pair sums (including diagonals) are unique."""
    n = len(arr)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in seen:
                return False
            seen.add(s)
    return True


def _max_sidon_size(arr):
    """
    Find the maximum size of a Sidon subset of arr.
    Uses backtracking with strong pruning.
    """
    n = len(arr)
    if n <= 1:
        return n
    
    best = [0]
    arr_sorted = sorted(arr)
    
    def backtrack(start, current, current_sums):
        if len(current) > best[0]:
            best[0] = len(current)
        
        # Upper bound: can we beat best even taking all remaining?
        remaining = n - start
        if len(current) + remaining <= best[0]:
            return
        
        # Try adding each remaining element
        for i in range(start, n):
            x = arr_sorted[i]
            
            # Check if x can be added to current
            ok = True
            new_sums = set()
            for y in current:
                s = x + y
                if s in current_sums:
                    ok = False
                    break
                new_sums.add(s)
            
            if ok:
                s2 = x + x
                if s2 in current_sums or s2 in new_sums:
                    ok = False
                else:
                    new_sums.add(s2)
            
            if ok:
                backtrack(i + 1, current + [x], current_sums | new_sums)
            
            # Early termination if we found max possible
            if best[0] == n:
                return
            
            # If we can't beat best even with all remaining, stop
            if len(current) + (n - i - 1) <= best[0]:
                return
    
    backtrack(0, [], set())
    return best[0]


def _generate_dense_candidates():
    """Generate candidate sets, prioritizing dense clusters."""
    candidates = []
    seen = set()
    
    def add_candidate(s):
        t = tuple(sorted(s))
        if len(t) == 12 and len(set(t)) == 12 and all(0 <= x <= 128 for x in t):
            if t not in seen:
                seen.add(t)
                candidates.append(t)
    
    # 1) Consecutive blocks - densest possible
    for start in range(0, 117):
        add_candidate(list(range(start, start + 12)))
    
    # 2) Near-consecutive with one gap
    for start in range(0, 116):
        for gap_pos in range(12):
            s = []
            for i in range(13):
                if i == gap_pos:
                    continue
                s.append(start + i)
            if len(s) == 12:
                add_candidate(s)
    
    # 3) Small value clusters
    for start in range(0, 30):
        for spread in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
            s = [start + i * spread for i in range(12)]
            if max(s) <= 128:
                add_candidate(s)
    
    # 4) Hand-crafted dense sets
    handcrafted = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15],
        [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17],
        [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16],
        [0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17],
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ]
    for s in handcrafted:
        add_candidate(s)
    
    # 5) Random dense sets
    rng = random.Random(42)
    for _ in range(500):
        start = rng.randint(0, 100)
        # Generate 12 values in a small window
        window = rng.randint(12, 40)
        values = set()
        while len(values) < 12:
            values.add(start + rng.randint(0, window))
        if max(values) <= 128:
            add_candidate(list(values))
    
    return candidates


def construct():
    best_set = None
    best_size = 10**9
    
    candidates = _generate_dense_candidates()
    
    # Sort by density (smaller range = denser = likely smaller Sidon subset)
    candidates.sort(key=lambda s: s[-1] - s[0])
    
    for cand in candidates:
        if _time_left() <= 0:
            break
        s = _max_sidon_size(cand)
        if s < best_size:
            best_size = s
            best_set = cand
        if best_size <= 4:
            break
    
    if best_set is None:
        return list(range(12))
    return list(best_set)
