"""Cardinality-separated subset sums: reverse greedy + forward greedy + swap local search."""
import time
import random

def _compute_dp(arr):
    """Compute cardinality-indexed subset sum bitsets. Returns list of ints where dp[k] has bit s set iff sum s achievable with exactly k elements."""
    n = len(arr)
    dp = [0] * (n + 1)
    dp[0] = 1  # empty subset, sum 0
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    return dp

def _is_valid_dp(dp):
    """Check if dp (list of bitsets) has no cross-cardinality collisions."""
    seen = 0
    for k in range(len(dp)):
        if dp[k] & seen:
            return False
        seen |= dp[k]
    return True

def _valid_subset_sums(arr):
    """Return True if no sum appears in two different cardinalities (incl. empty subset sum 0)."""
    return _is_valid_dp(_compute_dp(arr))

def _try_add(arr, x):
    """Return True if adding x keeps the cardinality-separation property."""
    return _valid_subset_sums(arr + [x])

def _reverse_greedy(deadline):
    """
    Reverse greedy: start from maximum possible size and work downward.
    For each position from largest to smallest, try the largest available integer
    that maintains the cardinality-separation property.
    
    Strategy: Build the set by trying to add elements in decreasing order of value,
    but validate using the full DP. This favors large values which spread out subset sums.
    """
    cur = []
    cur_set = set()
    
    # Try to build a set by considering candidates from 256 down to 1
    # At each step, try to add the largest remaining candidate that keeps validity
    for x in range(256, 0, -1):
        if time.time() > deadline:
            break
        if x in cur_set:
            continue
        if _try_add(cur, x):
            cur.append(x)
            cur_set.add(x)
    
    return cur

def _reverse_greedy_with_backtrack(deadline):
    """
    More sophisticated reverse greedy with limited backtracking.
    If we get stuck, try removing the most recently added element and trying the next largest.
    """
    cur = []
    cur_set = set()
    candidates = list(range(256, 0, -1))
    idx = 0
    
    # Simple forward pass first
    for x in candidates:
        if time.time() > deadline:
            break
        if x in cur_set:
            continue
        if _try_add(cur, x):
            cur.append(x)
            cur_set.add(x)
    
    # Now try to improve by swapping: for each element in cur, try replacing with a larger unused one
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(cur) - 1, -1, -1):  # from largest to smallest position
            if time.time() > deadline:
                break
            x = cur[i]
            # Try to replace x with a larger unused value
            for new_x in range(256, x, -1):
                if time.time() > deadline:
                    break
                if new_x in cur_set:
                    continue
                test_cur = cur[:i] + [new_x] + cur[i+1:]
                if _valid_subset_sums(test_cur):
                    cur_set.discard(x)
                    cur[i] = new_x
                    cur_set.add(new_x)
                    improved = True
                    break
    
    return cur

def _grow(cur, order, deadline):
    """Repeated-pass greedy growth: scan candidates, add any that fit, until a full pass adds nothing."""
    cur = list(cur)
    cur_set = set(cur)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in order:
            if time.time() > deadline:
                break
            if x in cur_set:
                continue
            if _try_add(cur, x):
                cur.append(x)
                cur_set.add(x)
                improved = True
    return cur

def _swap_search(cur, deadline):
    """Local search: try replacing one element with two new ones (net +1) if valid. Repeat until no gain."""
    cur = list(cur)
    n_max = 256
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i, x in enumerate(cur):
            if time.time() > deadline:
                return cur
            # candidate removal
            base = cur[:i] + cur[i+1:]
            # try adding two distinct new elements a < b
            for a in range(1, n_max + 1):
                if time.time() > deadline:
                    return cur
                if a == x or a in base:
                    continue
                if not _try_add(base, a):
                    continue
                base2 = base + [a]
                for b in range(a + 1, n_max + 1):
                    if time.time() > deadline:
                        return cur
                    if b == x or b in base2:
                        continue
                    if _try_add(base2, b):
                        new_cur = base2 + [b]
                        cur = new_cur
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return cur

def construct():
    deadline = time.time() + 75  # safety margin under 90s kill
    rng = random.Random(12345)
    best = []
    
    # Phase 1: Reverse greedy constructions
    # Try multiple variants of reverse greedy
    rev1 = _reverse_greedy(deadline)
    if len(rev1) > len(best):
        best = list(rev1)
    
    rev2 = _reverse_greedy_with_backtrack(deadline)
    if len(rev2) > len(best):
        best = list(rev2)
    
    # Phase 2: Forward greedy from various bases
    candidates_all = list(range(1, 257))
    
    # Base sets to try
    bases = [
        [1, 2, 4, 8, 16, 32, 64, 128, 256],  # powers of 2 (9 elements)
        [1, 3, 7, 15, 31, 63, 127, 255],      # 2^k - 1
        [2, 5, 11, 23, 47, 95, 191],          # 2*2^k + 1 style
        [1, 2, 4, 8, 16, 32, 64, 128, 200],   # mixed
        [],                                    # pure greedy from empty
    ]
    
    # Candidate orderings
    orders = [
        candidates_all,
        list(reversed(candidates_all)),
        [i for i in range(1, 257, 2)] + [i for i in range(2, 257, 2)],
        list(range(256, 0, -1)),
    ]
    # Alternating from both ends
    lo, hi = 1, 256
    alt = []
    while lo <= hi:
        alt.append(lo)
        if lo != hi:
            alt.append(hi)
        lo += 1
        hi -= 1
    orders.append(alt)
    # A few shuffled orders for diversification
    for _ in range(3):
        o = list(candidates_all)
        rng.shuffle(o)
        orders.append(o)
    
    for base in bases:
        if time.time() > deadline:
            break
        if base and not _valid_subset_sums(base):
            continue
        for order in orders:
            if time.time() > deadline:
                break
            cur = _grow(list(base), order, deadline)
            if len(cur) > len(best):
                best = list(cur)
    
    # Phase 3: swap-based local search on the best found
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            before = len(best)
            best = _swap_search(best, deadline - 5)
            improved = len(best) > before
    
    # Phase 4: final repeated-pass extension of best with all candidates
    if time.time() < deadline - 3:
        best = _grow(best, candidates_all, deadline - 1)
    
    # Ensure return type is list of Python ints
    return [int(v) for v in best]
