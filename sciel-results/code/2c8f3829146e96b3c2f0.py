"""Rewritten program: exact cardinality-indexed subset-sum DP to maximize set size."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 256, 'maximum_size': 60}

def _subset_sums_by_cardinality(elements):
    """
    Given a tuple/list of distinct positive ints, return a list of sets where
    index k is the set of all subset sums of cardinality k.
    DP: dp[k] is a set of sums.
    """
    n = len(elements)
    dp = [set() for _ in range(n + 1)]
    dp[0].add(0)
    for x in elements:
        # iterate backwards to avoid using x twice in same cardinality level
        for k in range(n - 1, -1, -1):
            if dp[k]:
                for s in dp[k]:
                    dp[k + 1].add(s + x)
    return dp

def _is_valid(elements):
    """
    Check that for any two different cardinalities k1 != k2,
    the sets of subset sums are disjoint.
    """
    dps = _subset_sums_by_cardinality(elements)
    n = len(elements)
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if dps[i] & dps[j]:
                return False
    return True

def _try_extend(current, candidates, max_size):
    """
    Greedy extension: try to add candidates one by one if valid.
    Returns the extended list.
    """
    result = list(current)
    for c in candidates:
        if len(result) >= max_size:
            break
        if c in result:
            continue
        new_result = result + [c]
        if _is_valid(new_result):
            result = new_result
    return result

def construct():
    deadline = time.time() + 80  # 80s safety margin
    best_set = None
    best_size = 0

    # Strategy 1: Arithmetic progression starting from various bases
    # We know from observations that tight APs work well.
    # Let's try APs with small differences.
    
    # First, try a deterministic search for good APs
    # An AP a, a+d, a+2d, ..., a+(n-1)d
    # The subset sums for cardinality k range from k*a + d*k*(k-1)/2 to k*a + d*k*(2n-k-1)/2
    # For two cardinalities k1 < k2, we need max sum for k1 < min sum for k2
    # max sum for k1 = k1*a + d*k1*(2n-k1-1)/2
    # min sum for k2 = k2*a + d*k2*(k2-1)/2
    # We need: k1*a + d*k1*(2n-k1-1)/2 < k2*a + d*k2*(k2-1)/2
    # => a*(k2-k1) > d * [k1*(2n-k1-1)/2 - k2*(k2-1)/2]
    # => a > d * [k1*(2n-k1-1) - k2*(k2-1)] / (2*(k2-k1))
    
    # Let's just brute force small d and a to find the largest n that works.
    
    def max_ap_size(a, d, max_n=60):
        """Find max n such that AP a, a+d, ..., a+(n-1)d is valid."""
        if d <= 0 or a < 1:
            return 0
        if a + (max_n-1)*d > 256:
            max_n = (256 - a) // d + 1
            if max_n < 1:
                return 0
        # Binary search on n
        lo, hi = 1, max_n
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            elements = tuple(a + i*d for i in range(mid))
            if elements[-1] > 256:
                hi = mid - 1
                continue
            if _is_valid(elements):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best
    
    # Try small d values
    for d in range(1, 10):
        for a in range(1, 257):
            if a + 59*d > 256:
                continue
            n = max_ap_size(a, d)
            if n > best_size:
                best_size = n
                best_set = tuple(a + i*d for i in range(n))
            if time.time() > deadline:
                return list(best_set) if best_set else [1]
    
    # Strategy 2: Randomized search with local improvement
    if time.time() < deadline - 10:
        rng = random.Random(42)
        for _ in range(1000):
            if time.time() > deadline:
                break
            # Generate a random set
            size = rng.randint(30, 60)
            elements = tuple(sorted(rng.sample(range(1, 257), size)))
            if _is_valid(elements):
                if len(elements) > best_size:
                    best_size = len(elements)
                    best_set = elements
                # Try to extend
                candidates = [x for x in range(1, 257) if x not in elements]
                rng.shuffle(candidates)
                extended = _try_extend(elements, candidates, 60)
                if len(extended) > best_size:
                    best_size = len(extended)
                    best_set = tuple(extended)
    
    # Strategy 3: Try to extend the best set found
    if best_set is not None and time.time() < deadline - 5:
        candidates = [x for x in range(1, 257) if x not in best_set]
        rng = random.Random(123)
        rng.shuffle(candidates)
        extended = _try_extend(list(best_set), candidates, 60)
        if len(extended) > best_size:
            best_size = len(extended)
            best_set = tuple(extended)
    
    if best_set is None:
        return [1]
    return list(best_set)
# EVOLVE-BLOCK-END
