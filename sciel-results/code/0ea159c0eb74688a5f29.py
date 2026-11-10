"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_dissociated_reciprocal(elements):
    """
    Check if all reciprocal subset sums are distinct.
    Uses exact integer arithmetic by scaling by LCM of denominators.
    """
    if not elements:
        return True
    
    # Compute LCM of all elements
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    lcm_val = 1
    for e in elements:
        lcm_val = lcm(lcm_val, e)
    
    # Scale each element to integer: lcm_val // e
    scaled = [lcm_val // e for e in elements]
    n = len(scaled)
    
    # Meet-in-the-middle to check for collisions
    # Split into two halves
    mid = n // 2
    left = scaled[:mid]
    right = scaled[mid:]
    
    def subset_sums(arr):
        """Generate all subset sums of arr as a set."""
        sums = set()
        m = len(arr)
        for mask in range(1 << m):
            s = 0
            for i in range(m):
                if mask & (1 << i):
                    s += arr[i]
            sums.add(s)
        return sums
    
    left_sums = subset_sums(left)
    right_sums = subset_sums(right)
    
    # Check if there are any collisions:
    # A collision occurs if left_sum1 + right_sum1 == left_sum2 + right_sum2
    # for distinct (left_mask1, right_mask1) != (left_mask2, right_mask2)
    # This is equivalent to: left_sum1 - left_sum2 == right_sum2 - right_sum1
    # i.e., the difference sets intersect in a non-trivial way.
    
    # Better approach: collect all 2^n sums and check for duplicates.
    # For n up to ~20, 2^20 = 1M which is feasible.
    # For larger n, use meet-in-the-middle more carefully.
    
    # Actually, let's just compute all 2^n sums for small n,
    # and use a smarter check for larger n.
    
    if n <= 20:
        all_sums = set()
        for mask in range(1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += scaled[i]
            if s in all_sums:
                return False
            all_sums.add(s)
        return True
    else:
        # Meet-in-the-middle: check if any two distinct subsets have the same sum
        # left_sums + right_sums should all be distinct
        # Build a dict from left sum to count, then for each right sum check
        # Actually, total sum = l + r. Two pairs (l1,r1) and (l2,r2) collide iff l1+r1 = l2+r2
        # with (l1,r1) != (l2,r2).
        # Equivalent: l1 - l2 = r2 - r1.
        # Let's just enumerate all and use a set. For n=22, 2^22=4M, feasible.
        all_sums = set()
        for mask in range(1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += scaled[i]
            if s in all_sums:
                return False
            all_sums.add(s)
        return True


def _is_dissociated_fast(elements):
    """
    Fast check using integer scaling. Returns True if all subset sums are distinct.
    """
    if not elements:
        return True
    
    n = len(elements)
    if n > 24:
        # Too large to check exhaustively; use probabilistic or return False
        # For safety, return False
        return False
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    lcm_val = 1
    for e in elements:
        lcm_val = lcm(lcm_val, e)
    
    scaled = [lcm_val // e for e in elements]
    
    # Compute all subset sums
    all_sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += scaled[i]
        if s in all_sums:
            return False
        all_sums.add(s)
    return True


def _greedy_construct():
    """
    Greedy construction: start with empty set, add elements from [1..22]
    in an order that maximizes separation. Try adding each candidate and
    keep it if dissociation holds.
    """
    current = []
    candidates = list(range(1, 23))
    
    # Sort candidates by some heuristic - smaller reciprocals might be better
    # Actually, let's try adding in order of decreasing value first, then try others
    for order in [sorted(candidates, reverse=True), sorted(candidates), 
                  sorted(candidates, key=lambda x: 1/x)]:
        trial = []
        for e in order:
            if e in trial:
                continue
            trial.append(e)
            if not _is_dissociated_fast(trial):
                trial.pop()
        if len(trial) > len(current):
            current = trial[:]
    
    return current


def _local_search(initial):
    """
    Try to improve the set by swapping or adding elements.
    """
    current = set(initial)
    improved = True
    while improved:
        improved = False
        # Try adding each unused element
        for e in range(1, 23):
            if e in current:
                continue
            trial = list(current) + [e]
            if _is_dissociated_fast(trial):
                current.add(e)
                improved = True
                break
        # Try swapping: remove one, add another
        for e in list(current):
            current.remove(e)
            for f in range(1, 23):
                if f in current:
                    continue
                trial = list(current) + [f]
                if _is_dissociated_fast(trial):
                    current.add(f)
                    improved = True
                    break
            if not improved:
                current.add(e)  # put it back
    
    return sorted(current)


def construct():
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    
    # Start with greedy
    result = _greedy_construct()
    
    if time.time() > deadline:
        return result
    
    # Try local search to improve
    improved = _local_search(result)
    if len(improved) > len(result):
        result = improved
    
    if time.time() > deadline:
        return result
    
    # Try different starting points with greedy
    for start_size in range(1, 23):
        if time.time() > deadline:
            break
        # Try random-ish orderings
        import random
        rng = random.Random(42)
        for _ in range(5):
            if time.time() > deadline:
                break
            candidates = list(range(1, 23))
            rng.shuffle(candidates)
            trial = []
            for e in candidates:
                trial.append(e)
                if not _is_dissociated_fast(trial):
                    trial.pop()
            if len(trial) > len(result):
                result = trial[:]
    
    # Final local search
    if time.time() < deadline - 5:
        improved = _local_search(result)
        if len(improved) > len(result):
            result = improved
    
    return result
# EVOLVE-BLOCK-END
