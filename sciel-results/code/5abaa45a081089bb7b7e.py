"""Program for large reciprocal-subsum-free sets using optimized greedy construction."""
import time
import random
from math import lcm as _math_lcm

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


# Precompute LCM of 1..24
_L = 1
for i in range(1, 25):
    _L = _math_lcm(_L, i)

# Precompute integer reciprocals: R[x] = _L // x
_R = [0] * 25
for i in range(1, 25):
    _R[i] = _L // i


def _subset_sum_exists(target, elements):
    """
    Check if target can be formed by a non-empty subset of elements.
    elements is a list of integers (scaled reciprocals).
    target is an integer (scaled reciprocal).
    Uses meet-in-the-middle for speed.
    """
    if not elements:
        return False
    if target <= 0:
        return False

    n = len(elements)
    if n == 0:
        return False

    # If target is larger than sum of all elements, impossible
    if target > sum(elements):
        return False

    # If any single element equals target, we found a subset of size 1
    for e in elements:
        if e == target:
            return True

    # Meet in the middle
    mid = n // 2
    left = elements[:mid]
    right = elements[mid:]

    def gen_sums(arr):
        sums = [0]
        for val in arr:
            # Add val to all existing sums
            new_sums = [s + val for s in sums]
            sums.extend(new_sums)
        return sums

    left_sums = gen_sums(left)
    right_sums = set(gen_sums(right))

    # We need to check if there exists L in left_sums and R in right_sums
    # such that L + R == target and (L != 0 or R != 0)
    # Since elements are positive, sums are non-negative.
    # If L=0 and R=target, that's a valid subset (from right).
    # If L=target and R=0, that's a valid subset (from left).
    # If L>0 and R>0, that's a valid subset.
    
    # Iterate over left_sums
    for l_val in left_sums:
        if l_val > target:
            continue
        r_val = target - l_val
        if r_val in right_sums:
            # Valid if not both zero (but target > 0 ensures at least one is non-zero if sum matches)
            # Actually, if l_val=0 and r_val=0, then target=0, which we excluded.
            # So any match is valid.
            return True

    return False


def _is_valid(S):
    """
    Check if set S is reciprocal-subsum-free.
    For each element x in S, 1/x must NOT equal the sum of reciprocals of any subset of S\{x}.
    """
    if not S:
        return True
    
    for x in S:
        target = _R[x]
        others = [_R[y] for y in S if y != x]
        if _subset_sum_exists(target, others):
            return False
    return True

def _greedy_construct(order):
    """
    Construct a valid set by greedily adding elements in the given order.
    """
    S = []
    for x in order:
        candidate = S + [x]
        if _is_valid(candidate):
            S.append(x)
    return S

def construct():
    """
    Construct a large reciprocal-subsum-free subset of [1, 24].
    Uses multiple greedy strategies (ascending, descending, random) to find a larger set.
    """
    best_set = []
    
    # 1. Greedy Ascending
    s_asc = _greedy_construct(range(1, 25))
    if len(s_asc) > len(best_set):
        best_set = s_asc
        
    # 2. Greedy Descending
    s_desc = _greedy_construct(range(24, 0, -1))
    if len(s_desc) > len(best_set):
        best_set = s_desc
        
    # 3. Randomized Greedy Restarts
    # The deterministic checks are fast, so we can afford many restarts.
    rng = random.Random(42)
    max_restarts = 1000 # We have plenty of time (60s limit, each restart is <1ms)
    
    for _ in range(max_restarts):
        if _time_remaining() < 1.0:
            break
            
        order = list(range(1, 25))
        rng.shuffle(order)
        s_rand = _greedy_construct(order)
        
        if len(s_rand) > len(best_set):
            best_set = s_rand
            
            # If we found a very large set, we might want to stop early or continue?
            # Max possible is 24. If we hit 20+, it's rare.
            if len(best_set) >= 20:
                # Do a few more to ensure we didn't miss 21+
                # But 24 is impossible? 
                # Actually, just keep going if time allows.
                pass

    if not best_set:
        # Fallback to a known valid set if something went wrong
        best_set = list(range(13, 25))
        
    return best_set


if __name__ == "__main__":
    result = construct()
    print(f"Result: {result}")
    print(f"Cardinality: {len(result)}")
    print(f"Valid: {_is_valid(result)}")
