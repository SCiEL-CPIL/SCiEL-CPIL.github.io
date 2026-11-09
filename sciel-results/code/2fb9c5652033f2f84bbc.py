"""Fresh design: multi-restart greedy + local search with bitset validity checks."""
import random
import time
from itertools import combinations

# Precompute subset-sum bitsets for small sets (up to ~20 elements) to speed up checks
# For a set A and target x, we need: does any subset of A of size >=2 sum to k*x for some k?
# Equivalently: for each k from 2 to |A|, is sum k*x achievable as a subset sum of A with exactly k elements?
# We can use a dictionary mapping sum -> bitmask of achievable sizes, or for each size track achievable sums.

def _validity_check(set_list):
    """Check if set_list (list of ints) satisfies the mean constraint.
    For each element a in set_list, check that a is NOT the arithmetic mean
    of any subset of >= 2 OTHER elements."""
    n = len(set_list)
    if n < 3:
        return True
    
    for i, a in enumerate(set_list):
        others = [x for j, x in enumerate(set_list) if j != i]
        m = len(others)
        
        # DP: dp[k] = set of achievable sums using exactly k elements from others
        # To save memory, use a dict: for each count k, track which sums are achievable
        # Since sums can be up to 512*47 ~ 24064, and k up to 47, this is manageable
        # Use a list of sets or a 2D approach
        # Better: for each k, maintain a set of achievable sums
        # But that's O(m * sum_range) which for m=47, sum~24000 is ~1.1M ops per element
        # With 48 elements, that's ~53M ops total - might be too slow in Python
        
        # Alternative: use bitset per size. dp[k] is a Python int where bit s is set if sum s is achievable with k elements.
        # Max sum = 512 * 47 = 24064, so we need ~24065 bits per size.
        # Python ints handle arbitrary precision, so this works.
        
        # dp[k] = bitset of achievable sums using exactly k elements
        max_sum = sum(others)
        # dp[k] is an integer; bit j is 1 if sum j is achievable with k elements
        dp = [0] * (m + 1)
        dp[0] = 1  # sum 0 with 0 elements
        
        for val in others:
            # Process in reverse to avoid using val multiple times in one iteration
            for k in range(m, 0, -1):
                # dp[k] |= dp[k-1] << val
                dp[k] |= (dp[k-1] << val)
        
        # Now check: for each k >= 2, is sum = k*a achievable?
        # i.e., is bit (k*a) set in dp[k]?
        for k in range(2, m + 1):
            target_sum = k * a
            if target_sum > max_sum:
                break
            if (dp[k] >> target_sum) & 1:
                return False
    
    return True

def _greedy_construction(order, check_fn=None):
    """Greedy: try to add each element in the given order if valid."""
    if check_fn is None:
        check_fn = _validity_check
    selected = []
    for x in order:
        candidate = selected + [x]
        if check_fn(candidate):
            selected = candidate
    return selected

def _local_search(initial_set, time_limit, rng, check_fn=None):
    """Local search: try additions, swaps to grow the set."""
    if check_fn is None:
        check_fn = _validity_check
    
    S = list(initial_set)
    S_set = set(S)
    best = len(S)
    
    # Candidates: all numbers 1..512 not in S
    all_nums = list(range(1, 513))
    
    while True:
        remaining = time_limit - (time.time() - time_limit)
        if remaining < 1.0:
            break
        
        # Try to add a random candidate
        candidates = [x for x in all_nums if x not in S_set]
        if candidates:
            # Try a few random candidates
            tries = min(len(candidates), 20)
            tried = set()
            for _ in range(tries):
                if not candidates:
                    break
                idx = rng.randrange(len(candidates))
                y = candidates[idx]
                if y in tried:
                    continue
                tried.add(y)
                new_S = S + [y]
                if check_fn(new_S):
                    S = new_S
                    S_set.add(y)
                    candidates.remove(y)
                    best = max(best, len(S))
                    # Try to add more immediately
                    continue
                else:
                    # Remove from candidates (it doesn't work with current S)
                    # Actually it might work later after other changes, so keep it
                    pass
        
        # Try swaps: remove one element, add another
        if S:
            # Try a few random swaps
            for _ in range(15):
                if time.time() - time_limit > time_limit - 1.0:
                    break
                i = rng.randrange(len(S))
                x_remove = S[i]
                candidates = [x for x in all_nums if x not in S_set]
                if not candidates:
                    break
                y_add = candidates[rng.randrange(len(candidates))]
                # Try the swap
                new_S = [v for v in S if v != x_remove] + [y_add]
                if check_fn(new_S):
                    S = new_S
                    S_set = set(S)
                    best = max(best, len(S) + 1)  # swap keeps size same, but maybe enables future adds
        
        # If we can't grow, try a different strategy: remove a random element and try to add two
        # This is more complex, skip for now
    
    return S

def construct():
    start_time = time.time()
    total_deadline = 55.0  # Safety margin for 60s limit
    
    rng = random.Random(42)
    
    best_set = []
    best_size = 0
    
    # Strategy: multiple random restarts with greedy + local search
    # Also try structured orders
    
    # First: a deterministic greedy in increasing order
    order = list(range(1, 513))
    S = _greedy_construction(order, _validity_check)
    best_set = S
    best_size = len(S)
    
    # Try some structured orders
    # Odd numbers first (avoids many 3-APs)
    odds = [x for x in range(1, 513) if x % 2 == 1]
    evens = [x for x in range(1, 513) if x % 2 == 0]
    S2 = _greedy_construction(odds + evens, _validity_check)
    if len(S2) > best_size:
        best_set = S2
        best_size = len(S2)
    
    # Now do random restarts with local search
    while time.time() - start_time < total_deadline - 5.0:
        # Random permutation
        order = list(range(1, 513))
        rng.shuffle(order)
        
        S = _greedy_construction(order, _validity_check)
        
        if len(S) <= best_size:
            # Still try local search from this starting point, but with limited time
            if time.time() - start_time < total_deadline - 3.0:
                S = _local_search(S, 2.0, rng, _validity_check)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            # Continue local search on the best
            if time.time() - start_time < total_deadline - 3.0:
                S = _local_search(best_set, 3.0, rng, _validity_check)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S
    
    # Final: try to extend best_set with more local search if time remains
    if time.time() - start_time < total_deadline - 1.0:
        remaining = total_deadline - (time.time() - start_time) - 1.0
        if remaining > 1.0:
            extended = _local_search(best_set, remaining, rng, _validity_check)
            if len(extended) > best_size:
                best_set = extended
                best_size = len(best_set)
    
    # Ensure we return a valid set
    if _validity_check(best_set):
        return best_set
    
    # Fallback: just return increasing greedy
    S = _greedy_construction(list(range(1, 513)), _validity_check)
    return S
