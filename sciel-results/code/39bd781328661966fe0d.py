"""Rewritten program: structured search for large sets without subset averages."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

MAX_N = 512
TARGET_SIZE = 48
DEADLINE_SEC = 75.0  # safety margin under 90s wall-clock


def _is_mean_conflict(a, others):
    """Check if a is the exact arithmetic mean of some subset of others (size >= 2)."""
    n = len(others)
    if n < 2:
        return False
    # For each subset size k >= 2, check if any subset sums to k*a
    # Use dynamic programming with bitsets for speed
    # dp[s] = bitmask of subset sizes that can achieve sum s
    # But sums can be up to 512*48 = 24576, which is manageable
    max_sum = sum(others)
    if max_sum < 2 * a:
        return False
    # dp[s] is a bitmask where bit k is set if a subset of size k sums to s
    # We only care about sizes 2..n
    # Use list of bitmasks indexed by sum
    dp = [0] * (max_sum + 1)
    dp[0] = 1  # size 0, sum 0
    for x in others:
        # iterate sums in reverse to avoid using x twice
        for s in range(max_sum - x, -1, -1):
            if dp[s]:
                # shift sizes up by 1
                dp[s + x] |= (dp[s] << 1)
    # Check if for any k >= 2, bit k is set in dp[k*a]
    for k in range(2, n + 1):
        target_sum = k * a
        if target_sum <= max_sum and (dp[target_sum] >> k) & 1:
            return True
    return False


def _is_valid_set(selected):
    """Check if no element in selected is the mean of any subset of >= 2 other elements."""
    for i, a in enumerate(selected):
        others = [x for j, x in enumerate(selected) if j != i]
        if _is_mean_conflict(a, others):
            return False
    return True


def _is_valid_addition(new_elem, selected):
    """Check if adding new_elem to selected keeps the set valid.
    Need to check:
    1. new_elem is not the mean of any subset of >= 2 elements from selected
    2. No existing element becomes the mean of a subset involving new_elem
    """
    # Check 1: new_elem not mean of subset of selected
    if _is_mean_conflict(new_elem, selected):
        return False
    # Check 2: for each existing element a, check if a is mean of subset of (selected - {a}) + {new_elem}
    # This is expensive; approximate by checking subsets that include new_elem
    for i, a in enumerate(selected):
        others = [x for j, x in enumerate(selected) if j != i]
        # Check subsets of others that include new_elem
        # i.e., subset = {new_elem} + T where T is subset of others, |T| >= 1, total size >= 2
        # sum = new_elem + sum(T) = (1 + |T|) * a
        # So sum(T) = (1 + |T|)*a - new_elem
        # Let k = |T|, k >= 1, total subset size = k+1 >= 2
        # sum(T) = (k+1)*a - new_elem
        # We need to check if there exists T subset of others with |T|=k and sum(T) = (k+1)*a - new_elem
        # This is equivalent to checking if (k+1)*a - new_elem is achievable as a sum of exactly k elements from others
        # Use DP: dp[s][k] = can we get sum s with exactly k elements from others
        # But this is expensive for each a. Let's use a smarter approach.
        
        # Actually, let's just do the full check with DP for the combined set
        combined_others = others + [new_elem]
        if _is_mean_conflict(a, combined_others):
            return False
    return True


def _greedy_construct():
    """Greedy construction: add elements in order if they don't create conflicts."""
    selected = []
    for x in range(1, MAX_N + 1):
        if _is_valid_addition(x, selected):
            selected.append(x)
        if len(selected) >= TARGET_SIZE:
            break
    return selected


def _random_greedy(seed):
    """Randomized greedy: shuffle candidates, add if valid."""
    rng = random.Random(seed)
    candidates = list(range(1, MAX_N + 1))
    rng.shuffle(candidates)
    selected = []
    for x in candidates:
        if _is_valid_addition(x, selected):
            selected.append(x)
        if len(selected) >= TARGET_SIZE:
            break
    return selected


def _local_search(selected, seed, time_limit):
    """Try to improve selected by swapping elements."""
    rng = random.Random(seed)
    start = time.time()
    best = list(selected)
    best_size = len(best)
    
    # Build a set for quick lookup
    best_set = set(best)
    
    # Try removals and additions
    for _ in range(500):
        if time.time() - start > time_limit:
            break
        # Try removing a random element and adding a better one
        if len(best) < TARGET_SIZE:
            # Try adding a random candidate
            candidates = [x for x in range(1, MAX_N + 1) if x not in best_set]
            if not candidates:
                break
            x = rng.choice(candidates)
            if _is_valid_addition(x, best):
                best.append(x)
                best_set.add(x)
                best_size = len(best)
        else:
            # Try swapping
            i = rng.randrange(len(best))
            removed = best[i]
            candidates = [x for x in range(1, MAX_N + 1) if x not in best_set and x != removed]
            if not candidates:
                break
            x = rng.choice(candidates)
            # Check if removing removed and adding x is valid
            temp = [y for y in best if y != removed]
            if _is_valid_addition(x, temp):
                best = temp + [x]
                best_set = set(best)
    return best


def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction
    best = _greedy_construct()
    best_size = len(best)
    best_set = set(best)
    
    # Phase 2: Try multiple random greedy approaches
    rng = random.Random(42)
    for seed in range(20):
        if time.time() - start_time > DEADLINE_SEC - 10:
            break
        candidate = _random_greedy(seed)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
            best_set = set(best)
    
    # Phase 3: Local search to try to reach 48
    if best_size < TARGET_SIZE:
        remaining_time = DEADLINE_SEC - (time.time() - start_time)
        if remaining_time > 5:
            improved = _local_search(best, 123, min(remaining_time - 2, 30))
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
    
    # Final verification
    if not _is_valid_set(best):
        # Fallback: rebuild greedily with full verification
        best = _greedy_construct()
    
    return best
# EVOLVE-BLOCK-END
