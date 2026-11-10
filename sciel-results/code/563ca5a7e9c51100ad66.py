"""Improved program: Randomized search for a valid 48-element subset from the Cantor set."""
import random

def _element_is_mean(a, S):
    """Return True if a is the exact arithmetic mean of some subset of S\\{a} of size >= 2.
    Uses a per-count bitset DP (Python big ints) over subset sums."""
    others = [x for x in S if x != a]
    n = len(others)
    if n < 2:
        return False
    max_sum = sum(others)
    mask = (1 << (max_sum + 1)) - 1
    dp = [0] * (n + 1)
    dp[0] = 1  # bit 0 set: empty subset sum 0
    for e in others:
        for c in range(n - 1, -1, -1):
            if dp[c]:
                dp[c + 1] |= (dp[c] << e) & mask
    for c in range(2, n + 1):
        target = c * a
        if target <= max_sum and ((dp[c] >> target) & 1):
            return True
    return False

def _is_fully_valid(S):
    for a in S:
        if _element_is_mean(a, S):
            return False
    return True

def _generate_cantor_set(n_max):
    """Generate all numbers <= n_max with base-3 digits in {0, 1}."""
    nums = []
    max_power = 0
    while (3 ** max_power) <= n_max:
        max_power += 1
    
    for mask in range(1 << max_power):
        val = 0
        for i in range(max_power):
            if mask & (1 << i):
                val += 3 ** i
        if val <= n_max and val > 0:
            nums.append(val)
    return sorted(nums)

def construct():
    n_max = 512
    target_size = 48
    
    cantor_nums = _generate_cantor_set(n_max)
    
    # The Cantor set has 64 elements.
    # We want to find a subset of size 48 that is fully valid.
    
    # Heuristic 1: Try the largest 48 (spread out, less likely to have small sums matching means?)
    # Heuristic 2: Try the smallest 48
    # Heuristic 3: Random subsets
    
    candidates = [
        sorted(cantor_nums[-target_size:]),
        sorted(cantor_nums[:target_size]),
    ]
    
    # Add some random samples
    rng = random.Random(42)
    for _ in range(50):
        candidates.append(sorted(rng.sample(cantor_nums, target_size)))
    
    best_set = None
    best_size = 0
    
    for S in candidates:
        if len(S) < target_size:
            continue
        if _is_fully_valid(S):
            return S
    
    # If none of the quick candidates worked, do a local search
    # Start with a random 48, and try to swap elements to fix violations
    
    # Actually, let's just do a more exhaustive random search if the above failed
    # 500 random tries
    for _ in range(500):
        S = sorted(rng.sample(cantor_nums, target_size))
        if _is_fully_valid(S):
            return S
            
    # If still failing, try to repair the largest set
    # Start with all 64, remove elements that cause violations until size 48?
    # Or start with 48, if invalid, swap a bad element with a good one from the rest.
    
    # Let's try a greedy construction from scratch using the Cantor set
    # Sort by value
    # Try to add elements one by one
    
    valid_set = set()
    for x in cantor_nums:
        new_set = valid_set | {x}
        if len(new_set) > target_size:
            # We already have 48, so we can't add more unless we remove one.
            # For simplicity, if we hit 48 and it's valid, we stop.
            # But we need to ensure the first 48 added are valid.
            pass
        
    # This greedy might get stuck.
    # Let's stick to the random search, it's likely to find one.
    # The probability of a random 48-subset being invalid is low because
    # 3-AP violations are impossible (Cantor set), and higher order violations are rare.
    
    # As a last resort, return the first 48 and repair (slow but works)
    S = set(cantor_nums[:target_size])
    # Simple repair: remove any element that is a mean.
    # This might drop below 48, but it's a valid set.
    # The task says "Maximize cardinality", but returning 40 is better than 0?
    # No, we want 48.
    
    # Let's try to fill up to 48 from the remaining
    # If repair drops it to 47, we try to add one from the rest.
    
    repaired = set(S)
    guard = 100
    while guard > 0:
        guard -= 1
        removed = False
        for a in list(repaired):
            if _element_is_mean(a, repaired):
                repaired.remove(a)
                removed = True
                break
        if not removed:
            break
            
    if len(repaired) < target_size:
        remaining = [x for x in cantor_nums if x not in repaired]
        for x in remaining:
            if len(repaired) >= target_size:
                break
            new_set = repaired | {x}
            if _is_fully_valid(new_set):
                repaired = new_set
                
    if len(repaired) >= target_size:
        return sorted(list(repaired))[:target_size]
        
    # Absolute fallback
    return sorted(list(repaired))
