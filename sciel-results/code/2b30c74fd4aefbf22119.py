# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _is_valid_fast(set_vals):
    """Fast check: no element is the mean of any subset of >=2 other elements."""
    n = len(set_vals)
    if n < 3:
        return True
    arr = np.array(set_vals, dtype=np.int64)
    max_val = int(arr.max())
    # We need to check for each element a, and for each k >= 2:
    # Is there a subset of (n-1) other elements of size k with sum = k*a?
    # Use DP with numpy bitsets.
    # dp[k] is a bitset (as int or numpy array) where bit s is set if we can pick k elements summing to s.
    
    # Max sum we need to track: for k elements, max sum is k * max_val.
    # But we only care about sums up to k * max_val for the target k*a.
    # Actually, the maximum target is n * max_val (when a = max_val and k = n-1... wait)
    # For element a, we check subsets of the OTHER n-1 elements.
    # For subset size k (2 <= k <= n-1), target sum = k * a.
    # Max possible target: (n-1) * max_val.
    
    # Let's use a list of Python ints as bitsets for flexibility, but with numpy for speed where possible.
    # Actually, Python ints with bit operations are quite fast for bitsets up to ~10000 bits.
    # n-1 elements, max sum = (n-1) * max_val. For n=48, max_val=512: 47*512 = 24064 bits.
    # That's manageable.
    
    for a in set_vals:
        others = [x for x in set_vals if x != a]
        m = len(others)  # m = n-1
        if m < 2:
            continue
        
        # DP: dp[k] = bitset where bit s is set if we can form sum s using k elements from others
        # We process elements one by one
        # dp[0] = 1 (bit 0)
        # For each x in others: for k from current_max down to 1: dp[k] |= dp[k-1] << x
        
        # We only need to track up to max_target = (n-1) * a for this particular a
        # Actually, for this a, we check k from 2 to m, target = k*a
        # Max target = m * a
        max_target = m * a
        
        # Use list of ints
        dp = [0] * (m + 1)
        dp[0] = 1  # bit 0 set
        
        for x in others:
            # Update in reverse
            # Only need to update k where k*a could be relevant, but we need all k for correctness
            # Actually we need all k because future elements might use any k
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << x)
                # Mask to avoid huge integers - we only care up to max_target
                # But different k have different max targets. Let's just mask to a reasonable bound.
                # Max sum of any k elements is k * max(others) <= k * max_val
                # We'll mask to max_target to keep numbers small
                if max_target < 20000:
                    dp[k] &= (1 << (max_target + 1)) - 1
        
        # Check for violations
        for k in range(2, m + 1):
            target = k * a
            if target <= max_target:
                if (dp[k] >> target) & 1:
                    return False
    
    return True

def _check_addition_fast(current_set, x):
    """Check if adding x to current_set creates any violation.
    Only new violations can involve x as the mean, or existing elements becoming means
    of subsets that include x.
    """
    new_set = current_set + [x]
    n = len(new_set)
    if n < 3:
        return True
    
    # Check 1: Is x the mean of any subset of >=2 elements from current_set?
    if len(current_set) >= 2:
        others = current_set
        m = len(others)
        # Check if x = sum(S)/len(S) for some S subset of others, |S| >= 2
        # i.e., sum(S) = x * |S|
        # DP on others
        max_target = m * x
        if max_target < 50000:
            dp = [0] * (m + 1)
            dp[0] = 1
            for elem in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << elem)
                    if max_target < 20000:
                        dp[k] &= (1 << (max_target + 1)) - 1
            for k in range(2, m + 1):
                target = k * x
                if target <= max_target and (dp[k] >> target) & 1:
                    return False
    
    # Check 2: For each a in current_set, is a the mean of some subset of new_set that includes x?
    # Subset includes x and some elements from current_set \ {a}
    # Let S = {x} ∪ T where T ⊆ current_set \ {a}
    # sum(S) / |S| = a => x + sum(T) = a * (1 + |T|)
    # => sum(T) = a * (1 + |T|) - x
    # For |T| = t >= 1 (so |S| = t+1 >= 2): sum(T) = a*(t+1) - x
    # T is a subset of current_set \ {a} of size t
    
    for a in current_set:
        others_no_a = [y for y in current_set if y != a]
        m = len(others_no_a)
        if m < 1:
            continue
        # We need to check if for some t >= 1, there exists a subset T of others_no_a of size t
        # with sum(T) = a*(t+1) - x
        # DP on others_no_a
        # Max target: a*(m+1) - x, but we need sum >= 0, so a*(t+1) > x for feasibility
        # Let's just do DP and check
        max_t = m
        max_target = a * (max_t + 1) - x
        if max_target < 0:
            continue
        if max_target > 50000:
            # Skip this check if too large (unlikely to be violated)
            continue
        dp = [0] * (m + 1)
        dp[0] = 1
        for elem in others_no_a:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << elem)
                if max_target < 20000:
                    dp[k] &= (1 << (max_target + 1)) - 1
        for t in range(1, m + 1):
            target = a * (t + 1) - x
            if 0 <= target <= max_target:
                if (dp[t] >> target) & 1:
                    return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    best_set = []
    
    # Strategy 1: Greedy from low to high
    selected = []
    for x in range(1, 513):
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if _check_addition_fast(selected, x):
            selected.append(x)
    
    if len(selected) > len(best_set):
        best_set = selected[:]
    
    # Strategy 2: Greedy from high to low
    selected = []
    for x in range(512, 0, -1):
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if _check_addition_fast(selected, x):
            selected.append(x)
    
    if len(selected) > len(best_set):
        best_set = selected[:]
    
    # Strategy 3: Randomized greedy with multiple seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        order = list(range(1, 513))
        rng.shuffle(order)
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if len(selected) >= 48:
                break
            if _check_addition_fast(selected, x):
                selected.append(x)
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # Strategy 4: Try to improve best_set by swapping
    if time.time() < deadline - 5:
        current = best_set[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                # Try replacing current[i] with some x
                for x in range(1, 513):
                    if x in current:
                        continue
                    candidate = current[:i] + [x] + current[i+1:]
                    if _is_valid_fast(candidate):
                        # Same size, but might enable further additions
                        # Check if we can add more
                        for y in range(1, 513):
                            if y in candidate:
                                continue
                            if len(candidate) >= 48:
                                break
                            if _check_addition_fast(candidate, y):
                                candidate.append(y)
                                improved = True
                                break
                        current = candidate
                        if len(current) > len(best_set):
                            best_set = current[:]
                        break
                if improved:
                    break
    
    # Strategy 5: Local search - try adding elements to best_set
    if time.time() < deadline - 3:
        current = best_set[:]
        for x in range(1, 513):
            if time.time() > deadline:
                break
            if x in current:
                continue
            if len(current) >= 48:
                break
            if _check_addition_fast(current, x):
                current.append(x)
        if len(current) > len(best_set):
            best_set = current[:]
    
    return best_set[:48]
# EVOLVE-BLOCK-END
