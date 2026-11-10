import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 55  # 55s safety margin before 60s kill
    
    denominators = list(range(2, 49))  # 47 denominators
    n = len(denominators)
    
    # Compute LCM
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
    
    # weights: weight of each denominator = L // d
    weights = [L // d for d in denominators]
    target = L  # We want sum < L (strictly below 1)
    
    # Meet-in-the-middle: split into two halves
    mid = 24  # 24 and 23 items
    left_weights = weights[:mid]   # 24 items
    right_weights = weights[mid:]  # 23 items
    
    # Generate all subset sums for the right half (23 items -> 2^23 = 8.4M)
    # Store as (sum, bitmask) pairs
    def gen_sums_with_mask(weights, deadline):
        """Generate all subset sums with bitmasks for a list of weights."""
        sums = [(0, 0)]
        for k, w in enumerate(weights):
            current_len = len(sums)
            for i in range(current_len):
                s, m = sums[i]
                ns = s + w
                if ns < target:  # Only keep sums strictly below target
                    sums.append((ns, m | (1 << k)))
            if time.time() > deadline:
                break
        return sums
    
    # Generate sums for right half (23 items)
    right_sums = gen_sums_with_mask(right_weights, deadline)
    if time.time() > deadline:
        return greedy_fallback(denominators, L, target)
    
    # Sort right sums by value, keep track of bitmask
    right_sums.sort(key=lambda x: x[0])
    right_values = [s for s, m in right_sums]
    right_masks = [m for s, m in right_sums]
    
    # For the left half (24 items), do DFS with pruning
    # At each node, we know the current sum and which items are remaining
    # We can compute the max additional sum possible from remaining items
    # to prune branches that can't beat the current best
    
    best_sum = -1
    best_mask_left = 0
    best_mask_right = 0
    
    # Precompute suffix sums for pruning: max sum achievable from items i..end
    left_n = len(left_weights)
    suffix_max = [0] * (left_n + 1)
    for i in range(left_n - 1, -1, -1):
        suffix_max[i] = suffix_max[i + 1] + left_weights[i]
    
    # Binary search helper
    def binary_search_max_below(arr, val):
        """Find the largest element in sorted arr that is < val. Return index or -1."""
        lo, hi = 0, len(arr) - 1
        result = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if arr[mid] < val:
                result = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return result
    
    # DFS over left half items
    def dfs(idx, current_sum, mask):
        nonlocal best_sum, best_mask_left, best_mask_right
        
        if time.time() > deadline:
            return
        
        # Prune: even if we take all remaining left items, can we beat best?
        if current_sum + suffix_max[idx] + right_values[-1] < best_sum:
            return
        
        # Base case: all left items processed
        if idx == left_n:
            # Find best right sum such that current_sum + right_sum < target
            rem = target - current_sum  # right sum must be < rem
            j = binary_search_max_below(right_values, rem)
            if j >= 0:
                total = current_sum + right_values[j]
                if total > best_sum:
                    best_sum = total
                    best_mask_left = mask
                    best_mask_right = right_masks[j]
            return
        
        # Option 1: skip item idx
        dfs(idx + 1, current_sum, mask)
        
        # Option 2: include item idx
        new_sum = current_sum + left_weights[idx]
        if new_sum < target:  # Must stay below target
            dfs(idx + 1, new_sum, mask | (1 << idx))
    
    dfs(0, 0, 0)
    
    if best_sum < 0:
        return greedy_fallback(denominators, L, target)
    
    # Reconstruct result
    result = []
    # Left half indices
    for bit in range(left_n):
        if best_mask_left & (1 << bit):
            result.append(bit)
    # Right half indices
    for bit in range(len(right_weights)):
        if best_mask_right & (1 << bit):
            result.append(mid + bit)
    
    return sorted(result)


def greedy_fallback(denominators, L, target):
    """Greedy fallback: add items in order of decreasing weight (increasing denominator),
    then try to improve with local search."""
    selected = []
    current_sum = 0
    
    # Sort by weight descending (denominator ascending)
    for d in sorted(denominators):
        w = L // d
        if current_sum + w < target:
            selected.append(d)
            current_sum += w
    
    # Local improvement: try swapping in/out items
    selected_set = set(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding an unselected item
        for d in denominators:
            if d not in selected_set:
                w = L // d
                if current_sum + w < target:
                    selected.append(d)
                    selected_set.add(d)
                    current_sum += w
                    improved = True
                    break
        # Try removing and re-adding
        for d in list(selected):
            w = L // d
            new_sum = current_sum - w
            # Try adding a different item
            for d2 in denominators:
                if d2 not in selected_set and d2 != d:
                    w2 = L // d2
                    if new_sum + w2 < target and new_sum + w2 > current_sum:
                        selected.remove(d)
                        selected_set.discard(d)
                        selected.append(d2)
                        selected_set.add(d2)
                        current_sum = new_sum + w2
                        improved = True
                        break
            if improved:
                break
    
    result = [denominators.index(d) for d in selected]
    return result
