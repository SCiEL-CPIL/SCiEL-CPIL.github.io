"""Improved program with bitset-based subset sum for fast mean checking."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s limit
    n = 512
    target_size = 48
    
    def _build_dp(arr):
        """Build DP table: dp[k] is bitset of all subset sums of size k."""
        m = len(arr)
        if m == 0:
            return [1]
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in arr:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << val)
        return dp
    
    def _is_valid_incremental(existing, x, dp_cache=None, sum_cache=None):
        """Check if adding x to existing set keeps it valid. Uses DP for speed."""
        if not existing:
            return True
        m = len(existing)
        total_sum = sum_cache if sum_cache is not None else sum(existing)
        
        if dp_cache is None:
            dp = _build_dp(existing)
        else:
            dp = dp_cache
        
        # Check 1: x is not the mean of any subset of existing of size >= 2
        for k in range(2, m + 1):
            target = k * x
            if target > total_sum:
                break
            if (dp[k] >> target) & 1:
                return False
        
        # Check 2: no element a in existing is the mean of a subset of existing∪{x} of size >= 2 that includes x
        # Such a subset has x plus t elements from existing, total size t+1 >= 2, so t >= 1
        # Mean = a means: (sum of t elements from existing + x) / (t+1) = a
        # => sum of t elements = (t+1)*a - x
        for a in existing:
            for t in range(1, m + 1):
                target = (t + 1) * a - x
                if target < 0:
                    continue
                if target > total_sum:
                    break
                if (dp[t] >> target) & 1:
                    return False
        return True
    
    def _greedy_from_order(order, selected_start=None):
        """Try to build a valid set by greedily adding elements in the given order.
        Caches DP for the current selected set to avoid rebuilding on every check."""
        if selected_start is None:
            selected = []
        else:
            selected = list(selected_start)
        
        dp_cache = None
        sum_cache = None
        selected_changed = True
        
        for x in order:
            if len(selected) >= target_size:
                break
            if x in selected:
                continue
            if selected_changed:
                dp_cache = _build_dp(selected)
                sum_cache = sum(selected)
                selected_changed = False
            
            if _is_valid_incremental(selected, x, dp_cache, sum_cache):
                selected.append(x)
                selected_changed = True
        return selected
    
    def _try_fill(selected, order, dp_cache=None, sum_cache=None):
        """Try to fill the set to target_size by adding elements from order."""
        selected = list(selected)
        if dp_cache is None:
            dp_cache = _build_dp(selected)
            sum_cache = sum(selected)
        
        for x in order:
            if len(selected) >= target_size:
                break
            if x in selected:
                continue
            if _is_valid_incremental(selected, x, dp_cache, sum_cache):
                selected.append(x)
                dp_cache = _build_dp(selected)
                sum_cache = sum(selected)
        return selected
    
    def _local_search(selected, max_iterations=50):
        """Try to improve the set by swapping elements and filling."""
        selected = list(selected)
        if not selected:
            return selected
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            
            # Try to fill first
            if len(selected) < target_size:
                dp_cache = _build_dp(selected)
                sum_cache = sum(selected)
                all_nums = list(range(1, n + 1))
                filled = _try_fill(selected, all_nums, dp_cache, sum_cache)
                if len(filled) > len(selected):
                    selected = filled
                    improved = True
                    continue
            
            # Try swaps
            dp_cache = _build_dp(selected)
            sum_cache = sum(selected)
            for i in range(len(selected)):
                if time.time() > deadline - 5.0:
                    break
                candidate_remove = selected[i]
                remaining = [s for j, s in enumerate(selected) if j != i]
                dp_remaining = _build_dp(remaining)
                sum_remaining = sum(remaining)
                
                for x in range(1, n + 1):
                    if x in remaining or x == candidate_remove:
                        continue
                    if _is_valid_incremental(remaining, x, dp_remaining, sum_remaining):
                        new_selected = remaining + [x]
                        new_selected.sort()
                        # Try to fill the new set
                        if len(new_selected) < target_size:
                            new_selected = _try_fill(new_selected, list(range(1, n + 1)))
                        if len(new_selected) >= len(selected):
                            selected = new_selected
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        return selected
    
    # Strategy 1: Powers of 2 and their neighbors
    powers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    best = _greedy_from_order(powers + list(range(3, n + 1)))
    
    if len(best) >= target_size:
        return best
    
    # Strategy 2: Greedy from natural order
    order = list(range(1, n + 1))
    result = _greedy_from_order(order)
    if len(result) > len(best):
        best = result
        if len(best) >= target_size:
            return best
    
    # Strategy 3: Greedy from descending order
    order_desc = list(range(n, 0, -1))
    result = _greedy_from_order(order_desc)
    if len(result) > len(best):
        best = result
        if len(best) >= target_size:
            return best
    
    # Strategy 4: Try different random orderings
    for trial in range(200):
        if time.time() > deadline - 20.0:
            break
        random.seed(trial)
        order = list(range(1, n + 1))
        random.shuffle(order)
        result = _greedy_from_order(order)
        if len(result) > len(best):
            best = result
            if len(best) >= target_size:
                return best
    
    # Strategy 5: Local search on best found
    if len(best) < target_size:
        improved = _local_search(best, max_iterations=30)
        if len(improved) > len(best):
            best = improved
            if len(best) >= target_size:
                return best
    
    # Strategy 6: Try starting with large numbers and working down
    for trial in range(100):
        if time.time() > deadline - 10.0:
            break
        random.seed(2000 + trial)
        large_nums = list(range(n, n // 2, -1))
        small_nums = list(range(1, n // 2 + 1))
        random.shuffle(large_nums)
        random.shuffle(small_nums)
        order = large_nums[:len(large_nums)//2] + small_nums[:len(small_nums)//2]
        order += large_nums[len(large_nums)//2:] + small_nums[len(small_nums)//2:]
        random.shuffle(order)
        result = _greedy_from_order(order)
        if len(result) > len(best):
            best = result
            if len(best) >= target_size:
                return best
    
    # Strategy 7: Modular residue ordering
    for mod in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        if time.time() > deadline - 5.0:
            break
        for rem in range(mod):
            order = [x for x in range(1, n + 1) if x % mod == rem]
            result = _greedy_from_order(order)
            if len(result) > len(best):
                best = result
                if len(best) >= target_size:
                    return best
    
    # Strategy 8: Final local search with more iterations
    if len(best) < target_size and time.time() < deadline - 3.0:
        improved = _local_search(best, max_iterations=100)
        if len(improved) > len(best):
            best = improved
    
    return best
# EVOLVE-BLOCK-END
