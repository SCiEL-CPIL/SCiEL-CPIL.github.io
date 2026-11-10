"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    deadline = time.time() + 85  # 85s safety margin
    
    def is_mean_of_subset(element, others):
        """Check if element is the arithmetic mean of any subset of others with size >= 2."""
        n = len(others)
        if n < 2:
            return False
        target = element
        # For each possible subset size k >= 2, check if sum of k elements = k * target
        # Use bitset DP: dp[s] = bitmask of achievable subset sizes for sum s
        # Actually, let's use: for each k, check if there exist k elements summing to k*target
        # Efficient approach: for each k from 2 to n, use DP
        max_sum = sum(others)
        # dp[j] = set of subset sizes achievable with sum j
        # Better: dp[j] = bitmask where bit k is set if we can make sum j with exactly k elements
        # But this is expensive. Let's use a different approach.
        # For each k, we need to check if k*target can be formed as sum of k elements from others.
        # This is a classic subset sum with cardinality constraint.
        
        # Use DP: dp[s][k] = can we form sum s using exactly k elements?
        # To save memory, use dp[k] = set of achievable sums with k elements
        # Or use bitset: achievable_sums[k] as a set
        
        for k in range(2, n + 1):
            target_sum = k * target
            if target_sum > max_sum or target_sum < 0:
                continue
            # Check if we can form target_sum using exactly k elements from others
            # DP: dp[j] = set of counts achievable for sum j
            # Use: dp[j] = bitmask of achievable cardinalities for sum j
            # Simpler: for each k, do a DP
            # dp[i][j] = can we form sum j using first i elements with exactly j of them?
            # Space optimized: dp[j] = set of achievable sums with exactly j elements
            # Even simpler: use a set-based DP
            sums_by_count = [set() for _ in range(k + 1)]
            sums_by_count[0].add(0)
            for val in others:
                for j in range(k, 0, -1):
                    for s in list(sums_by_count[j - 1]):
                        new_s = s + val
                        if new_s <= target_sum:
                            sums_by_count[j].add(new_s)
            if target_sum in sums_by_count[k]:
                return True
        return False
    
    def check_all_valid(selected):
        """Check that no element is the mean of any subset of >= 2 other elements."""
        for i, a in enumerate(selected):
            others = [x for x in selected if x != a]
            if is_mean_of_subset(a, others):
                return False
        return True
    
    # Greedy construction: try to build a large valid set
    # Strategy: iterate through candidates in a smart order
    # First, try to build from scratch with a good ordering
    
    best_set = []
    
    # Try multiple strategies
    strategies = [
        list(range(1, 513)),           # ascending
        list(range(512, 0, -1)),       # descending
        list(range(256, 513)) + list(range(1, 257)),  # upper half first
        [x for x in range(1, 513) if x % 2 == 1] + [x for x in range(1, 513) if x % 2 == 0],  # odds first
        [x for x in range(1, 513) if x % 3 != 0] + [x for x in range(1, 513) if x % 3 == 0],  # non-multiples of 3 first
    ]
    
    for strategy in strategies:
        if time.time() > deadline - 5:
            break
        
        selected = []
        for x in strategy:
            if time.time() > deadline - 5:
                break
            if x in selected:
                continue
            candidate = selected + [x]
            # Only need to check if x is mean of subset of others, or if any existing element becomes mean
            # But adding x could make x a mean of subset of others, or make existing elements means
            # Full check is expensive, so let's do incremental check
            # Check: is x the mean of any subset of selected?
            # Check: for each a in selected, is a the mean of any subset of (selected - {a}) + {x}?
            # This is complex. Let's just do full check but optimize.
            
            # Quick check: is x the mean of any subset of selected?
            if selected and is_mean_of_subset(x, selected):
                continue
            
            # Check if any existing element becomes a mean
            valid = True
            for i, a in enumerate(selected):
                others = [y for y in candidate if y != a]
                if is_mean_of_subset(a, others):
                    valid = False
                    break
            
            if valid:
                selected = candidate
        
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # If we have time, try to improve by adding more elements
    if best_set and time.time() < deadline - 10:
        # Try to add more elements from the remaining pool
        remaining = [x for x in range(1, 513) if x not in best_set]
        improved = best_set[:]
        for x in remaining:
            if time.time() > deadline - 5:
                break
            candidate = improved + [x]
            # Check validity
            valid = True
            for i, a in enumerate(candidate):
                others = [y for y in candidate if y != a]
                if is_mean_of_subset(a, others):
                    valid = False
                    break
            if valid:
                improved = candidate
        
        if len(improved) > len(best_set):
            best_set = improved
    
    return best_set
# EVOLVE-BLOCK-END
