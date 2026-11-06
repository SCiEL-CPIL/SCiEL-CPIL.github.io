import time
import math

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    divisors.sort(reverse=True)
    
    # Precompute suffix sums for pruning
    # suffix_max[i] = sum of divisors[i:]
    # This allows O(1) max sum check
    m = len(divisors)
    suffix_sum = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
    
    # Precompute prefix sums of smallest?
    # For min sum check with count k starting at idx:
    # We need the smallest k elements in divisors[idx:].
    # Since divisors is sorted descending, the smallest elements are at the end.
    # If we pick k elements from idx to end, the minimum sum is sum of the last k elements in that range.
    # i.e., sum(divisors[m-k : m]) if idx <= m-k.
    # More generally, if we are at index idx and need to pick 'count' items from idx..m-1:
    # The items available are divisors[idx], ..., divisors[m-1].
    # To minimize sum, we pick the smallest 'count' items from this set.
    # Since the set is sorted descending, the smallest 'count' items are the last 'count' items:
    # divisors[m-count], ..., divisors[m-1].
    # This is valid only if there are at least 'count' items available, i.e., m - idx >= count.
    # And we must ensure we don't pick items before idx?
    # Wait, if we pick divisors[m-count] and m-count < idx, that would be invalid.
    # So we need m-count >= idx.
    
    def min_sum_from(idx, count):
        # Smallest 'count' elements in divisors[idx:]
        # These are divisors[m-count : m]
        # Valid if m - count >= idx
        if m - count < idx:
            return float('inf')
        return suffix_sum[m - count]
    
    start_time = time.time()
    deadline = start_time + 80.0 # 10s safety margin
    
    # Try increasing k
    # Max k is bounded by m, but practically small
    for k in range(1, m + 1):
        # Quick check: max possible sum with k items
        # Max sum is sum of top k items: divisors[0:k]
        max_possible = suffix_sum[0] - suffix_sum[k]
        if max_possible < target:
            continue
            
        # Min possible sum with k items (from all items)
        min_possible = min_sum_from(0, k)
        if min_possible > target:
            # If even the smallest k items sum to more than target, and we are trying to minimize k,
            # then larger k will only increase min sum? No, larger k allows smaller items.
            # But if min_sum_from(0, k) > target, it means sum of smallest k items > target.
            # For k+1, sum of smallest k+1 items = min_sum_from(0, k) + smallest_item.
            # So min sum increases with k.
            # Thus if min_sum_from(0, k) > target, no solution for this k or any larger k.
            # Wait, is this true?
            # min_sum_from(0, k) is sum of the k smallest divisors.
            # If sum of k smallest > target, then any subset of size k has sum >= that.
            # So no solution of size k.
            # And for size k+1, the sum of k+1 smallest is even larger.
            # So we can break.
            break
            
        # DFS
        path = []
        found = False
        
        def dfs(idx, remaining, count):
            nonlocal found
            if found:
                return
            if time.time() > deadline:
                return
                
            if count == 0:
                if remaining == 0:
                    found = True
                return
            if idx >= m:
                return
                
            # Pruning: Max sum
            # Max sum we can achieve with 'count' items from idx..end is sum of first 'count' items in that range.
            # i.e., divisors[idx], ..., divisors[idx+count-1]
            # Sum = suffix_sum[idx] - suffix_sum[idx+count]
            if idx + count <= m:
                max_sum_here = suffix_sum[idx] - suffix_sum[idx + count]
                if remaining > max_sum_here:
                    return
            else:
                # Not enough items
                return

            # Pruning: Min sum
            min_sum_here = min_sum_from(idx, count)
            if remaining < min_sum_here:
                return
                
            # Option 1: Pick divisors[idx]
            if divisors[idx] <= remaining:
                path.append(divisors[idx])
                dfs(idx + 1, remaining - divisors[idx], count - 1)
                path.pop()
                
            if found:
                return
                
            # Option 2: Skip divisors[idx]
            # Check if we have enough items left to pick 'count' items
            if m - (idx + 1) >= count:
                dfs(idx + 1, remaining, count)
                
        dfs(0, target, k)
        
        if found:
            if sum(path) == target:
                return path

    # Fallback: Greedy
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen
