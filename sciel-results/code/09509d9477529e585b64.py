import time

def _get_divisors(n):
    """Generate all divisors of n, sorted in descending order."""
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    divs.sort(reverse=True)
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = 5040
    target = 4979
    
    # 1. Generate divisors
    divisors = _get_divisors(n)
    divisors = [d for d in divisors if d <= target]
    num_divs = len(divisors)
    
    if num_divs == 0:
        return []
    
    # Precompute prefix sums for O(1) max-sum queries
    prefix_sums = [0] * (num_divs + 1)
    for i in range(num_divs):
        prefix_sums[i + 1] = prefix_sums[i] + divisors[i]
    
    # Precompute min_sum_for_k[k] = sum of k smallest divisors
    min_sum_for_k = [0] * (num_divs + 1)
    for k in range(1, num_divs + 1):
        min_sum_for_k[k] = prefix_sums[num_divs] - prefix_sums[num_divs - k]
    
    def dfs(start_idx, remaining, count, selected):
        """Try to find 'count' more divisors from divisors[start_idx:] summing to 'remaining'."""
        if time.time() > deadline:
            return None
        if count == 0:
            return selected[:] if remaining == 0 else None
        if remaining < 0 or remaining == 0:
            return None
        if num_divs - start_idx < count:
            return None
        if start_idx + count > num_divs:
            return None
        
        max_possible = prefix_sums[start_idx + count] - prefix_sums[start_idx]
        if remaining > max_possible:
            return None
        if remaining < min_sum_for_k[count]:
            return None
        
        for i in range(start_idx, num_divs):
            d = divisors[i]
            if d > remaining:
                continue
            if remaining - d < min_sum_for_k[count - 1]:
                continue
            
            # Max check for remainder
            rem_idx = i + 1
            if rem_idx + (count - 1) <= num_divs:
                max_rem = prefix_sums[rem_idx + count - 1] - prefix_sums[rem_idx]
                if remaining - d > max_rem:
                    continue
            
            selected.append(d)
            result = dfs(i + 1, remaining - d, count - 1, selected)
            selected.pop()
            if result is not None:
                return result
        return None
    
    # Try increasing k from 1 upward
    best_result = None
    for k in range(1, 257):
        if time.time() > deadline:
            break
        result = dfs(0, target, k, [])
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        # Fallback: known valid solution
        best_result = [2520, 1680, 560, 210, 9]
    
    # Final verification
    assert sum(best_result) == target
    assert len(set(best_result)) == len(best_result)
    assert all(n % d == 0 for d in best_result)
    assert len(best_result) <= 256
    
    return best_result
