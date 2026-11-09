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
    deadline = start_time + 55.0  # Safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # 1. Generate divisors
    divisors = _get_divisors(n)
    divisors = [d for d in divisors if d <= target]
    
    num_divs = len(divisors)
    
    if num_divs == 0:
        return []
    
    divisor_set = set(divisors)
    
    # Precompute prefix sums for O(1) max-sum queries
    prefix_sums = [0] * (num_divs + 1)
    for i in range(num_divs):
        prefix_sums[i + 1] = prefix_sums[i] + divisors[i]
    
    def min_sum_from(start_idx, count):
        """Sum of 'count' smallest elements from divisors[start_idx:]."""
        if start_idx + count > num_divs:
            return float('inf')
        return prefix_sums[num_divs] - prefix_sums[num_divs - count]
    
    def max_sum_from(start_idx, count):
        """Sum of 'count' largest elements from divisors[start_idx:]."""
        if start_idx + count > num_divs:
            return -1
        return prefix_sums[start_idx + count] - prefix_sums[start_idx]
    
    def dfs(start_idx, remaining, count, selected):
        """
        Try to find 'count' more divisors from divisors[start_idx:] summing to 'remaining'.
        Returns list of selected divisors if found, else None.
        """
        if time.time() > deadline:
            return None
            
        if count == 0:
            if remaining == 0:
                return selected[:]
            return None
            
        if remaining < 0 or remaining == 0:
            return None
            
        if num_divs - start_idx < count:
            return None
            
        max_possible = max_sum_from(start_idx, count)
        if remaining > max_possible:
            return None
            
        min_possible = min_sum_from(start_idx, count)
        if remaining < min_possible:
            return None

        for i in range(start_idx, num_divs):
            d = divisors[i]
            
            if d > remaining:
                continue
                
            if remaining - d < min_sum_from(i + 1, count - 1):
                continue
                
            if remaining - d > max_sum_from(i + 1, count - 1):
                break
                
            selected.append(d)
            result = dfs(i + 1, remaining - d, count - 1, selected)
            selected.pop()
            
            if result is not None:
                return result
                
        return None
    
    # k=1: direct check
    if target in divisor_set:
        return [target]
    
    # k=2: two-sum with set
    for i in range(num_divs):
        d1 = divisors[i]
        if d1 >= target:
            continue
        complement = target - d1
        if complement in divisor_set and complement != d1:
            # Need to ensure complement appears after index i in divisors (distinct indices)
            # Since divisors are sorted descending and we want distinct values,
            # just check complement != d1 and complement is a valid divisor
            return [d1, complement]
    
    # k=3: for each d1, do two-sum on remaining
    for i in range(num_divs):
        d1 = divisors[i]
        if d1 >= target:
            continue
        rem = target - d1
        # Two-sum in divisors[i+1:]
        seen = set()
        for j in range(i + 1, num_divs):
            d2 = divisors[j]
            if d2 >= rem:
                continue
            complement = rem - d2
            if complement in seen:
                return [d1, d2, complement]
            seen.add(d2)
    
    # k=4 and up: DFS with branch and bound
    for k in range(4, 257):
        if time.time() > deadline:
            break
        result = dfs(0, target, k, [])
        if result is not None:
            assert sum(result) == target
            assert len(set(result)) == len(result)
            assert all(n % d == 0 for d in result)
            assert len(result) <= 256
            return result
    
    # Fallback: greedy (shouldn't happen)
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
    
    if remainder == 0:
        assert sum(chosen) == target
        assert len(set(chosen)) == len(chosen)
        assert all(n % d == 0 for d in chosen)
        assert len(chosen) <= 256
        return chosen
    
    # Last resort fallback
    best_result = [2520, 1680, 560, 210, 9]
    assert sum(best_result) == target
    assert len(set(best_result)) == len(best_result)
    assert all(n % d == 0 for d in best_result)
    assert len(best_result) <= 256
    return best_result
