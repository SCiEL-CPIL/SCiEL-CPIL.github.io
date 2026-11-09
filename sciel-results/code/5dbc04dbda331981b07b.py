def construct():
    import time
    start_time = time.time()
    DEADLINE = 55.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    mid = n // 2
    left = denominators[:mid]
    right = denominators[mid:]
    
    def gen_sums(dens):
        sums = [0.0]
        for d in dens:
            val = 1.0 / d
            new_sums = []
            for s in sums:
                ns = s + val
                if ns < 1.0:
                    new_sums.append(ns)
            sums.extend(new_sums)
        return sums
    
    def reconstruct(dens, target_sum):
        n = len(dens)
        result = []
        def dfs(idx, remaining):
            if idx == n:
                return abs(remaining) < 1e-12
            # Try exclude first
            if dfs(idx + 1, remaining):
                return True
            # Try include
            v = 1.0 / dens[idx]
            if remaining >= v - 1e-12:
                result.append(idx)
                if dfs(idx + 1, remaining - v):
                    return True
                result.pop()
            return False
        if dfs(0, target_sum):
            return result
        return []
    
    if time.time() - start_time > DEADLINE:
        return [0]
    
    left_sums = gen_sums(left)
    
    if time.time() - start_time > DEADLINE:
        selected = []
        total = 0.0
        for i, d in enumerate(denominators):
            if total + 1.0/d < 1.0:
                selected.append(i)
                total += 1.0/d
        return selected
    
    right_sums = gen_sums(right)
    
    if time.time() - start_time > DEADLINE:
        selected = []
        total = 0.0
        for i, d in enumerate(denominators):
            if total + 1.0/d < 1.0:
                selected.append(i)
                total += 1.0/d
        return selected
    
    right_sums.sort()
    
    import bisect
    best_sum = -1.0
    best_left_idx = 0
    best_right_idx = 0
    
    for i, ls in enumerate(left_sums):
        target = 1.0 - ls
        pos = bisect.bisect_left(right_sums, target)
        if pos > 0:
            rs = right_sums[pos - 1]
            total = ls + rs
            if total > best_sum:
                best_sum = total
                best_left_idx = i
                best_right_idx = pos - 1
    
    if time.time() - start_time > DEADLINE - 5.0:
        # Not enough time for reconstruction, use greedy
        selected = []
        total = 0.0
        for i, d in enumerate(denominators):
            if total + 1.0/d < 1.0:
                selected.append(i)
                total += 1.0/d
        return selected
    
    left_indices = reconstruct(left, left_sums[best_left_idx])
    right_indices = reconstruct(right, right_sums[best_right_idx])
    
    final_indices = []
    for i in left_indices:
        final_indices.append(i)
    for i in right_indices:
        final_indices.append(mid + i)
    
    if not final_indices:
        return [0]
    
    return sorted(final_indices)
