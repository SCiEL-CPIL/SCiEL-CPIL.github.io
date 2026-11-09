def construct():
    import time
    from math import factorial
    
    deadline = time.time() + 75
    
    target = factorial(12)  # 479001600
    
    def get_divisors(num):
        divs = [1]
        p = 2
        while p * p <= num:
            while num % p == 0:
                num //= p
                new_divs = [d * p for d in divs]
                divs.extend(new_divs)
            p += 1
        if num > 1:
            new_divs = [d * num for d in divs]
            divs.extend(new_divs)
        return sorted(set(divs))
    
    all_divs = get_divisors(target)
    candidates = [d for d in all_divs if d > 12]
    
    def find_factorization(max_bound):
        if time.time() > deadline:
            return None
        
        cands = sorted([c for c in candidates if c <= max_bound])
        n_cands = len(cands)
        
        memo = set()
        
        def dfs(remaining, start_idx, selected):
            if time.time() > deadline:
                return None
            if remaining == 1:
                return list(selected)
            if start_idx >= n_cands:
                return None
            if (remaining, start_idx) in memo:
                return None
            
            for i in range(start_idx, n_cands):
                c = cands[i]
                if c > remaining:
                    break
                if remaining % c == 0:
                    selected.append(c)
                    result = dfs(remaining // c, i + 1, selected)
                    selected.pop()
                    if result is not None:
                        return result
            
            memo.add((remaining, start_idx))
            return None
        
        return dfs(target, 0, [])
    
    lo = 13
    hi = target
    best = None
    
    while lo <= hi and time.time() < deadline - 3:
        mid = (lo + hi) // 2
        result = find_factorization(mid)
        if result is not None:
            best = result
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best is None:
        return [target]
    
    return best
