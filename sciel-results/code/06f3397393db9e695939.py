import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    target = math.factorial(12)
    
    # Get all divisors > 12
    divisors = []
    sqrt_n = int(math.isqrt(target))
    for i in range(1, sqrt_n + 1):
        if target % i == 0:
            if i > 12:
                divisors.append(i)
            other = target // i
            if other != i and other > 12:
                divisors.append(other)
    
    candidates = sorted(set(divisors))
    
    # Precompute for pruning
    # We need to check if a solution exists with max factor <= M
    def exists_solution(M):
        # Filter candidates <= M
        valid_cands = [c for c in candidates if c <= M]
        if not valid_cands:
            return None
        
        # DFS
        # rem: remaining product
        # start_idx: index in valid_cands to start picking from
        # path: list of chosen factors
        def dfs(rem, start_idx, path):
            if time.time() > deadline:
                return None
            
            if rem == 1:
                return path[:]
            
            if start_idx >= len(valid_cands):
                return None
            
            if len(path) >= 12:
                return None
            
            # Pruning: 
            # If rem > 1, we need at least one more factor.
            # The smallest available factor is valid_cands[start_idx].
            # If rem < valid_cands[start_idx], we can't form rem with distinct factors >= valid_cands[start_idx]
            # unless rem is 1 (handled) or rem is a factor itself that is >= start_idx?
            # No, the next factor must be from valid_cands[start_idx:].
            # So if rem > 1 and rem < valid_cands[start_idx], it's impossible.
            if rem > 1 and rem < valid_cands[start_idx]:
                return None
            
            # Another pruning:
            # If rem is prime, it must be in valid_cands[start_idx:] and not used.
            # Since we pick distinct factors and we are moving forward, if rem is prime,
            # it must be equal to some valid_cands[j] for j >= start_idx.
            # And since it's prime, it can't be a product of other factors.
            # So we just need to check if rem is in valid_cands[start_idx:].
            # If so, we can return [rem] if rem > 12 (which it is) and rem <= M (which it is).
            # But we must ensure it's not in path.
            # Since we are picking in increasing order? No, we are picking from sorted list, but path can have any order.
            # However, if we enforce that we pick factors in increasing order of value, then path is sorted.
            # If path is sorted, and rem is prime, then rem must be the last factor.
            # And rem > last factor in path.
            # So if rem is prime, we just check if rem in valid_cands and index > start_idx?
            # Actually, if we pick factors in increasing order, then all future factors are >= current.
            # So if rem is prime, it must be >= valid_cands[start_idx].
            # And it must not be in path. Since path contains factors < valid_cands[start_idx] (if we enforce increasing order),
            # and rem >= valid_cands[start_idx], rem is not in path.
            # So if rem is prime, we just need to check if rem is in valid_cands.
            
            # Let's stick to general DFS with increasing order constraint to ensure distinctness easily.
            # This means we only pick factors from valid_cands[start_idx:].
            
            for i in range(start_idx, len(valid_cands)):
                f = valid_cands[i]
                if f > rem:
                    break
                if rem % f == 0:
                    # Check if we can complete the rest
                    # The remaining product is rem // f
                    # We must form rem // f from valid_cands[i+1:]
                    
                    # Pruning: 
                    # If rem // f > 1, we need at least one more factor from valid_cands[i+1:].
                    # The smallest such factor is valid_cands[i+1] (if i+1 < len).
                    # If rem // f < valid_cands[i+1], then we can't form it.
                    # Unless rem // f is 1.
                    if rem // f > 1:
                        if i + 1 >= len(valid_cands):
                            continue
                        if rem // f < valid_cands[i + 1]:
                            continue
                    
                    res = dfs(rem // f, i + 1, path + [f])
                    if res is not None:
                        return res
            return None
        
        return dfs(target, 0, [])

    # Binary search for minimum M
    # Lower bound: 13
    # Upper bound: Let's start with a reasonable upper bound.
    # If we use 2 factors, max is ~21886.
    # If we use 12 factors, max is ~34.
    # Let's search in [13, 1000] first.
    low = 13
    high = 1000
    
    # Check if high is feasible
    # If not, increase high
    while time.time() < deadline - 5:
        sol = exists_solution(high)
        if sol is not None:
            # Found a solution with max <= high
            # Now binary search for the minimum M
            best_sol = sol
            best_M = high
            
            while low < high and time.time() < deadline - 2:
                mid = (low + high) // 2
                sol_mid = exists_solution(mid)
                if sol_mid is not None:
                    best_sol = sol_mid
                    best_M = mid
                    high = mid - 1
                else:
                    low = mid + 1
            
            return best_sol
        
        # If no solution with high, increase high
        # Double the high
        high *= 2
        if high > target:
            # Fallback
            return [target]

    # Fallback
    return [target]
