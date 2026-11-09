import math
import time
import bisect
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin before 60s limit

    target = math.factorial(12)  # 479001600
    
    # Known solution with max factor 33
    known_sol = [24, 25, 27, 28, 32, 33]
    best_sol = known_sol
    best_max = max(known_sol)
    
    # Verify known solution
    prod = 1
    for x in best_sol:
        prod *= x
    if prod != target:
        return [target]
    
    # Calculate all divisors of target that are > 12
    divs = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            if i > 12:
                divs.append(i)
            other = target // i
            if other != i and other > 12:
                divs.append(other)
    divs.sort()
    
    # Build a set for O(1) lookup
    div_set = set(divs)
    
    # We want to find if there's a solution with max < best_max
    # Try D from 13 up to best_max - 1
    for D in range(13, best_max):
        if time.time() > deadline:
            break
        
        # Candidates are divisors d such that 13 <= d <= D
        cands = [d for d in divs if 13 <= d <= D]
        if not cands:
            continue
        
        cand_set = set(cands)
        solution = None
        
        # Use a more efficient DFS with pruning
        # State: remaining product, minimum index to start from, current path
        def dfs(rem, min_idx, path, depth):
            nonlocal solution
            if solution is not None:
                return
            if time.time() > deadline:
                return
            if rem == 1:
                if len(path) >= 1:
                    solution = path[:]
                return
            
            # Max 12 factors
            if depth >= 12:
                return
            
            # Optimization: If rem itself is a valid candidate and we can use it
            if rem in cand_set:
                idx = bisect.bisect_left(cands, rem)
                if idx < len(cands) and cands[idx] == rem and idx > min_idx:
                    # Check distinctness: rem not already in path
                    if rem not in path:
                        solution = path + [rem]
                        return
            
            # Pruning: if the product of the smallest remaining candidates (from min_idx) 
            # cannot reach rem, we can prune
            # Actually, let's use a different pruning: 
            # If rem has a prime factor > D, we can't form it from candidates <= D
            # But all our candidates are divisors of target, so this is automatically handled
            
            # Iterate candidates from min_idx onwards
            for i in range(min_idx, len(cands)):
                d = cands[i]
                if d > rem:
                    break
                
                if rem % d == 0:
                    # Check that d is not already in path (distinctness)
                    if d in path:
                        continue
                    dfs(rem // d, i + 1, path + [d], depth + 1)
                    if solution is not None:
                        return
        
        dfs(target, 0, [], 0)
        
        if solution is not None:
            best_sol = solution
            best_max = D
            break  # Found the smallest D, so we're done
    
    # Return the best solution found
    return best_sol
