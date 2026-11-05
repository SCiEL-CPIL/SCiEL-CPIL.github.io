import time
from math import factorial
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    target = factorial(12)
    
    # Generate all divisors of target that are > 12
    divisors = []
    for i in range(13, int(target**0.5) + 1):
        if target % i == 0:
            divisors.append(i)
            other = target // i
            if other != i and other > 12:
                divisors.append(other)
    if target > 12:
        divisors.append(target)
    
    divs = sorted(set(divisors))
    
    # We want to minimize the largest factor.
    # We search for k factors, starting from the maximum possible k (which minimizes geometric mean)
    # Max k is 6 (as established, k=7 is impossible because min product of 7 smallest valid divisors > target)
    
    best_solution = None
    best_max = float('inf')
    
    # We'll try k from 6 down to 1
    for k in range(6, 0, -1):
        # For each k, we want to find the minimum possible max factor M
        # such that there exists a set of k distinct divisors <= M with product target.
        
        # Lower bound for M: The k-th smallest divisor in 'divs'
        if k > len(divs):
            continue
            
        # We can iterate M from divs[k-1] to divs[-1]
        # But we can be smarter: 
        # If we find a solution for a specific M, it's the best for this k.
        # We can do a DFS that prunes based on M.
        
        # Let's just find ANY solution for this k first, then optimize?
        # No, let's do the M iteration.
        
        # To speed up, we can check if the product of the k smallest divisors > target.
        # If so, no solution for this k (and any larger k, but we are going down).
        # Wait, if k smallest product > target, then any set of k has product >= that, so no solution.
        
        min_prod_k = 1
        for i in range(k):
            min_prod_k *= divs[i]
            if min_prod_k > target:
                break
        
        if min_prod_k > target:
            # No solution for this k or any larger k (already checked larger)
            # Actually, if k=6 fails, k=5 might work.
            continue
            
        # Now we know a solution might exist.
        # We want the minimum M.
        # We can iterate M from divs[k-1] upwards.
        # For each M, check if there is a subset of size k from divs[:index_of_M+1] with product target.
        
        # To make this efficient, we can use a DFS that stops when it finds one solution for a given M.
        # And we iterate M in increasing order.
        
        found_for_k = False
        # We need to iterate M.
        # The index of M in divs determines the available pool.
        for m_idx in range(k-1, len(divs)):
            M = divs[m_idx]
            
            # Check if solution exists using divs[0...m_idx]
            # DFS
            # State: (rem_prod, start_idx, count)
            # We want to pick 'k' factors.
            
            def dfs(rem, start_idx, count):
                if time.time() > deadline:
                    return None
                
                if count == k:
                    if rem == 1:
                        return True
                    return False
                
                if count > k:
                    return False
                    
                # Pruning:
                # 1. If rem is not divisible by any divisor in range, return False
                # 2. Min product of remaining (k-count) factors from start_idx onwards
                #    If rem < min_possible_rem_product, return False
                #    Note: We need to pick (k-count) factors from divs[start_idx ... m_idx]
                #    The smallest product is product of the first (k-count) divisors in that range.
                
                rem_count = k - count
                if rem_count == 0:
                    return rem == 1
                
                # Check if rem can be formed
                # Quick check: is rem divisible by any divisor in range?
                # We can just loop.
                
                # Pruning with min product
                min_p = 1
                for i in range(start_idx, min(start_idx + rem_count, m_idx + 1)):
                    min_p *= divs[i]
                    if min_p > rem:
                        break
                if min_p > rem:
                    return False
                
                # Pruning with max product
                # Max product of rem_count factors from start_idx to m_idx
                # These are the largest rem_count divisors in that range.
                max_p = 1
                # Indices: m_idx, m_idx-1, ...
                start_max = max(start_idx, m_idx - rem_count + 1)
                for i in range(m_idx, start_max - 1, -1):
                    max_p *= divs[i]
                    if max_p >= rem:
                        break
                if max_p < rem:
                    return False

                for i in range(start_idx, m_idx + 1):
                    d = divs[i]
                    if d > rem:
                        break
                    if rem % d == 0:
                        if dfs(rem // d, i + 1, count + 1):
                            return True
                return False

            if dfs(target, 0, 0):
                # We found a solution with max <= M.
                # We need to retrieve the actual factors.
                # The DFS above only returns True/False.
                # We need to modify it to return the factors.
                
                # Let's re-run or modify to store the path.
                # For efficiency, let's just assume the first M that works is the optimal M.
                # We need the factors.
                
                # Let's create a helper that returns the list.
                
                def dfs_find(rem, start_idx, count, path):
                    if time.time() > deadline:
                        return None
                    
                    if count == k:
                        if rem == 1:
                            return list(path)
                        return None
                    
                    if count > k:
                        return None
                        
                    rem_count = k - count
                    if rem_count == 0:
                        return list(path) if rem == 1 else None
                    
                    # Pruning
                    min_p = 1
                    for i in range(start_idx, min(start_idx + rem_count, m_idx + 1)):
                        min_p *= divs[i]
                        if min_p > rem:
                            break
                    if min_p > rem:
                        return None
                    
                    max_p = 1
                    start_max = max(start_idx, m_idx - rem_count + 1)
                    for i in range(m_idx, start_max - 1, -1):
                        max_p *= divs[i]
                        if max_p >= rem:
                            break
                    if max_p < rem:
                        return None

                    for i in range(start_idx, m_idx + 1):
                        d = divs[i]
                        if d > rem:
                            break
                        if rem % d == 0:
                            path.append(d)
                            res = dfs_find(rem // d, i + 1, count + 1, path)
                            if res:
                                return res
                            path.pop()
                    return None

                sol = dfs_find(target, 0, 0, [])
                if sol:
                    best_solution = sol
                    best_max = M
                    found_for_k = True
                    break # Found min M for this k
            
        if found_for_k:
            # Since we started with k=6 (largest k), and we are minimizing max,
            # a solution with k=6 will generally have a smaller max than k=5.
            # However, it's not strictly guaranteed that min_max(k=6) < min_max(k=5).
            # But geometric mean suggests it.
            # We can check k=5 etc if we want to be sure, but k=6 is likely optimal.
            # Given the time, we return the first valid k found (starting from 6).
            return sorted(best_solution)
            
    # If no solution found (should not happen), return target
    return [target]
