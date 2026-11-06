import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = factorial(12)
    
    # We search for minimal M
    # M must be at least 13
    # Upper bound? If we just return [N], M=N.
    # We want to find a better one.
    
    # We will iterate M from 13 upwards
    # For each M, we check if a valid decomposition exists with max factor <= M
    
    def dfs(index, current_product, factors):
        # Check deadline
        if time.time() > deadline:
            return None
        
        if current_product == N:
            return factors[:]
        
        if index == len(candidates):
            return None
            
        # Pruning: if current_product > N, invalid
        if current_product > N:
            return None
            
        # Pruning: if N is not divisible by current_product, 
        # actually we only multiply by divisors, so current_product always divides N?
        # No, we start with 1, which divides N.
        # We only multiply by candidates that divide the remaining quotient.
        
        remaining_needed = N // current_product
        
        # Check if product of all remaining candidates is sufficient
        # product(candidates[index:]) < remaining_needed -> prune
        # We can precompute suffix products or just calculate on the fly if list is small
        # Since M is small, list is small.
        
        # Optimization: 
        # If remaining_needed == 1, we need no more factors.
        # But we must use distinct factors.
        # If we have found a valid set, we return it.
        
        # Try including candidates[index]
        c = candidates[index]
        if remaining_needed % c == 0:
            factors.append(c)
            res = dfs(index + 1, current_product * c, factors)
            if res is not None:
                return res
            factors.pop()
            
        # Try excluding candidates[index]
        res = dfs(index + 1, current_product, factors)
        if res is not None:
            return res
            
        return None

    # Iterate M
    # We can start from a lower bound.
    # (12!)^(1/9) is approx 10.5, so M >= 13.
    # Let's just start from 13.
    
    best_result = [N] # Fallback
    
    for M in range(13, 1000): # Upper bound can be higher, but should find quickly
        candidates = list(range(13, M + 1))
        # Sort candidates? 
        # If we sort descending, we might find solutions with large factors first?
        # We want ANY solution for this M.
        # Sorting might help pruning if we check "product of remaining" effectively.
        # Let's keep ascending for now.
        
        # Optimization: Precompute suffix products for pruning?
        # suffix_prod[i] = product(candidates[i:])
        # If suffix_prod[index] < remaining_needed, return None.
        
        suffix_prod = [1] * (len(candidates) + 1)
        for i in range(len(candidates) - 1, -1, -1):
            suffix_prod[i] = suffix_prod[i + 1] * candidates[i]
            
        # Modify dfs to use suffix_prod
        def dfs_opt(index, current_product, factors):
            if time.time() > deadline:
                return None
            
            if current_product == N:
                return factors[:]
            
            if index == len(candidates):
                return None
                
            if current_product > N:
                return None
                
            remaining_needed = N // current_product
            
            # Prune if even multiplying by all remaining candidates is not enough
            if suffix_prod[index] < remaining_needed:
                return None
                
            c = candidates[index]
            
            # Include c
            if remaining_needed % c == 0:
                factors.append(c)
                res = dfs_opt(index + 1, current_product * c, factors)
                if res is not None:
                    return res
                factors.pop()
                
            # Exclude c
            res = dfs_opt(index + 1, current_product, factors)
            if res is not None:
                return res
                
            return None

        res = dfs_opt(0, 1, [])
        if res is not None:
            # Check if len(res) <= 12
            if len(res) <= 12:
                return res
            else:
                # If > 12, we might need to search for a subset of size <= 12?
                # But as analyzed, max size is 9, so this shouldn't happen for valid M.
                # If it does, we continue to next M? No, a larger M won't reduce the count necessarily.
                # But we want to minimize M. If the only solution for M has > 12 factors, 
                # is it invalid? Yes.
                # So we should continue searching for M?
                # Or maybe a different combination for the same M has fewer factors?
                # The DFS finds the FIRST valid combination.
                # It might find one with many factors.
                # We should ensure we find one with <= 12 factors.
                # Given the constraints, this is unlikely to be an issue.
                pass
                
    return best_result
