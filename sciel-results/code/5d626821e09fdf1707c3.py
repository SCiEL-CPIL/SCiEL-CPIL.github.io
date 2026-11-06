"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import factorial
    
    N = 12
    target = factorial(N)
    
    # Generate all divisors of target that are > N
    divisors = []
    i = 1
    while i * i <= target:
        if target % i == 0:
            if i > N:
                divisors.append(i)
            if i != target // i and target // i > N:
                divisors.append(target // i)
        i += 1
        
    divisors.sort()
    
    # We want to find the minimum M such that there exists a subset of distinct divisors 
    # from {d in divisors | d <= M} whose product is target.
    # We use Binary Search on the sorted list of divisors to find this M.
    
    def find_subset_for_max(max_val):
        """
        Returns a list of distinct divisors <= max_val whose product is target, 
        or None if no such subset exists.
        """
        # Filter divisors <= max_val
        # Since divisors is sorted, we can find the cutoff index
        # But for simplicity in recursion, we pass the list and limit
        allowed = []
        for d in divisors:
            if d > max_val:
                break
            allowed.append(d)
            
        # If the product of all allowed divisors is less than target, 
        # it's impossible (though unlikely to be the first check)
        # Actually, we don't need to multiply all, just check if it's possible to reach target.
        
        # Backtracking search
        # To optimize, we can sort allowed in descending order? 
        # No, ascending is better for pruning: if current * d > target, break.
        
        def dfs(idx, current_prod, current_subset):
            if current_prod == target:
                return current_subset
            if idx >= len(allowed):
                return None
            
            # Pruning: 
            # 1. If current_prod > target, return None (handled by loop break)
            # 2. If we can't reach target even by multiplying by the largest possible remaining factors?
            #    This is hard to compute quickly.
            #    Simple pruning: if current_prod * allowed[idx] > target, then 
            #    allowed[idx] and all subsequent (larger) are too big.
            #    Wait, if allowed is sorted ascending, and current_prod * allowed[idx] > target,
            #    then for any j >= idx, current_prod * allowed[j] >= current_prod * allowed[idx] > target.
            #    So we can stop the loop.
            
            for i in range(idx, len(allowed)):
                d = allowed[i]
                if current_prod * d > target:
                    break
                # Try including d
                res = dfs(i + 1, current_prod * d, current_subset + [d])
                if res is not None:
                    return res
            return None
            
        return dfs(0, 1, [])

    # Binary Search for the minimal max_val
    # The answer M must be one of the divisors.
    # low = 0, high = len(divisors) - 1
    low = 0
    high = len(divisors) - 1
    best_subset = None
    best_max = None
    
    # We need to be careful: the "max" of the subset is the largest element in it.
    # If we pick a subset with max element $M$, then $M$ must be in `divisors`.
    # We are searching for the smallest $M$ in `divisors` such that a subset exists with all elements $\le M$ and max element exactly $M$?
    # No, if a subset exists with elements $\le M$, its max is $\le M$.
    # If we find a subset for $M$, its actual max is some $M' \le M$.
    # Since we are searching for the *minimum* possible max, we want the smallest $M$ in the divisor list for which `find_subset_for_max(M)` returns non-None.
    # Because if a solution exists with max $M'$, then `find_subset_for_max(M')` will find it (or a subset of it? No, it finds *a* subset).
    # If `find_subset_for_max(M)` returns a subset, the max of that subset is $\le M$.
    # Let the max of that subset be $M_{actual}$.
    # Then $M_{actual}$ is also in `divisors` and $M_{actual} \le M$.
    # And `find_subset_for_max(M_{actual})` would also find a valid subset (the same one).
    # So the minimal $M$ for which the function returns non-None corresponds to the minimal possible maximum factor.
    
    while low <= high:
        mid = (low + high) // 2
        M = divisors[mid]
        subset = find_subset_for_max(M)
        if subset is not None:
            best_subset = subset
            best_max = M
            high = mid - 1  # Try to find a smaller max
        else:
            low = mid + 1   # M is too small, try larger
            
    if best_subset is not None:
        return best_subset
    else:
        # Fallback
        return [target]

# EVOLVE-BLOCK-END
