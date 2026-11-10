def construct():
    import time
    import math
    from functools import lru_cache
    
    deadline = time.time() + 85  # safety margin before 90s
    
    n = 12
    target = math.factorial(n)
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    # We'll represent a number by its prime exponent vector (e2, e3, e5, e7, e11)
    
    target_vec = (10, 5, 2, 1, 1)
    
    # Generate all divisors of 12! that are > 12
    # A divisor is determined by (a2, a3, a5, a7, a11) where 0<=a2<=10, etc.
    divisors = []
    for a2 in range(11):
        for a3 in range(6):
            for a5 in range(3):
                for a7 in range(2):
                    for a11 in range(2):
                        val = (2 ** a2) * (3 ** a3) * (5 ** a5) * (7 ** a7) * (11 ** a11)
                        if val > n:
                            divisors.append((val, (a2, a3, a5, a7, a11)))
    
    # Sort by value (smaller first) to help minimize max
    divisors.sort(key=lambda x: x[0])
    
    # Precompute: for each divisor, its vector
    # We'll use DFS with pruning
    
    best_max = target  # worst case: single factor
    best_solution = [target]
    
    # DFS: build a list of distinct factors > 12, product = target
    # State: remaining product (as vector), set of used factors, current list, current max
    # Prune: if current max >= best_max, stop (unless we can still improve)
    # Actually, we want to minimize max, so if current max >= best_max, we can prune
    #   (since adding more factors can only keep max the same or increase it... wait no)
    # Actually the max is the max of all factors chosen so far. Adding more factors
    # doesn't decrease the max. So if current max >= best_max, we can prune.
    
    # But we also need: the remaining product must be expressible as product of
    # distinct divisors > 12, none equal to already used ones, and each <= best_max - 1
    # (to improve on best_max)
    
    # Let's use a different approach: iterative deepening on the max factor
    # For a given max M, check if there exists a valid decomposition with all factors <= M
    # Binary search on M
    
    # Actually, let me just do a DFS that tries to find good solutions
    # and keeps track of the best (minimum max) found
    
    used = set()
    current = []
    
    def vec_sub(v1, v2):
        return tuple(v1[i] - v2[i] for i in range(5))
    
    def vec_geq(v1, v2):
        return all(v1[i] >= v2[i] for i in range(5))
    
    def vec_eq_zero(v):
        return all(x == 0 for x in v)
    
    def dfs(remaining, current_list, current_max, min_factor):
        nonlocal best_max, best_solution
        
        if time.time() > deadline:
            return
        
        if vec_eq_zero(remaining):
            if len(current_list) >= 1:
                if current_max < best_max:
                    best_max = current_max
                    best_solution = list(current_list)
            return
        
        # Prune: if current_max >= best_max, no improvement possible
        # (since max can only stay same or increase)
        if current_max >= best_max:
            return
        
        # Try adding each possible next factor
        # The next factor must be:
        # - a divisor of remaining (vec-wise)
        # - > 12
        # - not in used
        # - >= min_factor (to avoid permutations; we build in non-decreasing order)
        # - <= best_max - 1 (to potentially improve)
        
        for val, vec in divisors:
            if time.time() > deadline:
                return
            
            if val < min_factor:
                continue
            if val >= best_max:
                break  # since sorted, all subsequent are larger
            
            if val in used:
                continue
            
            # Check if vec divides remaining
            if not vec_geq(remaining, vec):
                continue
            
            new_remaining = vec_sub(remaining, vec)
            new_max = max(current_max, val)
            
            if new_max >= best_max:
                continue
            
            used.add(val)
            current_list.append(val)
            
            dfs(new_remaining, current_list, new_max, val + 1)
            
            current_list.pop()
            used.remove(val)
    
    dfs(target_vec, [], 13, 13)
    
    return best_solution
