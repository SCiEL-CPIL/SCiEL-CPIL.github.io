def construct():
    import time
    import random
    from fractions import Fraction
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    target = Fraction(5, 31)
    
    # Helper to check if a list of denominators sums to target
    def is_valid(denoms):
        if not denoms:
            return False
        if len(denoms) > 256:
            return False
        if len(set(denoms)) != len(denoms):
            return False
        if any(d < 1 or d > 10000000 for d in denoms):
            return False
        s = sum(Fraction(1, d) for d in denoms)
        return s == target

    # Greedy algorithm to find a valid initial Egyptian fraction
    def greedy_expansion(target_frac):
        denoms = []
        rem = target_frac
        min_d = 1
        while rem > 0:
            # Find smallest d >= min_d such that 1/d <= rem
            d = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
            # Ensure 1/d <= rem
            while Fraction(1, d) > rem:
                d += 1
            denoms.append(d)
            rem -= Fraction(1, d)
            min_d = d + 1
            if len(denoms) > 256:
                return None
        return denoms

    # Initialize with a few different greedy starts to get diverse starting points
    initial_solutions = []
    
    # Standard greedy
    sol = greedy_expansion(target)
    if sol and is_valid(sol):
        initial_solutions.append(sol)
        
    # Try greedy with slight variations (e.g., skipping first few candidates)
    # For 5/31, 1/7 is the natural first step. Let's try starting with 1/8, 1/9 etc. if possible?
    # Greedy is deterministic. To get diversity, we can try "randomized greedy"
    # or just use the standard one and rely on local search.
    
    # A better initial solution might be found by trying to minimize max denominator directly
    # Let's try a few specific known small expansions or generate them via search
    # 5/31 = 1/7 + 1/109 + 1/15263 (max 15263)
    # 5/31 = 1/8 + 1/28 + 1/1736 (max 1736) -> This is much better!
    # Let's hardcode a good starting point if we can find it quickly, or use the greedy one.
    # The greedy for 5/31:
    # 1/7, rem = 5/31 - 1/7 = (35-31)/217 = 4/217
    # 1/55 (ceil(217/4)=55), rem = 4/217 - 1/55 = (220-217)/11935 = 3/11935
    # 1/4000 (ceil(11935/3)=3979), rem = 3/11935 - 1/3979 ... this grows large.
    
    # The solution [8, 28, 1736] is known. Let's verify it.
    # 1/8 + 1/28 + 1/1736 = 217/1736 + 62/1736 + 1/1736 = 280/1736 = 5/31. Yes.
    # Max is 1736.
    
    # Can we do better than 1736?
    # Let's start with [8, 28, 1736] as a primary seed.
    seed_solution = [8, 28, 1736]
    if is_valid(seed_solution):
        initial_solutions.append(seed_solution)
        
    if not initial_solutions:
        # Fallback to standard greedy if seed fails (it shouldn't)
        return [8, 28, 1736]

    best_max = min(max(s) for s in initial_solutions)
    best_solution = min(initial_solutions, key=lambda s: max(s))
    
    # Local Search: Simulated Annealing / Hill Climbing
    # Moves:
    # 1. Split: Replace 1/d with 1/(d+1) + 1/(d(d+1)). Increases term count, might increase max denom.
    #    Actually, we want to DECREASE max denom.
    #    Splitting 1/d into 1/a + 1/b where a,b > d usually increases max denom.
    #    However, we can split 1/d into 1/x + 1/y such that 1/x + 1/y = 1/d.
    #    (x-1)(y-1) = d. So x = k+1, y = d/k + 1.
    #    This replaces one term with two. Max denom becomes max(k+1, d/k+1).
    #    If d is large, we might reduce the max if the current max is d and k+1, d/k+1 are smaller.
    #    But d/k+1 is at least 2. k+1 is at least 2.
    #    If d is the current max, and we split it, the new max is max(k+1, d/k+1).
    #    To reduce max, we need max(k+1, d/k+1) < d. This is always true for d > 4.
    #    So splitting a large denominator can reduce the max denominator!
    
    # 2. Merge: Replace 1/a + 1/b with 1/c if 1/a + 1/b = 1/c.
    #    1/c = (a+b)/(ab) => c = ab/(a+b). This requires a+b to divide ab.
    #    This reduces term count and might reduce max denom if c < max(a,b).
    #    But c = ab/(a+b) < a and c < b is always true for positive integers.
    #    So merging always reduces the max denominator (or keeps it same if a,b < c, which is impossible).
    #    Wait, c is smaller than both a and b. So merging reduces the max denom.
    #    Why don't we just merge everything? Because we need distinct denominators.
    #    Merging might create a denominator that already exists in the list.
    
    # Strategy:
    # We have a set of denominators.
    # We try to find moves that reduce the maximum denominator.
    # Since merging always reduces max denom (if valid and distinct), we should prioritize merging.
    # Splitting increases term count but can reduce max denom of a specific large term.
    
    # Let's use a stochastic local search.
    # State: list of denominators.
    # Cost: max(denoms).
    
    current_solution = best_solution[:]
    current_max = best_max
    
    # Precompute some properties
    def get_max(denoms):
        return max(denoms)
    
    # To speed up, we can track the sum as a Fraction.
    # But checking validity is expensive.
    # Let's just try moves and check validity.
    
    # We will perform random restarts and local searches.
    
    def local_search(start_denoms, time_limit):
        nonlocal best_max, best_solution
        denoms = start_denoms[:]
        curr_max = max(denoms)
        # Keep track of denominators in a set for O(1) distinct check
        d_set = set(denoms)
        
        # We need to be careful with Fraction arithmetic.
        # Let's represent the remaining sum? No, the sum is fixed to target.
        # Any valid move preserves the sum.
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try all possible merges
            # Merge 1/a + 1/b -> 1/c
            # c = ab/(a+b)
            # We iterate over pairs. O(n^2). n <= 256. 256^2 = 65536.
            # This is fast enough per iteration if we do a few iterations.
            
            found_merge = False
            # Sort denominators to make it easier to break? No, we need to check all pairs.
            # But if we find a merge that reduces max, we take it.
            
            # To find the best merge, we can just look for any valid merge that reduces max.
            # Since any merge reduces max (c < a and c < b), we just need one that is valid (distinct).
            
            # Optimization: Only consider pairs where max(a,b) == curr_max?
            # Not necessarily. Merging two small ones might free up a slot? No, we minimize max.
            # Merging reduces the max of the pair. If the global max is M, and we merge a,b < M,
            # the global max remains M. So it doesn't help reduce the global max immediately.
            # However, it might enable future merges?
            # Actually, if we merge a,b into c, and c is distinct, the new set has max M.
            # It doesn't reduce the objective function (max denom).
            # So we should ONLY merge if one of a or b is the current maximum.
            
            # Let's check pairs involving the current maximum.
            max_den = curr_max
            candidates = [d for d in denoms if d == max_den]
            # Since distinct, there is at most one max_den.
            if candidates:
                max_d = candidates[0]
                for other_d in denoms:
                    if other_d == max_d:
                        continue
                    a, b = max_d, other_d
                    # Check if 1/a + 1/b = 1/c for integer c
                    # c = ab / (a+b)
                    numerator = a * b
                    denominator = a + b
                    if numerator % denominator == 0:
                        c = numerator // denominator
                        if c not in d_set:
                            # Valid merge!
                            # Update state
                            denoms.remove(max_d)
                            denoms.remove(other_d)
                            denoms.append(c)
                            d_set.remove(max_d)
                            d_set.remove(other_d)
                            d_set.add(c)
                            curr_max = max(denoms)
                            improved = True
                            found_merge = True
                            if curr_max < best_max:
                                best_max = curr_max
                                best_solution = denoms[:]
                            break
            if found_merge:
                continue
            
            # If no merge involving the max, try splitting the max?
            # Split 1/d into 1/x + 1/y.
            # x = k+1, y = d/k + 1.
            # We want max(x, y) < d. This is always true.
            # But we need x, y distinct and not in d_set.
            # Also, we need to ensure the new max is lower than current max?
            # If we split the current max d, the new max is max(x, y, other_max).
            # If other_max < max(x,y), new max is max(x,y).
            # If other_max > max(x,y), new max is other_max.
            # Since we are splitting the global max d, and x,y < d,
            # the new global max will be max(x, y, second_max).
            # If second_max < d, the new max is max(x, y, second_max).
            # This could be lower than d.
            
            # Let's try to split the current max.
            if len(denoms) < 256:
                max_d = curr_max
                # Find divisors k of d such that k+1 and d/k+1 are valid
                # k must be a divisor of d.
                # k < d (so that y > 1)
                # We want to minimize the new max.
                # New max = max(k+1, d/k+1, second_max)
                # We can try all divisors k.
                
                # Get second max
                sorted_denoms = sorted(denoms, reverse=True)
                second_max = sorted_denoms[1] if len(sorted_denoms) > 1 else 0
                
                best_split_k = None
                best_new_max = curr_max
                
                # Iterate divisors of max_d
                # k from 1 to sqrt(max_d)
                limit = int(max_d**0.5)
                for k in range(1, limit + 1):
                    if max_d % k == 0:
                        # k is a divisor
                        x = k + 1
                        y = max_d // k + 1
                        # Check distinctness
                        if x == y:
                            continue
                        if x in d_set or y in d_set:
                            continue
                        # Also x, y must be distinct from each other (checked)
                        new_max_candidate = max(x, y, second_max)
                        if new_max_candidate < best_new_max:
                            best_new_max = new_max_candidate
                            best_split_k = (x, y)
                        
                        # Also check the other divisor
                        k2 = max_d // k
                        if k2 != k:
                            x2 = k2 + 1
                            y2 = k // k2 + 1 # Wait, y = d/k + 1. If k2 = d/k, then y2 = k + 1?
                            # Formula: x = k+1, y = d/k + 1.
                            # If we use k2 as the multiplier:
                            # x' = k2+1, y' = d/k2 + 1 = k + 1.
                            # So the pair is {k2+1, k+1}. Same set.
                            pass
                
                if best_split_k:
                    x, y = best_split_k
                    # Perform split
                    denoms.remove(max_d)
                    d_set.remove(max_d)
                    denoms.append(x)
                    denoms.append(y)
                    d_set.add(x)
                    d_set.add(y)
                    curr_max = best_new_max
                    improved = True
                    if curr_max < best_max:
                        best_max = curr_max
                        best_solution = denoms[:]
                    continue
            
            # If no simple greedy move works, try a random perturbation?
            # Or just break out of this local search phase.
            # To avoid getting stuck, we can try a random split of a non-max term?
            # But that might increase max.
            # Let's just break and restart from a different seed or randomize.
            break
        
        return denoms, curr_max

    # Run local search from initial solutions
    # We have limited time. Let's run local search on the best initial solution.
    # And maybe try a few random restarts if time permits.
    
    # First, run on the hardcoded good seed
    if time.time() < deadline:
        local_search(seed_solution, deadline - 5.0)
        
    # Try to find better seeds via a quick random search?
    # The space of Egyptian fractions is complex.
    # Let's try a few greedy variations.
    # Greedy with "skip"
    def greedy_with_skip(target_frac, skip_first=1):
        denoms = []
        rem = target_frac
        min_d = 1
        for _ in range(256):
            if rem == 0:
                break
            d = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
            # Try to skip some candidates to see if we get a smaller max
            # This is heuristic.
            for _ in range(skip_first):
                if Fraction(1, d) <= rem:
                    break
                d += 1
            denoms.append(d)
            rem -= Fraction(1, d)
            min_d = d + 1
        if rem == 0:
            return denoms
        return None

    # Try a few skip values
    for skip in range(0, 5):
        if time.time() > deadline - 10.0:
            break
        sol = greedy_with_skip(target, skip)
        if sol and is_valid(sol):
            max_val = max(sol)
            if max_val < best_max:
                best_max = max_val
                best_solution = sol
            # Run local search on this new seed
            local_search(sol, deadline - 5.0)
            
    # Final validation
    if best_solution and is_valid(best_solution):
        return best_solution
        
    # Fallback
    return [8, 28, 1736]
