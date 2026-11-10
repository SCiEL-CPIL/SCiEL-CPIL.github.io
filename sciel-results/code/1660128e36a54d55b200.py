"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    deadline = time.time() + 85  # 85s safety margin out of 90s
    best_solution = None
    best_max_den = float('inf')
    
    def greedy_complete(r, min_d, used):
        """Fast greedy completion to get a valid incumbent."""
        if r <= 0:
            return []
        terms = []
        while r > 0:
            # d such that 1/d <= r -> d >= 1/r
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d < min_d:
                d = min_d
            while d in used:
                d += 1
            terms.append(d)
            r -= Fraction(1, d)
            min_d = d + 1
            used.add(d)
        return terms

    # Initial incumbent via greedy
    init_terms = greedy_complete(target, 1, set())
    if init_terms:
        best_solution = init_terms
        best_max_den = max(init_terms)
        
    # Branch and Bound DFS
    # State: (remaining_fraction, min_d_allowed, used_set, current_terms_list)
    # We search for denominators d >= min_d_allowed
    # Lower bound for next denominator d:
    # 1. 1/d <= r  => d >= ceil(1/r)
    # 2. We need to represent r with k terms (k is not fixed, but we assume we can use as many as needed up to 256)
    # Actually, to minimize the MAX denominator, we want to pick d as small as possible?
    # No. If we pick d small, 1/d is large, r becomes small, subsequent denominators might be large.
    # If we pick d large, 1/d is small, r stays large, subsequent denominators must be smaller? No, they must be larger.
    # Wait. Sum of reciprocals.
    # If we fix d1, r1 = r0 - 1/d1.
    # To minimize max(d1, d2, ...), we generally want denominators to be "balanced" or as small as possible overall.
    # However, the constraint is distinctness and increasing order (we can enforce d1 < d2 < ... to avoid permutations).
    # If we enforce d1 < d2 < ... < dk, then max is dk.
    # We want to minimize dk.
    
    def dfs(r, min_d, used, terms):
        nonlocal best_solution, best_max_den
        
        if time.time() > deadline:
            return
        
        if r == 0:
            if terms:
                current_max = max(terms)
                if current_max < best_max_den:
                    best_max_den = current_max
                    best_solution = terms[:]
            return
        
        if r < 0:
            return
            
        # Bound: 
        # The largest denominator in the final solution will be at least the largest in `terms` so far.
        # Also, the remaining fraction r must be represented by distinct denominators > min_d-1.
        # Let the next denominator be d.
        # We require 1/d <= r. So d >= ceil(1/r).
        # Also d >= min_d.
        # Let d_start = max(min_d, ceil(1/r)).
        
        # Additional pruning:
        # If we use k more terms, the maximum denominator D_max must satisfy:
        # r = 1/d1 + ... + 1/dk <= k / d_start (if d1...dk >= d_start)
        # This doesn't help bound D_max directly from above, but helps bound feasibility.
        
        # More useful bound:
        # The last term 1/dk is the smallest.
        # r >= 1/dk => dk >= ceil(1/r)
        # So the max denominator will be at least ceil(1/r) if we only used one more term.
        # But we might use more terms.
        
        # Let's try to bound the maximum possible denominator we *could* end up with if we are not careful.
        # Actually, the key insight for "Minimize Largest Denominator" is that we should prefer smaller denominators early?
        # If we pick a very small d, r decreases rapidly.
        # If we pick a large d, r decreases slowly.
        
        # Let's iterate d from d_start upwards.
        # But how far?
        # If d is too large, 1/d is small.
        # We know that in the optimal solution, the denominators are likely "tight".
        
        # Heuristic bound: 
        # If we pick d, the new remaining is r' = r - 1/d.
        # The max denominator of the rest will be at least ceil(1/r') (if 1 term) or larger.
        # So total max >= max(d, ceil(1/r')).
        # If max(d, ceil(1/r')) >= best_max_den, we can prune this branch?
        # Yes, because the final max will be >= that value.
        
        d_start = max(min_d, (r.denominator + r.numerator - 1) // r.numerator)
        
        # Upper limit for d in this branch:
        # If d > best_max_den, then the solution max is already > best_max_den. Prune.
        # So we only need to try d up to best_max_den - 1 (to strictly improve) or best_max_den (if we want to explore ties, but we want min, so < best).
        # Actually, if d == best_max_den, the rest must have max < best_max_den? No, distinct.
        # If we pick d, and d >= best_max_den, we cannot improve the max (unless we replace the current best max with something smaller, but d itself is part of the solution).
        # So if d >= best_max_den, this branch cannot yield a solution with max < best_max_den.
        # Since we only care about improving (or finding first), we can stop if d >= best_max_den.
        
        d_limit = best_max_den - 1
        if d_start > d_limit:
            return
            
        # Also, we need enough "capacity" for the remaining fraction.
        # Max sum of k terms with denominators >= d_start is roughly k/d_start.
        # Min sum is ...
        
        for d in range(d_start, d_limit + 1):
            if time.time() > deadline:
                return
                
            # Check distinctness
            if d in used:
                continue
                
            new_r = r - Fraction(1, d)
            if new_r < 0:
                break # Since d is increasing, 1/d is decreasing, so new_r will only get larger? 
                      # Wait. r - 1/d. As d increases, 1/d decreases. So r - 1/d INCREASES.
                      # So if new_r < 0, it means 1/d > r. 
                      # But we started d such that 1/d <= r. 
                      # So new_r should be >= 0.
                      # Let's double check.
                      # d_start = ceil(1/r). So 1/d_start <= r.
                      # So new_r >= 0 always.
                      # So `break` is incorrect logic for < 0. It should never happen.
                      # But if d is very small? No, d_start ensures 1/d <= r.
                      
            # Prune based on new lower bound
            # The next terms must be > d.
            # The smallest possible next term is d+1.
            # The max denominator of the remaining part will be at least ceil(1/new_r) if new_r > 0.
            # Let next_min_d = d + 1
            # Let lb_rest = ceil(1/new_r) if new_r > 0 else 0
            # Actually, if new_r == 0, we are done.
            
            if new_r == 0:
                if len(terms) + 1 <= 256:
                    new_terms = terms + [d]
                    cur_max = max(new_terms)
                    if cur_max < best_max_den:
                        best_max_den = cur_max
                        best_solution = new_terms[:]
                continue
            
            # Calculate lower bound for the max of the rest
            # The rest must sum to new_r using denominators > d.
            # The largest denominator in the rest, say D_max_rest, must satisfy:
            # new_r <= (number of terms) / (d+1) ... this is loose.
            # Better: The single largest term 1/D_max_rest is part of the sum.
            # So 1/D_max_rest <= new_r => D_max_rest >= ceil(1/new_r).
            # So the final max denominator will be at least max(d, ceil(1/new_r)).
            # If max(d, ceil(1/new_r)) >= best_max_den, we cannot improve.
            
            lb_next = (new_r.denominator + new_r.numerator - 1) // new_r.numerator
            if lb_next < d + 1:
                lb_next = d + 1 # Must be distinct and larger than d
            
            if max(d, lb_next) >= best_max_den:
                # Can this branch still improve?
                # The final max will be >= max(d, lb_next).
                # If this lower bound is already >= best_max_den, we can't get a strictly smaller max.
                continue
                
            # Additional pruning: 
            # Can we represent new_r with distinct denominators > d?
            # Max possible sum with denominators > d is infinite (if we allow infinite terms).
            # But we are limited to 256 terms.
            # Max sum with 256 terms, smallest being d+1, d+2, ...
            # This is a harmonic series sum.
            # If new_r is too large to be represented by 256 terms starting from d+1, prune.
            # Sum_{i=1}^{256} 1/(d+i)
            # If new_r > this sum, prune.
            # This check is expensive to compute every time. 
            # Approximation: 256 / (d+1) is an overestimate of the sum? 
            # No, 1/(d+1) + ... < 256/(d+1).
            # So if new_r > 256/(d+1), it's impossible.
            if new_r > Fraction(256, d + 1):
                continue
                
            # Recursive call
            used.add(d)
            dfs(new_r, d + 1, used, terms + [d])
            used.remove(d)

    # Start DFS
    # We can order the search to find good solutions quickly.
    # Starting with small d is usually good for minimizing the max?
    # Let's try.
    
    # To help the DFS find a good incumbent fast, we already have one from greedy.
    # We can also try a few specific manual starts to tighten the bound before full DFS.
    
    # For example, try starting with d=7, 8, 9... and completing greedily to update best_max_den.
    for start_d in range(7, 15):
        if start_d in best_solution:
             # Just in case, though greedy usually picks smallest
             pass
        r_rem = target - Fraction(1, start_d)
        if r_rem < 0:
            continue
        terms = [start_d]
        used = {start_d}
        comp = greedy_complete(r_rem, start_d + 1, used)
        if comp:
            full = terms + comp
            if len(full) <= 256:
                m = max(full)
                if m < best_max_den:
                    best_max_den = m
                    best_solution = full[:]

    # Now run DFS
    dfs(target, 1, set(), [])
    
    if best_solution is None:
        # Fallback
        r, values = target, []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
    
    return best_solution
# EVOLVE-BLOCK-END
