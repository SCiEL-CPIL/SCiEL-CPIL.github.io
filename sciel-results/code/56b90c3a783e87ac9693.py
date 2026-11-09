"""Rewritten program using DFS with branch and bound to minimize the largest denominator."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import sys
    sys.setrecursionlimit(10000)

    target = Fraction(5, 31)
    max_denom_limit = 10**7
    max_terms = 20  # Safe upper bound for number of terms
    
    best_solution = []
    best_max = max_denom_limit + 1

    def dfs(rem, min_d, current_terms, current_max):
        nonlocal best_solution, best_max
        
        if rem == 0:
            if current_max < best_max:
                best_max = current_max
                best_solution = current_terms[:]
            return True

        if len(current_terms) >= max_terms:
            return False
            
        if current_max >= best_max:
            return False

        # Lower bound for next denominator
        # 1/d <= rem  => d >= 1/rem
        L = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        
        # Upper bound for next denominator
        # We need rem - 1/d to be representable by (max_terms - len(current_terms)) terms
        # The maximum sum of k distinct denominators starting from d+1 is approx k/(d+1)
        # So rem - 1/d <= k / (d+1)
        # This is hard to solve exactly, but we know d cannot be too large.
        # A simple bound: 1/d must be less than rem.
        # Also, the remaining sum rem' = rem - 1/d must be > 0.
        # And rem' must be <= sum_{i=1}^k 1/(d+i).
        # Approximation: rem' <= k/(d+1)
        # rem - 1/d <= k/(d+1)
        # Let k = max_terms - len(current_terms)
        k = max_terms - len(current_terms)
        
        # We iterate d from L upwards.
        # We can stop if d becomes too large such that even with k terms, we can't reach rem.
        # Actually, if d is too large, 1/d is small, rem' is close to rem.
        # We need rem' <= sum_{i=1}^k 1/(d+i).
        # For large d, sum ~ k/d.
        # So rem <= k/d => d <= k/rem.
        U = min(max_denom_limit, (k * rem.denominator) // rem.numerator)
        
        # Also d must be <= best_max - 1 to improve
        if U >= best_max:
            U = best_max - 1
            
        if L > U:
            return False

        found = False
        for d in range(L, U + 1):
            next_rem = rem - Fraction(1, d)
            if next_rem < 0:
                break # Since d is increasing, 1/d is decreasing, so next_rem increases. 
                      # Wait. If 1/d > rem, next_rem < 0. 
                      # L is ceil(1/rem), so 1/L <= rem. 
                      # So for d >= L, 1/d <= 1/L <= rem. 
                      # So next_rem >= 0 always.
            
            if next_rem == 0:
                # Solution found
                if d < best_max:
                    best_max = d
                    best_solution = current_terms + [d]
                    return True # Can we find a better one? 
                                # Since we want to minimize max, and d is the last term,
                                # any other solution with same prefix would have max >= d.
                                # But maybe a different d' < d? No, d is increasing.
                                # Maybe a different prefix?
                                # We should continue searching to see if we can get a smaller max?
                                # No, current_max is max(prefix, d).
                                # If we found a solution with max M, we prune branches with max >= M.
                                # So returning True just stops this branch.
                                # The outer loop will try other d's?
                                # No, we return True to stop the current DFS call.
                                # The parent will try the next d.
                                # This is correct.
                
            # Prune if current_max is already too big
            new_max = max(current_max, d)
            if new_max >= best_max:
                continue
                
            # Recursive call
            if dfs(next_rem, d + 1, current_terms + [d], new_max):
                found = True
                # We don't break immediately because a smaller d might lead to a smaller max?
                # No, d is increasing. new_max is non-decreasing with d.
                # If we found a solution with d, any larger d will have new_max >= d.
                # But maybe the *other* terms in the sub-solution are smaller?
                # The max is max(prefix, d, sub_terms).
                # If we found a solution with max M, we want M' < M.
                # If we increase d, new_max increases. It's unlikely to beat the previous best.
                # However, it's possible that a larger d allows a "tighter" tail with smaller max?
                # Unlikely. Let's just continue to be safe, or break if we are confident.
                # Given the pruning, continuing is safe but slow.
                # Let's just return True to prune this branch quickly, 
                # as the parent will try other d values (smaller ones were already tried).
                # Wait, we are iterating d from L to U.
                # If we find a solution at d, we update best.
                # The next d is larger. new_max will be larger.
                # So it cannot beat the best_max we just set (unless the sub-solution max is much smaller? 
                # No, max is max(prefix, d, sub). d is larger. 
                # So we can break?
                # Not necessarily, because the sub-solution might have a very small max, 
                # but the max is dominated by d.
                # So if d >= best_max, we already pruned.
                # If d < best_max, we updated best_max to max(prefix, d, sub_max).
                # The next d' > d. new_max' >= d'.
                # If d' > best_max, it will be pruned.
                # So we can break?
                # Let's just return True to stop the search for this prefix.
                return True

        return found

    dfs(target, 1, [], 0)
    
    if best_solution:
        return best_solution
    else:
        # Fallback to greedy if DFS fails (should not happen)
        r, values = Fraction(5, 31), []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
# EVOLVE-BLOCK-END
