import time
from fractions import Fraction
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # seconds, leaving 10s safety margin for 90s kill

    target = Fraction(5, 31)

    # We want to find distinct denominators d1 < d2 < ... < dk such that
    # sum(1/di) = 5/31, and we want to minimize dk (the largest denominator).
    
    # Key insight: if we fix the largest denominator to be M, then we need
    # sum(1/di) = 5/31 where all di <= M and di are distinct.
    # 
    # Strategy: binary search / iterate on M (the max denominator), and for each M,
    # check if there exists a set of distinct denominators in [1, M] that sum to 5/31.
    # 
    # A more practical approach: use DFS with pruning.
    # 
    # The Egyptian fraction expansion of 5/31:
    # 5/31 = 1/8 + 1/424 + 1/... (greedy gives large terms)
    # 
    # Let's use a search that tries to find expansions with small max denominator.
    # We search for the minimum M such that 5/31 can be written as sum of distinct unit fractions with max denominator M.
    
    # Use a DFS approach:
    # State: (remaining_fraction, min_denominator, max_denominator_allowed)
    # We want to find if remaining_fraction can be expressed as sum of distinct unit fractions
    # where each denominator is in [min_denominator, max_denominator_allowed].
    
    # To minimize the maximum denominator, we can do a binary search on M, or
    # we can use a branch-and-bound that tracks the current maximum denominator.
    
    # Better approach: enumerate possible sets of small denominators and check if their
    # reciprocal sum equals 5/31.
    
    # Since 5/31 ≈ 0.16129, and we need distinct denominators, let's think about
    # what denominators are reasonable.
    
    # If all denominators are >= d_min, then we need at least ceil(1/d_min / (5/31)) terms... 
    # Actually, if max denominator is M, min denominator is at least 1 (but we want distinct).
    
    # Let me try a direct search:
    # For a given max denominator M, can we find distinct d1 < d2 < ... < dk <= M
    # such that sum(1/di) = 5/31?
    
    # This is a subset-sum-like problem in the rational domain.
    
    # Alternative: use the fact that 5/31 has a specific structure.
    # 5/31 = 1/a + 1/b + ... 
    
    # Let's try: pick some small denominators and see if the remainder can be completed.
    
    # I'll use a recursive search with memoization on (remainder, min_denom) pairs,
    # but limit the search depth and max denominator.
    
    # Key optimization: the denominator of the remainder fraction grows, so we
    # can bound the search.
    
    best_max_denom = None
    best_solution = None
    
    # Try to find solutions with increasing max denominator
    # First, let's establish an upper bound using greedy
    r = target
    greedy_sol = []
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        greedy_sol.append(d)
        r -= Fraction(1, d)
    greedy_max = max(greedy_sol) if greedy_sol else 10**7
    best_max_denom = greedy_max
    best_solution = greedy_sol[:]
    
    # Now try to improve: search for solutions with max denominator < greedy_max
    # We'll try M from some lower bound up to greedy_max - 1
    
    # Lower bound: since 5/31 ≈ 0.161, and 1/6 ≈ 0.167 > 5/31, we need at least
    # denominators that don't exceed 5/31 when summed. 
    # The smallest possible max denominator: we need sum of distinct 1/di = 5/31.
    # If we use denominators 8, 9, 10, ...: 1/8+1/9+1/10+... 
    # 1/8 = 0.125, remainder = 5/31 - 1/8 = (40-31)/248 = 9/248
    # 9/248 ≈ 0.0363, need 1/d ≈ 0.0363, d ≈ 27.5, so d >= 28
    # 1/8 + 1/28 = 7/56 + 2/56 = 9/56 ≈ 0.1607, close to 5/31 ≈ 0.16129
    # 5/31 - 1/8 - 1/28 = 5/31 - 9/56 = (280 - 279)/1736 = 1/1736
    # So 1/8 + 1/28 + 1/1736 = 5/31! Max denom = 1736.
    
    # Let's verify: 1/8 + 1/28 + 1/1736
    # LCM of 8, 28, 1736: 8=2^3, 28=2^2*7, 1736=8*217=8*7*31=2^3*7*31
    # LCM = 2^3*7*31 = 1736
    # 1736/8 + 1736/28 + 1736/1736 = 217 + 62 + 1 = 280
    # 280/1736 = 35/217 = 5/31. Yes!
    
    # So we have a solution with max denom 1736. Can we do better?
    
    # Let's search systematically for smaller max denominators.
    # We need distinct di such that sum(1/di) = 5/31 and max(di) = M.
    
    # For a fixed M, the number of terms is bounded. If min denom is a, max is M,
    # and we have k terms, then k/M <= 5/31, so k <= 5M/31.
    # Also 1/a + 1/(a+1) + ... >= k/a roughly.
    
    # Let's use a DFS that builds up the sum, trying denominators in increasing order,
    # and prunes when the sum exceeds 5/31 or when even adding all remaining
    # possible terms can't reach 5/31.
    
    def dfs(remaining, min_d, max_d, current):
        """
        remaining: Fraction, the amount still to be expressed
        min_d: smallest denominator allowed (to ensure distinctness and ordering)
        max_d: largest denominator allowed
        current: list of denominators chosen so far
        Returns: (success, solution) or (False, None)
        """
        nonlocal best_max_denom, best_solution, start_time
        
        if time.time() - start_time > DEADLINE:
            return False, None
        
        if remaining == 0:
            if current:
                m = max(current)
                if best_max_denom is None or m < best_max_denom:
                    best_max_denom = m
                    best_solution = current[:]
                return True, current[:]
            return False, None
        
        if remaining < 0:
            return False, None
        
        # Upper bound on how many more terms we can add: each term <= 1/min_d
        # So we need at most remaining * min_d terms... actually at most remaining / (1/max_d) = remaining * max_d terms
        # And at least... well, if remaining > (max_d - min_d + 1) / min_d, impossible
        
        # The maximum sum we can achieve with distinct denominators from min_d to max_d:
        # This is 1/min_d + 1/(min_d+1) + ... + 1/max_d
        # We can approximate: if this is less than remaining, prune
        
        # Quick check: the sum of 1/d for d from min_d to max_d
        # If max_d - min_d is small, compute directly; otherwise approximate
        span = max_d - min_d + 1
        if span > 0 and span <= 2000:
            max_possible = sum(Fraction(1, d) for d in range(min_d, max_d + 1))
        else:
            # Approximate with integral
            max_possible = Fraction(1, min_d - 1) if min_d > 1 else Fraction(1)
        
        if max_possible < remaining:
            return False, None
        
        # Minimum sum: just 1/max_d (one term)
        # If remaining > 1/min_d + 1/(min_d+1) + ... we can't reach it, already checked above
        
        # Try adding one more term
        # The next denominator d must satisfy: 1/d <= remaining, so d >= 1/remaining = remaining.denominator / remaining.numerator
        min_next = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        # Also d <= max_d
        if min_next > max_d:
            return False, None
        
        # Upper bound on d: we need 1/d >= remaining - (max possible sum of terms after d)
        # But that's complex. Let's just iterate.
        
        # To limit the search, we can note that if we use a term 1/d, the remaining
        # fraction is remaining - 1/d, and we need to express that with denominators in [d+1, max_d].
        
        for d in range(min_next, max_d + 1):
            if time.time() - start_time > DEADLINE:
                return False, None
            
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                # Since we're going in order, larger d gives smaller 1/d, so new_remaining will be less negative... 
                # wait no, larger d means smaller 1/d, so new_remaining is LARGER (less negative). 
                # Actually if 1/d > remaining, then new_remaining < 0. As d increases, 1/d decreases,
                # so new_remaining increases. So we should break when new_remaining >= 0? No,
                # we want new_remaining >= 0. If new_remaining < 0, we continue to larger d.
                continue
            
            if new_remaining == 0:
                if best_max_denom is None or d < best_max_denom:
                    best_max_denom = d
                    best_solution = current + [d]
                continue
            
            # Recurse
            success, sol = dfs(new_remaining, d + 1, max_d, current + [d])
            if success and sol:
                # Found a solution
                m = max(sol)
                if best_max_denom is None or m < best_max_denom:
                    best_max_denom = m
                    best_solution = sol
                return True, sol
        
        return False, None
    
    # Search with decreasing max_d to find the smallest possible
    # Start from a reasonable lower bound. We know 1/8 + 1/28 + 1/1736 works.
    # Let's try to find something better.
    
    # Lower bound on max denominator: 
    # We need sum of distinct 1/di = 5/31.
    # If max denominator is M, and we use denominators from some set, the sum is at most
    # the sum of the largest possible reciprocals.
    # The absolute maximum sum with max denominator M is 1/1 + 1/2 + ... but we can't use 1 since 1/1 > 5/31.
    # We can use denominators starting from some d_min.
    # The maximum sum with denominators in [a, M] is sum_{d=a}^{M} 1/d.
    # We need this to be >= 5/31.
    
    # For M = 30: max sum with denominators in [8, 30] = 1/8+1/9+...+1/30
    # Let's compute: this is roughly ln(30/8) + ... ≈ 1.322, which is way more than 5/31 ≈ 0.161.
    # So the constraint isn't tight on the upper sum.
    
    # The real constraint is that we need EXACTLY 5/31, not just at least.
    
    # Let's try a different approach: enumerate small sets.
    # Since 5/31 ≈ 0.161, reasonable denominators are around 6-100.
    
    # Try all pairs, triples, etc. of small denominators.
    
    # First, check if any single denominator works: 1/d = 5/31 => d = 31/5, not integer. No.
    
    # Check pairs: 1/a + 1/b = 5/31 => (a+b)/(ab) = 5/31 => 31(a+b) = 5ab => 31a + 31b = 5ab
    # => 5ab - 31a - 31b = 0 => (5a - 31)(5b - 31) = 31^2 = 961
    # 961 = 1 * 961 = 31 * 31
    # Case 1: 5a - 31 = 1, 5b - 31 = 961 => a = 32/5, not integer
    # Case 2: 5a - 31 = 31, 5b - 31 = 31 => a = 62/5, not integer
    # Case 3: 5a - 31 = 961, 5b - 31 = 1 => same as case 1
    # No integer solutions for pairs.
    
    # Check triples: 1/a + 1/b + 1/c = 5/31
    # This is more complex. Let's use the DFS above but with a tighter bound.
    
    # Let's try max_d values from 100 up, looking for any solution.
    # But we already know 1736 works. Let's try to find something smaller.
    
    # Run DFS with max_d = 200 first
    for max_d in [100, 150, 200, 300, 500, 800, 1000, 1200, 1400, 1600, 1735]:
        if time.time() - start_time > DEADLINE:
            break
        
        # Reset for this search
        found = False
        # Use a more targeted search
        def dfs_limited(remaining, min_d, max_d, current):
            nonlocal start_time
            
            if time.time() - start_time > DEADLINE:
                return None
            
            if remaining == 0:
                return current[:]
            
            if remaining < 0:
                return None
            
            # Prune: if even the max possible sum from min_d to max_d is less than remaining
            span = max_d - min_d + 1
            if span <= 1000:
                max_possible = sum(Fraction(1, d) for d in range(min_d, max_d + 1))
            else:
                if min_d > 1:
                    max_possible = Fraction(1, min_d - 1)
                else:
                    max_possible = Fraction(1)
            
            if max_possible < remaining:
                return None
            
            min_next = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
            
            if min_next > max_d:
                return None
            
            # Limit the number of terms: we can have at most (max_d - min_d + 1) more terms
            # but practically, the sum decreases quickly
            
            for d in range(min_next, max_d + 1):
                if time.time() - start_time > DEADLINE:
                    return None
                
                new_remaining = remaining - Fraction(1, d)
                if new_remaining < 0:
                    continue
                
                if new_remaining == 0:
                    return current + [d]
                
                result = dfs_limited(new_remaining, d + 1, max_d, current + [d])
                if result is not None:
                    return result
            
            return None
        
        solution = dfs_limited(target, 1, max_d, [])
        if solution is not None:
            m = max(solution)
            if best_max_denom is None or m < best_max_denom:
                best_max_denom = m
                best_solution = solution
            break  # Found a solution with this max_d, which is optimal for this max_d
    
    if best_solution is None or len(best_solution) > 256:
        # Fallback to greedy
        return greedy_sol
    
    # Ensure distinctness and validity
    assert len(set(best_solution)) == len(best_solution), "Denominators not distinct"
    assert all(1 <= d <= 10000000 for d in best_solution), "Denominator out of range"
    assert len(best_solution) <= 256, "Too many denominators"
    
    # Verify
    total = sum(Fraction(1, d) for d in best_solution)
    assert total == target, f"Sum {total} != {target}"
    
    return best_solution
