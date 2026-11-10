"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import itertools
from functools import lru_cache

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def _is_valid_answer(ans, divisors):
    """Check if answer is valid: 10 distinct positive ints <= 1e6, each divisible by corresponding divisor."""
    if not isinstance(ans, (list, tuple)):
        return False
    if len(ans) != len(divisors):
        return False
    seen = set()
    for i, (val, div) in enumerate(zip(ans, divisors)):
        if not isinstance(val, int) or isinstance(val, bool):
            return False
        if val < 1 or val > P['maximum']:
            return False
        if val % div != 0:
            return False
        if val in seen:
            return False
        seen.add(val)
    return True

def _span(ans):
    return max(ans) - min(ans)

def construct():
    """
    Find 10 distinct positive integers <= 1000000, aligned with divisors [6,10,14,15,21,22,26,33,35,39].
    Each representative must be divisible by its corresponding input.
    Minimize max(representatives) - min(representatives).
    """
    divisors = P['divisors']
    n = len(divisors)
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Fallback solution: the divisors themselves
    best_ans = divisors[:]
    best_span = _span(best_ans)
    
    # Strategy: Search for the minimum span S.
    # For a given span S, we need to find a window [L, L+S] that contains distinct multiples for all divisors.
    # We iterate S from 1 upwards.
    # For each S, we try a set of candidate L values.
    # To be efficient, we only need to check L values where the window alignment is "critical".
    # Specifically, if a solution exists for span S, it can be shifted. 
    # However, the set of valid multiples changes as L crosses a multiple of any divisor.
    # So, we should test L values that are:
    # 1. L = 1
    # 2. L = k*d - S (just before a multiple enters the window? No, we want the multiple inside)
    # 3. L = k*d (just after a multiple enters)
    # Actually, the set of available multiples for divisor d in [L, L+S] changes only when L passes a multiple of d.
    # So the "states" of the problem change at L = k*d - S and L = k*d.
    # We can iterate over all such critical L values for small S.
    
    # But for large S, there are too many critical points.
    # Let's start with small S and expand the search range.
    
    # Helper: Check if a specific window [L, L+S] has a valid assignment
    def check_window(L, S):
        if L < 1:
            L = 1
        if L + S > P['maximum']:
            return None
        
        # For each divisor, find the range of valid multipliers k such that L <= k*d <= L+S
        options = []
        for i, d in enumerate(divisors):
            k_min = (L + d - 1) // d  # ceil(L/d)
            k_max = (L + S) // d       # floor((L+S)/d)
            if k_min < 1:
                k_min = 1
            if k_min > k_max:
                return None
            # The actual values are k*d
            # We store the index, the divisor, and the list of possible values
            vals = [k * d for k in range(k_min, k_max + 1)]
            options.append((i, d, vals))
        
        # Sort by number of options (MRV heuristic)
        options.sort(key=lambda x: len(x[2]))
        
        assigned = [None] * n
        used_values = set()
        
        def backtrack(idx):
            if idx == n:
                return True
            
            orig_idx, d, vals = options[idx]
            
            # Try each value
            for v in vals:
                if v not in used_values:
                    used_values.add(v)
                    assigned[orig_idx] = v
                    if backtrack(idx + 1):
                        return True
                    used_values.remove(v)
                    assigned[orig_idx] = None
            return False
        
        if backtrack(0):
            return assigned[:]
        return None

    # Iterate over possible spans S
    # We know best_span is at most 39-6=33? No, max divisor 39, min 6.
    # Trivial span: 39 - 6 = 33 is not necessarily the min span.
    # Actually, the divisors themselves are 6, 10, 14, 15, 21, 22, 26, 33, 35, 39.
    # Span = 39 - 6 = 33.
    # Can we do better?
    # Let's try S from 1 up to best_span - 1.
    
    # To speed up, we can limit the range of L we check.
    # For a given S, L can be any integer such that 1 <= L and L+S <= 1e6.
    # However, due to periodicity (LCM of divisors), the pattern repeats every LCM.
    # LCM(6,10,14,15,21,22,26,33,35,39) is huge.
    # But for small S, the window is small, so the relative positions of multiples matter.
    # We can restrict L to a range like [1, max_divisor * (S + 1)]?
    # Actually, if a window [L, L+S] works, then [L + k*LCM, L + k*LCM + S] works.
    # So we only need to check L in [1, LCM]. But LCM is too big.
    
    # Alternative: Just check L in a reasonable range, say up to 1000 or 5000.
    # If S is small, the optimal L is likely small.
    
    # Let's refine the L search strategy:
    # For a fixed S, the set of valid values for divisor d depends on L mod d.
    # We can iterate L from 1 to some limit.
    
    # Let's try S from 1 to best_span - 1.
    # For each S, try L from 1 to limit.
    # Limit can be increased if S is small.
    
    current_best_span = best_span
    
    # We'll search S from 1 to current_best_span - 1
    # If we find a solution for S, we update best_span and continue to smaller S?
    # No, we are iterating S from 1 upwards. If we find a solution for S, that's the minimum.
    # So we can stop as soon as we find a valid S.
    
    # Wait, the loop `for S in range(1, current_best_span)` finds the *first* valid S.
    # Since S increases, the first valid S is the minimum span.
    # So we can return immediately upon finding the first valid S?
    # Yes, because we are searching for the *minimum* span.
    
    # However, we must ensure we check enough L values for each S to not miss a solution.
    # If we miss a solution for S=10, we might find one for S=11, but S=10 was possible.
    # So we need to be confident that our L search space is sufficient.
    
    # For small S, the number of multiples in the window is small.
    # The "state" of the window changes when L crosses a multiple of any divisor.
    # The number of such critical points in [1, X] is roughly sum(X/d).
    # For X=1000, this is manageable.
    
    # Let's try L from 1 to 2000 for small S, and maybe less for larger S?
    # Actually, if S is close to best_span (33), we might need to look further?
    # No, if S is 33, the window is large, so it's easier to find a solution.
    # The hard case is small S, where alignment is critical.
    # So for small S, we should check MORE L values.
    
    # Let's set L_limit based on S.
    # For S < 10, L_limit = 5000
    # For S < 20, L_limit = 2000
    # Else L_limit = 500
    
    for S in range(1, current_best_span):
        if time.time() > deadline:
            break
        
        # Determine L limit
        if S < 10:
            l_limit = 5000
        elif S < 20:
            l_limit = 2000
        else:
            l_limit = 500
            
        # Optimization: We don't need to check every L.
        # We only need to check L values where the set of available multiples changes.
        # This happens when L = k*d or L = k*d - S.
        # Let's generate these critical L values.
        
        critical_Ls = set()
        # Add 1
        critical_Ls.add(1)
        
        for d in divisors:
            # L such that L is a multiple of d
            # k*d <= l_limit => k <= l_limit/d
            for k in range(1, l_limit // d + 1):
                L = k * d
                if 1 <= L <= l_limit:
                    critical_Ls.add(L)
                # L such that L+S is a multiple of d?
                # i.e., L = k*d - S
                L2 = k * d - S
                if 1 <= L2 <= l_limit:
                    critical_Ls.add(L2)
        
        # Sort critical Ls
        critical_Ls = sorted(critical_Ls)
        
        # Also, it might be safer to just check all L in a small range for very small S?
        # No, critical points should suffice because the feasibility set is constant between critical points.
        # Wait, is it?
        # The set of multiples of d in [L, L+S] is {k*d | L <= k*d <= L+S}.
        # This set changes when L passes a multiple of d (a new multiple enters/exits) 
        # or when L+S passes a multiple of d (a multiple enters/exits).
        # So yes, the set is constant between critical points.
        # So checking one L from each interval is sufficient.
        # The critical points are the boundaries.
        # So checking L = critical_point and L = critical_point - 1 (or +1) should cover it.
        # Actually, checking the critical points themselves and maybe 1 after/before is safe.
        # Let's just check the critical points and a few neighbors.
        
        # To be safe, let's expand the set of L to check
        Ls_to_check = set()
        for L in critical_Ls:
            for delta in range(-2, 3):
                Lc = L + delta
                if 1 <= Lc <= l_limit:
                    Ls_to_check.add(Lc)
        
        # If the set is too small (e.g., for very large S where critical points are sparse?), 
        # we might miss something? No, critical points are dense for small d.
        # But for large S, the window is large, so it's likely to contain multiples.
        # The risk is for small S where alignment is tight.
        
        # Let's iterate through Ls_to_check
        for L in Ls_to_check:
            if time.time() > deadline:
                break
            
            ans = check_window(L, S)
            if ans is not None:
                # Found a valid solution with span S
                # Since we are iterating S from 1 upwards, this is the minimum span
                best_span = S
                best_ans = ans[:]
                # We can return immediately? 
                # The task says "Minimize max - min". 
                # If we found a valid S, and we started from 1, this is the global minimum.
                # So we can return.
                # However, we must ensure the solution is valid.
                if _is_valid_answer(best_ans, divisors):
                    return [int(v) for v in best_ans]
                else:
                    # Should not happen, but break to try next
                    break
        
        # If we didn't find a solution for this S, continue to next S
    
    # If no better span found, return the best known
    # Ensure it's a list of Python ints
    result = [int(v) for v in best_ans]
    return result

# EVOLVE-BLOCK-END
