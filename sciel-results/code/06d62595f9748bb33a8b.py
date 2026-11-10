"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

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
    max_val = P['maximum']
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Fallback: The divisors themselves are always a valid solution.
    # Span is 39 - 6 = 33.
    best_ans = divisors[:]
    best_span = _span(best_ans)
    
    # Binary search for the minimum span S
    # Lower bound: Since we need 10 distinct integers, the span must be at least 9.
    # Upper bound: We start with the known solution span (33).
    low = 9
    high = best_span
    
    # We will search for the smallest S such that a valid window exists.
    # We iterate S from low to high.
    # However, binary search is better if the feasibility check is monotonic (it is: if a window of size S exists, 
    # a window of size S+1 might not necessarily exist in the same place, but the *existence* of ANY window 
    # of size S implies existence for larger S? 
    # Actually, if we can fit in span S, we can fit in span S+1 by just extending the window.
    # So feasibility is monotonic with respect to S.
    
    # Wait, is it strictly monotonic? 
    # If there is a solution with max-min <= S, then there is a solution with max-min <= S+1.
    # Yes. So we can binary search on S.
    
    # However, finding *a* solution for a specific S might be hard if S is too small.
    # Let's just iterate S from low upwards until we find one. 
    # Given the small search space, linear iteration from 9 upwards is likely very fast.
    # If we find a solution at S=9, we stop. If not, try 10, etc.
    # This is safer than binary search if the "feasibility" check is expensive and we want the *first* valid one.
    # But binary search finds the *minimum* S faster if the check is cheap.
    # Let's use a hybrid: Check small S linearly first (9 to 15), then binary search if not found?
    # Or just linear search from 9 to 33. It's only 25 iterations.
    
    for S in range(low, high + 1):
        if time.time() > deadline:
            break
            
        # Generate candidate window starts L
        # L must be such that [L, L+S] contains multiples for all divisors.
        # Critical Ls are multiples of divisors, or multiples - S.
        critical_Ls = set()
        
        # We only need to consider L such that at least one divisor has a multiple in [L, L+S]
        # that is "close" to the boundaries or is the only option.
        # A sufficient set of candidates:
        # For each divisor d, for each multiple m = k*d:
        # If m is in [L, L+S], then L is in [m-S, m].
        # So any valid L must be in the union of intervals [m-S, m] for all multiples m.
        # The "critical" points are the endpoints of these intervals.
        
        # To keep the set small, we limit the range of multiples we consider.
        # Since S is small (< 33 initially), the window is narrow.
        # We can just iterate L from 1 to some limit? No, L can be up to 1e6.
        
        # Better approach:
        # Iterate through possible "anchors".
        # Pick the divisor with the largest value (39) or smallest count of multiples in a typical window?
        # Actually, let's just generate all L that are "aligned" with the grid of divisors.
        
        # Let's collect all potential L values.
        # For each divisor d:
        #   For each multiple m of d:
        #     L can be m, m-1, ..., m-S (if >=1)
        # This is too many.
        
        # Optimized Candidate Generation:
        # We only need to check L values where the set of available candidates for some divisor changes.
        # This happens when L crosses a multiple of a divisor.
        # So L = k*d and L = k*d - S are the critical points.
        
        # Let's gather these.
        # To avoid huge memory, we limit k.
        # Since we are looking for small S, the values won't be huge? 
        # Actually, the optimal solution might be at any magnitude.
        # However, due to periodicity (LCM), the pattern repeats.
        # LCM of divisors is large.
        
        # Heuristic: Check L in a reasonable range, e.g., 1 to 10000.
        # If the optimal span is small, the values are likely small? Not necessarily.
        # But if we find a solution at L=100, span S, we are done.
        
        # Let's try a randomized approach combined with critical points for robustness.
        # 1. Check critical Ls in a small range (1 to 5000).
        # 2. If not found, check random Ls.
        
        critical_Ls.clear()
        
        # Generate critical Ls for small multiples
        for d in divisors:
            # k from 1 to 5000//d
            for k in range(1, 5000 // d + 1):
                m = k * d
                # L can be m, m-S
                critical_Ls.add(m)
                critical_Ls.add(max(1, m - S))
                # Also neighbors
                critical_Ls.add(max(1, m - 1))
                critical_Ls.add(max(1, m - S - 1))
        
        # Also check a few random Ls
        for _ in range(10):
            critical_Ls.add(random.randint(1, 5000))
            
        # Sort to be deterministic? Not necessary for correctness, but good for debugging.
        # We just need to find ONE valid L.
        
        for L in critical_Ls:
            if time.time() > deadline:
                break
                
            if L < 1:
                continue
            if L + S > max_val:
                continue
                
            # Check feasibility for window [L, L+S]
            # For each divisor i, find valid multiples in [L, L+S]
            options = []
            valid = True
            
            for i, d in enumerate(divisors):
                # Smallest multiple >= L
                k_min = (L + d - 1) // d
                # Largest multiple <= L+S
                k_max = (L + S) // d
                
                if k_min > k_max:
                    valid = False
                    break
                    
                # Generate list of values
                vals = [k * d for k in range(k_min, k_max + 1)]
                options.append((i, d, vals))
            
            if not valid:
                continue
                
            # Sort options by length (most constrained first)
            options.sort(key=lambda x: len(x[2]))
            
            # Backtracking search
            assigned = [None] * n
            used_values = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                orig_idx, d, vals = options[idx]
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
                # Found a valid solution
                ans = [int(v) for v in assigned]
                if _is_valid_answer(ans, divisors):
                    # Since we iterate S from low to high, the first one found is optimal
                    return ans
                    # Update best just in case, though we return immediately
                    # best_ans = ans
                    # best_span = S
                    # return ans
    
    # If no better solution found, return the fallback
    return [int(v) for v in best_ans]

# EVOLVE-BLOCK-END
