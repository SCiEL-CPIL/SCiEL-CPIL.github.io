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
    
    # Strategy:
    # 1. Iterate L from 1 to a limit (e.g., 20000).
    # 2. For each L, determine the "tightest" window [L_min, L_max] that contains 
    #    at least one multiple of each divisor.
    #    Actually, we don't know the exact window in advance. 
    #    Instead, for a given L, we can define a candidate window [L, L+S] for various S.
    #    But a better heuristic: 
    #    For a fixed L, let's try to find the minimal S such that [L, L+S] has a solution.
    #    Or, even better: 
    #    Let's scan L. For each L, consider the set of multiples of each divisor that are "near" L.
    #    Specifically, for each divisor d, let m_d be the multiple of d closest to L.
    #    Let L_min = min(m_d) and L_max = max(m_d).
    #    The span is S = L_max - L_min.
    #    We check if there is a valid assignment within [L_min, L_max].
    #    If yes, update best_span.
    #    To ensure we don't miss solutions where the window is shifted slightly, 
    #    we can also check windows [L_min, L_max + delta] or [L_min - delta, L_max].
    #    But usually, the minimal span for a given alignment is determined by the extremes.
    
    # Optimization:
    # The pattern of multiples repeats with period LCM(divisors). 
    # LCM is huge, but the "local" structure repeats with period LCM of small subsets? No.
    # However, we only need to check L up to a certain limit. 
    # If a solution exists with span S, it exists in any window of length S.
    # The "phase" of the window relative to the divisors matters.
    # There are at most LCM phases, but we can sample L.
    # Let's scan L from 1 to 10000.
    
    def solve_window(L_start, L_end):
        """
        Given a window [L_start, L_end], find distinct multiples for all divisors.
        Returns list of values or None.
        """
        if L_start < 1 or L_end > P['maximum']:
            return None
            
        # Gather options for each divisor
        options = []
        for i, d in enumerate(divisors):
            k_min = (L_start + d - 1) // d
            k_max = L_end // d
            if k_min < 1:
                k_min = 1
            if k_min > k_max:
                return None
            vals = [k * d for k in range(k_min, k_max + 1)]
            options.append((i, d, vals))
        
        # Sort by number of options (MRV)
        options.sort(key=lambda x: len(x[2]))
        
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
            return assigned[:]
        return None

    # Main search loop
    # We scan L from 1 to 5000. 
    # For each L, we define a candidate window based on the closest multiples.
    
    for L in range(1, 5001):
        if time.time() > deadline:
            break
            
        # Calculate closest multiples to L for each divisor
        closest_mults = []
        for d in divisors:
            # Find multiple of d closest to L
            # k = round(L/d)
            k1 = L // d
            k2 = k1 + 1
            v1 = k1 * d
            v2 = k2 * d
            
            # Handle k1=0
            if v1 == 0:
                v1 = d # next multiple is d, but L>=1, so if L<d, v1=0 is invalid, use d? 
                # Actually if L < d, closest positive multiple is d.
                # If L >= d, closest is either floor or ceil.
            
            # Let's just find min |v - L| for v in {floor(L/d)*d, ceil(L/d)*d}
            # Ensure positive
            candidates = []
            if k1 >= 1:
                candidates.append(v1)
            candidates.append(v2)
            
            # Find closest
            closest = min(candidates, key=lambda x: abs(x - L))
            closest_mults.append(closest)
            
        L_min = min(closest_mults)
        L_max = max(closest_mults)
        
        # The span of this specific configuration is L_max - L_min
        current_span = L_max - L_min
        
        # If this span is not better than best, we might still need to check 
        # if a slightly larger window allows a valid assignment when this one doesn't?
        # No, if this window (which is the tightest containing one mult each) fails,
        # a larger window might succeed. But we are looking for MINIMAL span.
        # If the tightest window for this L fails, we should try expanding it?
        # Actually, if the tightest window fails due to collisions, 
        # maybe a different combination of multiples (not the closest ones) works with a larger span?
        # Yes. So we should check windows [L_min, L_min + S] for increasing S?
        
        # Efficient approach:
        # Check the tightest window first.
        if current_span < best_span:
            ans = solve_window(L_min, L_max)
            if ans is not None:
                if _is_valid_answer(ans, divisors):
                    best_span = current_span
                    best_ans = ans[:]
                    # Since we found a valid span, and we are scanning L, 
                    # we can't stop immediately because a different L might yield smaller span.
                    # But if best_span becomes 1, we can stop (min possible is 1? No, distinct ints, min span is 1? 
                    # e.g. 10, 11? No, must be divisible by specific divisors. 
                    # Min span is at least 1. If we find 1, it's optimal).
                    if best_span == 1:
                        break
        else:
            # Even if current_span >= best_span, it's possible that a valid assignment 
            # exists in a window of size < best_span that is NOT the tightest window for this L?
            # No, the tightest window for a given L is the smallest window containing 
            # *some* multiple for each divisor. If that window has size S_tight, 
            # any other valid window for this L must have size >= S_tight.
            # So if S_tight >= best_span, we can skip this L.
            pass
            
    # If we didn't find a better solution, return the initial one
    # Ensure it's a list of Python ints
    result = [int(v) for v in best_ans]
    return result

# EVOLVE-BLOCK-END
