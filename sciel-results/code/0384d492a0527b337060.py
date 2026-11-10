"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import product

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    best_span = float('inf')
    best_assignment = None
    
    # Precompute: for each divisor d_i, we need to pick a multiple k_i * d_i
    # such that all values are distinct and in [1, max_val]
    # We want to minimize max - min
    
    # Strategy: sliding window over possible L (lower bound)
    # For a given window [L, L+W], check if we can assign each divisor a distinct multiple in that range
    # Use backtracking with pruning
    
    # First, let's get a good initial upper bound on span
    # The divisors have LCM considerations. Let's try small windows first.
    
    # Key insight: the multiples are k_i * d_i. For a window [L, L+W],
    # divisor d_i has valid multiples: ceil(L/d_i)*d_i, (ceil(L/d_i)+1)*d_i, ..., floor((L+W)/d_i)*d_i
    # We need to pick one for each d_i, all distinct.
    
    # Let's search over W from small to large, and for each W, try different L positions.
    # But that's expensive. Instead, let's use a smarter approach.
    
    # Alternative: enumerate possible "anchor" values. Since we need distinct multiples,
    # the span is at least n-1 = 9 (for 10 distinct integers).
    
    # Let's try a direct search: for candidate span W, slide a window and try to fit.
    
    # Better approach: since the divisors are fixed and not too many, 
    # we can try to find tight clusters by checking small spans.
    
    # For each possible minimum value m (or rather, for each window position),
    # determine which multiples of each divisor fall in [m, m+W].
    
    # Let's be smart: the multiples of d in [L, L+W] are determined by 
    # k_start = ceil(L/d) and k_end = floor((L+W)/d).
    # Number of choices for divisor d is k_end - k_start + 1.
    
    # We'll iterate W from a lower bound upward, and for each W, 
    # try several L positions (e.g., L = 1, and L values near multiples of LCM-ish quantities).
    
    # Actually, let's just do a systematic search:
    # For W in range starting from a small value:
    #   For L in a set of candidate positions:
    #     Check if we can assign distinct multiples via backtracking
    
    # Candidate L positions: we can restrict L to be such that the window aligns well.
    # But to be safe, let's try L in steps. However, the range is up to 1000000,
    # so we can't try all L. 
    # 
    # Insight: the pattern of multiples repeats with period LCM of all divisors.
    # LCM of [6,10,14,15,21,22,26,33,35,39]:
    # 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    # LCM = 2 * 3 * 5 * 7 * 11 * 13 = 30030
    
    lcm_all = 30030
    
    # So the pattern repeats every 30030. We only need to check L in [1, 30030] 
    # (and possibly a bit more for edge effects, but since max is 1000000 >> 30030,
    # any window that fits in one period also fits elsewhere).
    
    # Actually, we need the window [L, L+W] to be within [1, 1000000].
    # And the pattern of available multiples repeats every 30030 in L.
    # So we can restrict L to [1, 30030 + W] or so, but to be safe, [1, 2*30030].
    
    def can_assign(L, W, divs):
        """Check if we can assign distinct multiples of each divisor in [L, L+W]."""
        n = len(divs)
        # For each divisor, compute the list of valid multiples in [L, L+W]
        choices = []
        for d in divs:
            k_start = (L + d - 1) // d  # ceil(L/d)
            k_end = (L + W) // d
            if k_start > k_end:
                return None
            # Valid multiples: k_start*d, (k_start+1)*d, ..., k_end*d
            # Store as list of values
            mults = [k * d for k in range(k_start, k_end + 1)]
            if not mults:
                return None
            choices.append(mults)
        
        # Backtracking: assign a multiple to each divisor, ensuring distinctness
        used = set()
        assignment = [None] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            for val in choices[idx]:
                if val not in used:
                    used.add(val)
                    assignment[idx] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
                    assignment[idx] = None
            return False
        
        if backtrack(0):
            return assignment[:]
        return None
    
    # Search over W from small to large
    # Lower bound on W: at least n-1 = 9 for 10 distinct integers
    # But due to divisibility constraints, it might be larger.
    
    # Let's start W from a reasonable lower bound.
    # The smallest possible span for 10 distinct positive integers is 9.
    # But with divisibility constraints, let's start from there.
    
    for W in range(9, 500):  # Try spans up to 500, should be enough
        if time.time() > deadline:
            break
        
        # Try L positions. Due to periodicity with period 30030,
        # we can restrict L to a range. But for small W, the window is small,
        # so the pattern within a period matters.
        # Let's try L in [1, min(30030, max_val - W)] with step 1 for small W,
        # but that's 30030 iterations per W, times up to 500 W values = 15M iterations.
        # Each can_assign is potentially expensive. Let's optimize.
        
        # Better: for a given W, the set of valid L positions is constrained.
        # Let's just try L in steps. For small W, we can try all L in [1, 30030].
        # For larger W, we can sample.
        
        if W <= 50:
            # Try all L in [1, 30030]
            for L in range(1, min(30030, max_val - W) + 1):
                if time.time() > deadline:
                    break
                result = can_assign(L, W, divisors)
                if result is not None:
                    span = max(result) - min(result)
                    if span < best_span:
                        best_span = span
                        best_assignment = result[:]
                    # Found a valid assignment with span <= W, but actual span might be < W
                    # We can break early for this W since we found one
                    break
        else:
            # Sample L positions
            step = max(1, 30030 // 100)
            for L in range(1, min(30030, max_val - W) + 1, step):
                if time.time() > deadline:
                    break
                result = can_assign(L, W, divisors)
                if result is not None:
                    span = max(result) - min(result)
                    if span < best_span:
                        best_span = span
                        best_assignment = result[:]
                    break
        
        # If we found a good solution, we can try to refine
        if best_span <= W and W > 15:
            # We might have a solution, let's try smaller W
            pass
    
    # If we found a solution, return it
    if best_assignment is not None:
        # Verify constraints
        for i, val in enumerate(best_assignment):
            if not isinstance(val, int) or isinstance(val, bool):
                continue
            if val < 1 or val > max_val:
                continue
            if val % divisors[i] != 0:
                continue
        # Check distinctness
        if len(set(best_assignment)) != len(best_assignment):
            # Fallback
            return [d for d in divisors]
        return list(best_assignment)
    
    # Fallback: return the divisors themselves (valid but not optimal)
    return list(divisors)
# EVOLVE-BLOCK-END
