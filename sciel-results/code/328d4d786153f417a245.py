import math
import time

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

def _lcm(a, b):
    return a * b // math.gcd(a, b)

# LCM of all divisors
LCM_ALL = 1
for d in DIVISORS:
    LCM_ALL = _lcm(LCM_ALL, d)

def _solve_window(L, W):
    """
    Try to find distinct multiples of each divisor in [L, L+W] using MRV backtracking.
    Returns list of chosen values (one per divisor, in order) or None.
    """
    R = L + W
    
    # 1. Generate candidates for each divisor
    candidates = []
    for d in DIVISORS:
        k_start = (L + d - 1) // d
        k_end = R // d
        if k_start > k_end:
            return None
        mults = list(range(k_start * d, k_end * d + 1, d))
        candidates.append(mults)
    
    # 2. Sort divisors by number of candidates (MRV heuristic)
    # We need to keep track of original indices to return the correct order
    indices = list(range(N))
    indices.sort(key=lambda i: len(candidates[i]))
    
    # 3. Backtracking search
    used_values = set()
    assignment = [None] * N
    
    def backtrack(idx):
        if idx == len(indices):
            return True
        
        div_idx = indices[idx]
        div = DIVISORS[div_idx]
        cands = candidates[div_idx]
        
        # Try each candidate for this divisor
        for val in cands:
            if val not in used_values:
                used_values.add(val)
                assignment[div_idx] = val
                
                if backtrack(idx + 1):
                    return True
                
                used_values.remove(val)
                assignment[div_idx] = None
        
        return False
    
    if backtrack(0):
        return assignment
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    best_result = None
    best_width = float('inf')
    
    # We know the minimum span is at least 9 (10 distinct integers).
    # The LCM is 32730. We can iterate L in [1, LCM_ALL] for each W.
    # However, for small W, most L are invalid.
    
    # Optimization: Pre-compute "active" L ranges?
    # For a fixed W, the condition "d has a multiple in [L, L+W]" is true for
    # L in union of intervals [k*d - W, k*d] for all k.
    # We can just iterate L from 1 to LCM_ALL.
    
    # To speed up, we can jump L.
    # If L is invalid for some d, we can jump to the next L where d might be valid.
    # But checking all d is cheap.
    
    for W in range(9, 200):  # Upper bound on W is conservative
        if time.time() > deadline:
            break
            
        found = False
        # Iterate L within one period
        # We can step by 1.
        # To optimize, we can skip L where any divisor has no multiple.
        # Let's just iterate and let _solve_window fail fast.
        
        for L in range(1, LCM_ALL + 1):
            if time.time() > deadline:
                break
            
            # Quick check: does each divisor have at least one multiple in [L, L+W]?
            feasible = True
            for d in DIVISORS:
                # Next multiple of d >= L
                k = (L + d - 1) // d
                val = k * d
                if val > L + W:
                    feasible = False
                    break
            if not feasible:
                continue
            
            result = _solve_window(L, W)
            if result is not None:
                # Verify
                if len(set(result)) == N:
                    if all(isinstance(v, int) and v > 0 and v <= MAX_VAL for v in result):
                        if all(result[i] % DIVISORS[i] == 0 for i in range(N)):
                            width = max(result) - min(result)
                            if width < best_width:
                                best_width = width
                                best_result = result[:]
                            found = True
                            break
        
        if found:
            break
    
    if best_result is None:
        # Fallback: just take the first multiple for each
        # This won't be minimal but is valid
        best_result = []
        for d in DIVISORS:
            best_result.append(d)
        # Ensure distinctness? The fallback might fail distinctness.
        # But the main loop should find a solution.
        # A robust fallback:
        # Start with 1, 2, ... and adjust?
        # Let's just return the divisors themselves if they are distinct (they are)
        # 6, 10, 14, 15, 21, 22, 26, 33, 35, 39 are distinct.
        # Span is 39 - 6 = 33.
        
    return [int(v) for v in best_result]
# EVOLVE-BLOCK-END
