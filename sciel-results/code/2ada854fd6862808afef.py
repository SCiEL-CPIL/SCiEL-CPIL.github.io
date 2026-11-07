import math
import time

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

def _lcm(a, b):
    return a * b // math.gcd(a, b)

# LCM of all divisors - useful for periodicity
LCM_ALL = 1
for d in DIVISORS:
    LCM_ALL = _lcm(LCM_ALL, d)

def _find_assignment(L, W):
    """
    Try to find distinct multiples of each divisor in [L, L+W].
    Returns list of chosen values (one per divisor, in order) or None.
    Uses backtracking with MRV heuristic.
    """
    R = L + W
    # For each divisor, compute candidate multiples in [L, R]
    candidates = []
    for d in DIVISORS:
        # First multiple >= L
        k_start = (L + d - 1) // d
        k_end = R // d
        if k_start > k_end:
            return None
        mults = list(range(k_start * d, k_end * d + 1, d))
        candidates.append(mults)
    
    # Order divisors by number of candidates (MRV)
    order = sorted(range(N), key=lambda i: len(candidates[i]))
    
    chosen = [None] * N
    used = set()
    
    def backtrack(idx):
        if idx == N:
            return True
        i = order[idx]
        for val in candidates[i]:
            if val not in used:
                used.add(val)
                chosen[i] = val
                if backtrack(idx + 1):
                    return True
                used.remove(val)
                chosen[i] = None
        return False
    
    if backtrack(0):
        return chosen
    return None

def _get_interesting_L_values(W):
    """
    Generate interesting L values for a given window width W.
    These are positions where the window boundaries align with multiples of divisors.
    """
    interesting = set()
    
    # For each divisor, the window [L, L+W] contains a multiple of d
    # when L is in [k*d - W, k*d] for some k
    # So interesting L values are near multiples of each divisor
    
    for d in DIVISORS:
        # Multiples of d that could be in a window of width W
        # For k from 1 to MAX_VAL//d
        k_min = max(1, (1 - W) // d + 1)  # k*d >= 1 - W => k >= (1-W)/d
        k_max = MAX_VAL // d
        
        # We only need to check a limited range
        # For each multiple m = k*d, interesting L values are in [m-W, m]
        # But we can't check all, so sample strategically
        
        # Check L values that are multiples of d, or offset by small amounts
        # Actually, let's just collect all multiples and their nearby positions
        for k in range(1, min(k_max + 1, 1000)):  # Limit to reasonable range
            m = k * d
            if m > MAX_VAL:
                break
            # L values where m is in [L, L+W]: L in [m-W, m]
            # Add m itself and m-W
            interesting.add(max(1, m))
            interesting.add(max(1, m - W))
            # Also add some offsets
            for offset in range(0, min(W + 1, 20)):
                interesting.add(max(1, m - offset))
    
    # Also add 1 and some small values
    for v in range(1, 50):
        interesting.add(v)
    
    return sorted(interesting)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    best_result = None
    best_width = float('inf')
    
    # Try increasing window widths
    # Minimum possible width is 9 (10 distinct integers)
    # But realistically much larger due to divisibility constraints
    
    for W in range(9, 500):
        if time.time() > deadline:
            break
        
        # Get interesting L values for this width
        L_values = _get_interesting_L_values(W)
        
        found = False
        for L in L_values:
            if time.time() > deadline:
                break
            if L > MAX_VAL - W:
                continue
            
            result = _find_assignment(L, W)
            if result is not None:
                # Verify distinctness and validity
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
            # We found a solution with this width, but maybe a smaller W exists?
            # No, we're iterating W from small to large, so first found is best
            break
    
    if best_result is None:
        # Fallback: try a broader search
        for W in range(9, 1000):
            if time.time() > deadline:
                break
            # Try L in steps
            for L in range(1, min(MAX_VAL - W + 1, 10000), 7):
                if time.time() > deadline:
                    break
                result = _find_assignment(L, W)
                if result is not None:
                    if len(set(result)) == N:
                        if all(isinstance(v, int) and v > 0 and v <= MAX_VAL for v in result):
                            if all(result[i] % DIVISORS[i] == 0 for i in range(N)):
                                width = max(result) - min(result)
                                if width < best_width:
                                    best_width = width
                                    best_result = result[:]
                                break
            if best_result is not None:
                break
    
    if best_result is None:
        # Last resort: use the divisors themselves
        best_result = DIVISORS[:]
    
    # Ensure return type is list of ints
    return [int(v) for v in best_result]
# EVOLVE-BLOCK-END
