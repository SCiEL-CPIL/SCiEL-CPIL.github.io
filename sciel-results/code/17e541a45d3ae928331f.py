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

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    best_result = None
    best_width = float('inf')
    
    # Try increasing window widths
    # Minimum possible width is 9 (10 distinct integers)
    # But realistically much larger due to divisibility constraints
    # Let's try widths from 9 upward
    
    # First, try to find any solution with a reasonable width
    # We'll sweep L for each width
    
    for W in range(9, 500):
        if time.time() > deadline:
            break
        # For this width, try different starting positions L
        # L can range from 1 to MAX_VAL - W
        # But we can be smarter: L only needs to be tried at positions
        # where at least one divisor's multiple boundary is near
        # For efficiency, try L in steps, but also key positions
        
        # Try a set of candidate L values
        # Key insight: the pattern repeats with period LCM_ALL, but LCM_ALL is large
        # Instead, just try a range of L values
        # To keep it fast, try L from 1 to some limit, stepping by 1 for small W
        # For larger W, we can step more
        
        found = False
        # Try L values - for small W, step by 1; for larger, step by gcd of divisors or 1
        # Actually, let's just try L from 1 upward, but limit the range
        # For width W, we need L such that for each d, there's at least one multiple in [L, L+W]
        # This means L <= k*d <= L+W for some k, i.e., L in [k*d - W, k*d]
        
        # Simple approach: try L from 1 to min(MAX_VAL - W, 10000)
        # If W is small, this is fast. If W is large, we may need to go higher.
        # But the optimal solution likely has small values.
        
        max_L_try = min(MAX_VAL - W, 20000)
        for L in range(1, max_L_try + 1):
            if time.time() > deadline:
                break
            result = _find_assignment(L, W)
            if result is not None:
                # Verify distinctness and validity
                if len(set(result)) == N:
                    if all(isinstance(v, int) and v > 0 and v <= MAX_VAL for v in result):
                        if all(result[i] % DIVISORS[i] == 0 for i in range(N)):
                            width = result[-1] if False else max(result) - min(result)
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
        # Fallback: use the divisors themselves (may not be optimal but valid)
        # Actually, divisors themselves might not be distinct or might exceed limits
        # Let's just return the divisors as a last resort
        best_result = DIVISORS[:]
    
    # Ensure return type is list of ints
    return [int(v) for v in best_result]
# EVOLVE-BLOCK-END
