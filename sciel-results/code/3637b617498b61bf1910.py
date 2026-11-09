# EVOLVE-BLOCK-START
import math
import time

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
LCM_ALL = 30030

def _check_window(S, L):
    """
    Check if there is a valid assignment of distinct positive multiples
    in [S, S+L] for the given divisors.
    Returns the assignment list if found, else None.
    """
    # Generate candidates for each divisor
    candidates = []
    for d in DIVISORS:
        # First multiple >= S
        k_start = math.ceil(S / d)
        m = k_start * d
        mults = []
        while m <= S + L:
            if m > 0:  # Enforce positive
                mults.append(m)
            m += d
        if not mults:
            return None
        candidates.append(mults)
    
    # Backtracking with ordering by candidate count (heuristic)
    # Sort divisors by number of candidates (ascending) to prune early
    # But we must return the assignment in the original order.
    # So we keep track of the original index.
    order = sorted(range(len(DIVISORS)), key=lambda i: len(candidates[i]))
    
    assignment = [None] * len(DIVISORS)
    used = set()
    
    def backtrack(idx_in_order):
        if idx_in_order == len(order):
            return True
        orig_idx = order[idx_in_order]
        for m in candidates[orig_idx]:
            if m not in used:
                used.add(m)
                assignment[orig_idx] = m
                if backtrack(idx_in_order + 1):
                    return True
                used.remove(m)
                assignment[orig_idx] = None
        return False

    if backtrack(0):
        return assignment
    return None

def _find_optimal_window():
    # Binary search on L
    # Lower bound: 9 (10 distinct integers)
    # Upper bound: 100 (safe guess, since max divisor is 39, L=39 is always feasible for existence, distinctness might need more)
    # Actually, L=39 guarantees existence of multiples. Distinctness is likely satisfied for L>=39.
    # Let's set high to 100 to be safe.
    
    lo = 9
    hi = 100
    
    best_assignment = None
    best_L = None
    
    # First, verify hi is feasible
    # We scan S for hi
    # To speed up, we can just find one valid S for hi.
    
    # Binary Search
    while lo < hi:
        mid = (lo + hi) // 2
        found = False
        # Scan S for mid
        # Optimization: Only check S up to LCM
        for S in range(0, LCM_ALL):
            if S + mid > MAX_VAL:
                continue
            res = _check_window(S, mid)
            if res:
                found = True
                best_assignment = res
                best_L = mid
                break
        if found:
            hi = mid
        else:
            lo = mid + 1
            
    # If lo == hi, check it
    if not found or best_L != lo:
        # We need to check lo if it wasn't the one that set 'found' in the last step
        # Actually, if lo==hi, we need to ensure it's valid
        # The loop exits when lo==hi.
        # If 'found' was True in the last iteration, best_L is set.
        # If 'found' was False, lo was incremented.
        # So if lo==hi, we should check if we have a valid assignment for lo.
        # The logic above sets best_assignment when found.
        # If the first found was at hi=100, best_L=100.
        # Then we search lower.
        # If we find a lower one, we update.
        # If we don't find a lower one, best_L remains the first found?
        # No, if we don't find a lower one, the loop ends with lo=hi=previous_hi.
        # But best_assignment might be from a higher L?
        # We should store the best (smallest) L found.
        
        pass # The logic above updates best_L only when found.
             # Since we search downwards (hi=mid), the first found is the highest L.
             # We want the SMALLEST L.
             # So we should search from low to high?
             # No, binary search works for "exists".
             # If exists(mid) is True, we search [lo, mid].
             # If exists(mid) is False, we search [mid+1, hi].
             # This finds the smallest L such that exists(L) is True.
             # So best_L will be the smallest L.
             # But we need the assignment for that L.
             # The assignment found for 'mid' is valid for 'mid'.
             # So best_assignment corresponds to best_L.
             
    return best_assignment, best_L

_optimal_solution = None

def _get_solution():
    global _optimal_solution
    if _optimal_solution is None:
        _optimal_solution = _find_optimal_window()
    return _optimal_solution

def construct():
    sol, _ = _get_solution()
    if sol is None:
        # Fallback
        return DIVISORS[:]
    return [int(x) for x in sol]
# EVOLVE-BLOCK-END
