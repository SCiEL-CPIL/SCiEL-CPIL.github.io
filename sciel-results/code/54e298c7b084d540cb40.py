"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 85.0  # seconds, safety margin before 90s limit


def _is_mean_of_subset(C, i):
    """
    Check if C[i] is the mean of any subset of at least 2 OTHER elements in C.
    Uses bitset DP for efficient checking.
    """
    a = C[i]
    others = C[:i] + C[i+1:]
    n = len(others)
    if n < 2:
        return False
    
    zs = [y - a for y in others]
    
    # Calculate max absolute sum for offset
    max_sum = sum(abs(z) for z in zs)
    if max_sum == 0:
        return False
        
    offset = max_sum
    mask = (1 << (2 * max_sum + 1)) - 1
    
    # dp[k] is a bitset where bit s is set if sum s is achievable with exactly k elements
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    for z in zs:
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if z >= 0:
                shifted = dp[k-1] << z
            else:
                shifted = dp[k-1] >> (-z)
            
            dp[k] |= (shifted & mask)
            
            # Early exit: if sum 0 is achievable with k >= 2 elements
            if k >= 2 and (dp[k] & (1 << offset)):
                return True
                
    # Final check for any k >= 2
    for k in range(2, n + 1):
        if dp[k] & (1 << offset):
            return True
            
    return False

def _full_check(C):
    """
    Check if any element in C is the mean of a subset of at least 2 other elements.
    Returns True if a violation is found.
    """
    for i in range(len(C)):
        if _is_mean_of_subset(C, i):
            return True
    return False

def _fast_pair_check(C, new_idx):
    """
    Quick check for 2-element subsets (pairs).
    1. Is C[new_idx] the mean of a pair in C[:new_idx]?
    2. Is any C[i] (i < new_idx) the mean of a pair involving C[new_idx] and some C[j]?
    """
    x = C[new_idx]
    selected = C[:new_idx]
    n_sel = len(selected)
    
    # 1. Check if x is mean of a pair in selected
    # 2x = a + b
    for i in range(n_sel):
        for j in range(i+1, n_sel):
            if selected[i] + selected[j] == 2 * x:
                return True
                
    # 2. Check if any a in selected is mean of pair (x, y) where y in selected, y != a
    # 2a = x + y  =>  y = 2a - x
    sel_set = set(selected)
    for a in selected:
        y = 2 * a - x
        if y in sel_set and y != a:
            return True
            
    return False

def _generate_sidon_like_candidates(n_max=512):
    """
    Generate a candidate list biased towards 'spread out' values which are less likely
    to form arithmetic progressions or have subset means.
    We use a simple heuristic: interleave large and small numbers, or use a specific
    pattern that avoids small differences.
    """
    candidates = list(range(1, n_max + 1))
    # Reorder to prioritize extremes
    # One strategy: take from ends
    lo, hi = 0, len(candidates) - 1
    reordered = []
    while lo <= hi:
        reordered.append(candidates[hi])
        if lo != hi:
            reordered.append(candidates[lo])
        lo += 1
        hi -= 1
    return reordered

def construct():
    start_time = time.time()
    best_set = []
    
    # Candidate pool
    candidates = list(range(1, 513))
    
    # Generate diverse orders
    orders = []
    
    # 1. Spread out order (extremes first)
    orders.append(_generate_sidon_like_candidates(512))
    
    # 2. Random orders
    rng = random.Random(42)
    num_random_orders = 500  # Adjust based on time budget
    
    # 3. Structured: Odds then Evens, or Primes (approx), or Coprime to 6
    orders.append([x for x in candidates if x % 2 != 0] + [x for x in candidates if x % 2 == 0])
    
    # 4. Powers of 2 first, then others? (Powers of 2 are very safe)
    pows = [2**k for k in range(9) if 2**k <= 512]
    rest = [x for x in candidates if x not in pows]
    orders.append(pows + rest)
    
    # Add random orders dynamically if time permits, or pre-generate a few
    for _ in range(10):
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        orders.append(rand_order)
        
    for order in orders:
        if time.time() - start_time > _DEADLINE - 5:
            break
            
        selected = []
        
        for x in order:
            if time.time() - start_time > _DEADLINE - 5:
                break
                
            # Fast pre-check: pairs
            # Construct C = selected + [x]
            # We can do the pair check on the fly
            
            # 1. Is x mean of pair in selected?
            valid = True
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    if selected[i] + selected[j] == 2 * x:
                        valid = False
                        break
                if not valid:
                    break
            
            # 2. Is any a in selected mean of pair (x, y)?
            if valid:
                sel_set = set(selected)
                for a in selected:
                    y = 2 * a - x
                    if y in sel_set and y != a:
                        valid = False
                        break
                        
            if not valid:
                continue
                
            # Full check for subsets of size >= 3
            C = selected + [x]
            
            # We only need to check if x is mean of subset of others (size>=3)
            # OR if any existing element is mean of subset involving x (size>=3)
            # _is_mean_of_subset(C, i) checks if C[i] is mean of subset of OTHERS.
            # So we need to check:
            # 1. x (new element, index -1 or len(selected)) is mean of subset of selected (size>=3)
            # 2. Any a in selected is mean of subset of (selected - {a} + {x}) with size>=3
            
            # Optimization:
            # Check new element x:
            if _is_mean_of_subset(C, len(selected)):
                continue
            
            # Check existing elements:
            # Only need to check if the subset includes x.
            # If a subset of selected (not including x) already made a mean, it would have been caught earlier.
            # So we only need to check subsets that include x.
            # This is equivalent to: is a the mean of a subset S of (selected - {a}) such that sum(S) = |S|*a - x?
            # Or simpler: just run _is_mean_of_subset for each existing element, but it's expensive.
            # However, N is small (<=48).
            # Let's just run the full check for all elements. It's O(N^2 * 2^N) worst case but bitset is fast.
            # Actually, _is_mean_of_subset is O(N * Range). Range is ~1000. N=48. 48*1000 = 48000 ops.
            # 48 elements * 48000 = 2.3M ops. This is fast in Python if optimized, but bitset is in C.
            
            violation = False
            for i in range(len(selected)):
                if _is_mean_of_subset(C, i):
                    violation = True
                    break
            
            if violation:
                continue
                
            selected.append(x)
            
            if len(selected) >= 48:
                break
                
        if len(selected) > len(best_set):
            best_set = selected[:]
            
    # Final verification of best_set
    if best_set:
        if _full_check(best_set):
            # Fallback
            return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        else:
            return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
