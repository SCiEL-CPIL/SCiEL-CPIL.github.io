"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 55.0  # seconds, safety margin before 60s limit (evaluator kills at 90s, but 60s is task spec)

def _is_mean_of_subset(C, i):
    """
    Check if C[i] is the mean of any subset of at least 2 OTHER elements in C.
    Uses bitset DP on shifted sums.
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
    # Mask to keep bitset within 2 * max_sum + 1 bits
    mask = (1 << (2 * max_sum + 1)) - 1
    
    # dp[k] is a bitset where bit s is set if sum s is achievable with exactly k elements
    # We shift sums by offset, so sum 0 is at bit offset
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    for z in zs:
        # Iterate k downwards to avoid using the same element multiple times in one step
        # We are adding z to subsets of size k-1 to form subsets of size k
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
    # y must be in selected, y != a, y != x (x is not in selected yet, so y != x is automatic if y in selected)
    # Also y must be distinct from a.
    sel_set = set(selected)
    for a in selected:
        y = 2 * a - x
        if y in sel_set and y != a:
            return True
            
    return False

def _incremental_check(selected, x):
    """
    Check if x can be added to selected without violating the condition.
    1. x is not the mean of any subset of selected (size >= 2).
    2. No element a in selected is the mean of any subset of (selected \ {a}) U {x} (size >= 2).
       This reduces to checking if there is a subset T of (selected \ {a}) with size t >= 1
       such that sum(T) + x = (t+1)*a  =>  sum(T) = (t+1)*a - x.
    """
    n = len(selected)
    if n < 2:
        return True
        
    # Check 1: x is not the mean of any subset of selected
    # sum(S) = k*x for some S subset of selected, |S|=k >= 2
    # We can use a bitset DP for subset sums of selected.
    # However, building the full DP for selected every time is O(N^2 * WordSize).
    # Since N is small (< 48), this is acceptable.
    # Let's optimize by just checking pairs first? No, subsets of any size.
    
    # DP for selected
    max_sum_sel = sum(selected)
    dp_sel = [0] * (n + 1)
    dp_sel[0] = 1
    for y in selected:
        for k in range(n, 0, -1):
            if dp_sel[k-1]:
                dp_sel[k] |= (dp_sel[k-1] << y)
                
    for k in range(2, n + 1):
        target = k * x
        if target > max_sum_sel:
            break
        if dp_sel[k] & (1 << target):
            return False
            
    # Check 2: For each a in selected, check if a is mean of subset involving x
    # We need DP for selected \ {a}.
    # To avoid O(N) DP constructions, we can precompute prefix/suffix DPs?
    # Or just do it O(N^2) with bitsets. N=48, N^2=2304. Bitset ops are fast.
    for i, a in enumerate(selected):
        others = selected[:i] + selected[i+1:]
        m = len(others)
        if m < 1:
            continue
            
        dp_others = [0] * (m + 1)
        dp_others[0] = 1
        for y in others:
            for k in range(m, 0, -1):
                if dp_others[k-1]:
                    dp_others[k] |= (dp_others[k-1] << y)
                    
        max_sum_others = sum(others)
        # We need subset T of others, size t >= 1, such that sum(T) = (t+1)*a - x
        for t in range(1, m + 1):
            target = (t + 1) * a - x
            if target < 0:
                continue
            if target > max_sum_others:
                break
            if dp_others[t] & (1 << target):
                return False
                
    return True

def construct():
    start_time = time.time()
    best_set = []
    
    # Candidate pool: Numbers of form 2^i * 3^j * 5^k <= 512 are "smooth" and less likely to form arithmetic progressions or mean relations with other smooth numbers?
    # Actually, powers of 2 are a known valid set. 2^i * 3^j might be too dense.
    # Let's stick to the full range [1, 512] but use smart ordering.
    
    candidates = list(range(1, 513))
    
    # Generate orders
    orders = []
    
    # 1. Structured orders
    orders.append(candidates[:])       # Increasing
    orders.append(candidates[::-1])    # Decreasing
    
    # 2. "Spread out" orders: pick from top and bottom
    spread_order = []
    lo, hi = 1, 512
    while lo <= hi:
        spread_order.append(hi)
        if lo != hi:
            spread_order.append(lo)
        lo += 1
        hi -= 1
    orders.append(spread_order)
    
    # 3. Random orders
    rng = random.Random(42)
    # We will generate random orders on the fly or pre-generate a few.
    # Pre-generating 100 random orders.
    for _ in range(100):
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        orders.append(rand_order)
        
    # 4. Orders prioritizing numbers with few divisors (primes, powers of 2)
    # This is a heuristic: numbers with few additive relations might be safer.
    # Actually, let's just stick to the above.
    
    for order in orders:
        if time.time() - start_time > _DEADLINE:
            break
            
        selected = []
        selected_set = set()
        
        for x in order:
            if time.time() - start_time > _DEADLINE:
                break
                
            if x in selected_set:
                continue
                
            # Fast pair check first (very cheap)
            # Adapt _fast_pair_check logic for incremental addition
            valid = True
            # 1. Is x mean of pair in selected?
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    if selected[i] + selected[j] == 2 * x:
                        valid = False
                        break
                if not valid:
                    break
            
            # 2. Is any a in selected mean of pair (x, y)?
            if valid:
                for a in selected:
                    y = 2 * a - x
                    if y in selected_set and y != a:
                        valid = False
                        break
                        
            if not valid:
                continue
                
            # Full incremental check
            if _incremental_check(selected, x):
                selected.append(x)
                selected_set.add(x)
                
                if len(selected) > len(best_set):
                    best_set = selected[:]
                    
                if len(selected) >= 48:
                    break
                    
    # Final verification of best_set
    if best_set:
        if _full_check(best_set):
            # Fallback: return a known valid set
            # Powers of 2 are safe?
            return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        else:
            return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
