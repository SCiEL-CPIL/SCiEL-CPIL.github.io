import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    target = Fraction(5, 31)
    
    # We know 1/7 + 1/56 + 1/1736 = 5/31 (max=1736)
    # Try to find solutions with smaller max denominator
    
    best_solution = None
    best_max = float('inf')
    
    def check_and_update(denoms):
        nonlocal best_solution, best_max
        if len(denoms) < 1 or len(denoms) > 256:
            return
        # Verify distinctness and range
        if len(set(denoms)) != len(denoms):
            return
        if any(d < 1 or d > 10000000 for d in denoms):
            return
        # Verify sum
        s = sum(Fraction(1, d) for d in denoms)
        if s != target:
            return
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_solution = list(denoms)
    
    # DFS with pruning: search for solutions with max_denom < current best
    # Use bounded DFS: pick denominators in increasing order, cap at max_allowed
    
    def dfs(remaining, min_d, max_allowed, current, depth=0, max_depth=6):
        nonlocal best_solution, best_max
        if time.time() > deadline:
            return
        if remaining == 0:
            if len(current) >= 1:
                check_and_update(current)
            return
        if remaining < 0:
            return
        if len(current) >= max_depth:
            return
        if len(current) >= 256:
            return
        
        # Lower bound on next denominator: 1/d <= remaining => d >= 1/remaining
        d_lo = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        if d_lo < min_d:
            d_lo = min_d
        if d_lo > max_allowed:
            return
        
        # Upper bound: we need at least (max_depth - len(current)) more terms
        # Each future term >= 1/(d_lo+1), so remaining must be achievable
        # Simple bound: 1/d < remaining => d > 1/remaining, already handled
        # Also: remaining = 1/d + rest, rest >= 0, so 1/d <= remaining => d >= ceil(1/remaining)
        # Upper bound on d: if we have k more terms to fill, each at most 1/(d+1)
        # So remaining <= 1/d + (k-1)*1/(d+1)... this is complex, use simpler bound
        # remaining = 1/d + rest, rest >= 0, so 1/d <= remaining
        # Also rest = remaining - 1/d must be expressible with (max_depth-len(current)-1) terms
        # Each term in rest <= 1/(d+1), so rest <= (max_depth-len(current)-1)/(d+1)
        # remaining - 1/d <= (max_depth-len(current)-1)/(d+1)
        
        k_remaining = max_depth - len(current) - 1  # terms after this one
        
        d_hi = min(max_allowed, int(remaining.denominator / max(1, remaining.numerator - 1)) + 1) if remaining.numerator > 1 else max_allowed
        # Better upper bound: 1/d >= remaining/(k_remaining+1) roughly
        # Actually: remaining = 1/d + rest, rest <= k_remaining * 1/(d+1)
        # So remaining <= 1/d + k_remaining/(d+1)
        # This gives a lower bound on d, not upper.
        # Upper bound: 1/d <= remaining => d >= 1/remaining (already have)
        # And: remaining - 1/d >= 0 => 1/d <= remaining
        # For upper bound on d: if d is too large, 1/d is too small to help
        # remaining = 1/d + rest, rest >= 0, so 1/d <= remaining
        # But also rest must be positive (if we need more terms) or zero
        # If k_remaining == 0, then remaining = 1/d, so d = 1/remaining exactly
        if k_remaining == 0:
            if remaining.numerator == 1:
                d_exact = remaining.denominator
                if d_exact >= d_lo and d_exact <= max_allowed:
                    dfs(Fraction(0), d_exact + 1, max_allowed, current + [d_exact], depth + 1, max_depth)
            return
        
        # Upper bound: 1/d must be at least remaining/(k_remaining+1) roughly
        # Actually: remaining = 1/d + rest, rest <= k_remaining * 1/(d+1)
        # So remaining <= 1/d + k_remaining/(d+1) = (d+1+k_remaining*d)/(d*(d+1)) = ((k_remaining+1)*d + 1)/(d*(d+1))
        # This is complex. Use: 1/d >= remaining - k_remaining/(d+1)
        # For practical purposes, iterate d from d_lo up to some reasonable limit
        d_upper = min(max_allowed, d_lo + 500)  # Limit search width per level
        
        for d in range(d_lo, d_upper + 1):
            if time.time() > deadline:
                return
            if d > max_allowed:
                break
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break
            # Prune: if new_remaining > 0, we need k_remaining more terms
            # Each term >= 1/(max_allowed), so new_remaining <= k_remaining/max_allowed
            if new_remaining > 0 and k_remaining > 0:
                if new_remaining > Fraction(k_remaining, max_allowed):
                    continue
            dfs(new_remaining, d + 1, max_allowed, current + [d], depth + 1, max_depth)
    
    # Search with progressively tighter max_allowed
    # Start with known solution max=1736, try to beat it
    # Try max_allowed from 1735 down to find better
    
    # First, do a broad DFS with max_allowed = 1735 to see if we can beat 1736
    # But that's expensive. Instead, try specific strategies:
    
    # Strategy 1: Try all triplets with max < 1736
    # 1/a + 1/b + 1/c = 5/31, a < b < c < 1736
    # a >= 7 (since 1/a <= 5/31 => a >= 31/5 = 6.2)
    
    found_better = False
    for a in range(7, 50):
        if time.time() > deadline:
            break
        ra = target - Fraction(1, a)
        if ra <= 0:
            continue
        # 1/b + 1/c = ra, b > a, c > b, c < 1736
        # b >= ceil(1/ra)
        b_lo = (ra.denominator + ra.numerator - 1) // ra.numerator
        if b_lo <= a:
            b_lo = a + 1
        b_hi = min(1735, 1000)  # Limit b search
        for b in range(b_lo, b_hi + 1):
            if time.time() > deadline:
                break
            rb = ra - Fraction(1, b)
            if rb <= 0:
                break
            if rb.numerator != 1:
                continue
            c = rb.denominator
            if c <= b:
                continue
            if c < 1736:
                # Found a triplet with max < 1736!
                check_and_update([a, b, c])
                found_better = True
    
    # Strategy 2: Try 4-term solutions with max < 1736
    # Use DFS with max_allowed = 1735, max_depth = 4
    if not found_better:
        dfs(target, 7, 1735, [], 0, 4)
    
    # Strategy 3: Try 5-term solutions with max < 1736
    if best_max >= 1736:
        if time.time() < deadline - 10:
            dfs(target, 7, 1735, [], 0, 5)
    
    # Strategy 4: If still at 1736, try to find 4-term with max < 1736 using different approach
    if best_max >= 1736 and time.time() < deadline - 5:
        # More aggressive DFS with wider search
        dfs(target, 7, 1735, [], 0, 6)
    
    # Fallback: ensure we have a valid solution
    if best_solution is None:
        # Known valid: 1/7 + 1/56 + 1/1736
        best_solution = [7, 56, 1736]
        best_max = 1736
    
    return best_solution
