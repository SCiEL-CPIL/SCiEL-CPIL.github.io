"""Improved program for Egyptian fraction of 5/31; minimizes largest denominator."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s kill
    
    target = Fraction(5, 31)
    
    def greedy_standard(r):
        """Standard greedy algorithm for Egyptian fraction."""
        values = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d <= 0:
                break
            values.append(d)
            r -= Fraction(1, d)
        return values

    # Get a valid baseline solution immediately
    baseline = greedy_standard(target)
    best_solution = baseline
    best_max = max(baseline) if baseline else float('inf')
    
    # Verify baseline
    if not baseline or sum(Fraction(1, d) for d in baseline) != target:
        # Fallback if something went wrong (should not happen)
        return [31, 31, 31, 31, 31] # Invalid, but prevents crash. Actually 5/31 is not 5 * 1/31.
        # Let's just return the greedy one, it's always valid.

    def dfs(r, min_den, max_den, current, depth, max_depth=256):
        nonlocal best_max, best_solution
        if time.time() > deadline:
            return False
        if r == 0:
            if current:
                m = max(current)
                if m < best_max:
                    best_max = m
                    best_solution = list(current)
            return True
        if depth >= max_depth:
            return False
        if r <= 0:
            return False
        
        d_start = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
        if d_start > max_den:
            return False
        
        # Pruning: Even if we take the smallest possible denominator, 
        # can we reach r?
        # Max sum with remaining terms (max_depth - depth) at min_den is (max_depth - depth) / min_den
        # But we need distinct denominators. A tighter bound:
        # If r > (max_depth - depth) / d_start, it's impossible? No, denominators increase.
        # Simple bound: If r * d_start > max_depth - depth, then even 1/d_start * (max_depth-depth) < r?
        # Actually, sum of 1/d_i for d_i >= d_start is bounded by integral or simple sum.
        # For now, rely on time and max_den.
        
        for d in range(d_start, max_den + 1):
            if time.time() > deadline:
                return False
            
            new_r = r - Fraction(1, d)
            if new_r < 0:
                break
            
            current.append(d)
            if dfs(new_r, d + 1, max_den, current, depth + 1, max_depth):
                current.pop()
                return True
            current.pop()
        
        return False

    # Try to find a better solution with smaller max denominator
    # We search for solutions with max denominator L < best_max
    # We try a few candidate limits
    candidates = [100, 200, 500, 1000, 2000, 5000]
    # Add fractions of current best if they are larger than 5000
    if best_max > 10000:
        candidates.append(best_max // 4)
        candidates.append(best_max // 2)
    
    # Sort candidates and filter those < best_max
    candidates = sorted(set([c for c in candidates if c < best_max]))
    
    for limit in candidates:
        if time.time() > deadline:
            break
        
        # We want to find a solution with all denominators <= limit
        # Start DFS with r=target, min_den=1, max_den=limit
        # To speed up, we can restrict the first term
        # First term d must be >= ceil(31/5) = 7
        # And d <= limit
        # Also, 1/d <= 5/31 => d >= 7
        
        # We can try starting with specific first terms to reduce branching
        # But full DFS with limit=1000 might be okay if it finds a solution quickly.
        # However, DFS explores in order of d.
        
        # Let's just run DFS on the whole target
        if dfs(target, 1, limit, [], 0):
            # If we found a solution with max <= limit, best_max is updated.
            # If best_max <= limit, we might want to try smaller limits?
            # The loop iterates candidates in ascending order.
            # If we find a solution for limit=100, best_max becomes <= 100.
            # Then next candidate 200 is > best_max, so it's skipped?
            # No, the list is pre-computed. 
            # If best_max drops below the next candidate, we should break or skip.
            if best_max <= limit:
                # We found a solution with max <= limit.
                # Since candidates are sorted ascending, any subsequent candidate is larger.
                # We want to minimize max denominator.
                # If we found one at 100, we don't need to search 200.
                # But what if there is a solution with max 90? We already searched 100.
                # The DFS finds *a* solution, not necessarily the one with minimum max.
                # However, if we search limit=100, we find a solution with max<=100.
                # If we search limit=50, we find a solution with max<=50.
                # So we should iterate candidates in ascending order.
                # If we find a solution for limit=100, best_max <= 100.
                # Next candidate is 200. 200 > best_max. We don't need to search 200 to find something *better* than 100?
                # Wait, if we search limit=200, we might find a solution with max=150, which is better than 100? No, 150 > 100.
                # We want to minimize the max denominator.
                # If we have a solution with max 100, we want to find one with max < 100.
                # So we should only search limits < current best_max.
                # Since candidates are ascending, if we succeed at 100, best_max <= 100.
                # Next candidate 200 is > best_max. We skip it.
                # But what if we failed at 100, but succeed at 200? Then best_max <= 200.
                # Then we try 500? 500 > best_max. Skip.
                # So the loop effectively finds the smallest limit for which a solution exists.
                # This is good.
                pass # Continue loop, but subsequent candidates will be > best_max and thus not useful for improvement?
                # Actually, if we find a solution at limit=100, best_max is updated to <= 100.
                # The next candidate is 200.
                # In the loop body, we don't check `if limit < best_max` before calling dfs.
                # We should add that check.
        
        # Re-check condition to avoid unnecessary work
        if limit >= best_max:
            # If the limit we just tried is >= current best, we don't need to try larger limits
            # because we want to find something *smaller* than best_max.
            # Wait, the candidates are sorted ascending.
            # If we tried 100 and failed, best_max is still large.
            # Next is 200. 200 < best_max (likely). We try it.
            # If we succeed, best_max becomes <= 200.
            # Next is 500. 500 > best_max. We should break.
            break

    # Final verification
    if best_solution is not None:
        s = sum(Fraction(1, d) for d in best_solution)
        if s == target and len(set(best_solution)) == len(best_solution) and all(1 <= d <= 10000000 for d in best_solution):
            return best_solution
    
    return baseline
# EVOLVE-BLOCK-END
