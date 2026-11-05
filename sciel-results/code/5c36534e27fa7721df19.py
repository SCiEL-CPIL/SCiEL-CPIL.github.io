"""Combined program for Egyptian fraction of 5/31; minimizes largest denominator."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill (or 90s)
    
    target = Fraction(5, 31)
    
    best_max = float('inf')
    best_solution = None
    
    # Helper to check if a remainder can be completed greedily within bounds
    def can_complete_greedy(r, min_den, max_den, max_terms=256):
        """Try to complete r using greedy with bounds. Return list or None."""
        if r == 0:
            return []
        values = []
        used = set()
        for _ in range(max_terms):
            if time.time() > deadline:
                return None
            if r == 0:
                return values
            # Need 1/d <= r, so d >= ceil(1/r)
            d_min = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
            if d_min > max_den:
                return None
            d = d_min
            if d in used:
                # Greedy might pick a used denominator. In a strict Egyptian fraction,
                # denominators must be distinct. If greedy picks a used one, it fails
                # for this specific path, but we could try d+1. However, for speed,
                # we treat this as a failure for this specific greedy attempt.
                # A more robust approach would try d+1, but that complicates the 'greedy' nature.
                # Given the small search space, if greedy fails due to distinctness, 
                # the DFS or other branches might succeed.
                return None
            used.add(d)
            values.append(d)
            r -= Fraction(1, d)
            if r < 0:
                return None
            min_den = d + 1
        return values if r == 0 else None

    # 1. Try known good 3-term solutions first to establish a tight bound
    # 5/31 = 1/8 + 1/31 + 1/248 (Max 248)
    # 5/31 = 1/7 + 1/62 + 1/434 (Max 434)
    # Let's verify these
    candidates_3 = [
        [8, 31, 248],
        [7, 62, 434]
    ]
    for sol in candidates_3:
        if len(set(sol)) == len(sol):
            s = sum(Fraction(1, x) for x in sol)
            if s == target:
                mx = max(sol)
                if mx < best_max:
                    best_max = mx
                    best_solution = sol

    # 2. Search for 4-term solutions to potentially improve max denominator
    # We want to find a, b, c, d such that 1/a+1/b+1/c+1/d = 5/31
    # and max(a,b,c,d) < best_max.
    # Assume a < b < c < d.
    # a must be >= 7 (since 1/7 < 5/31).
    # If we fix a and b, we need 1/c + 1/d = R = 5/31 - 1/a - 1/b.
    # Then d = 1 / (R - 1/c).
    
    # We iterate a and b.
    # Bound for a: a < 1/(5/31 - 3/best_max) roughly? 
    # Simpler: just iterate a from 7 to some reasonable limit.
    # Since we want max < best_max, a must be < best_max.
    # Also 1/a < 5/31 => a > 6.2 => a >= 7.
    
    # Let's iterate a from 7 to min(30, best_max-1)
    # Why 30? If a is large, 1/a is small, remainder is large, requiring larger b,c,d.
    # But if a is small, 1/a is large, remainder is small.
    
    # Let's refine the search using the structure from Inspiration 1/2.
    # Iterate d1 (a) and d2 (b).
    # For each pair, calculate remainder R2.
    # Then we need to solve 1/c + 1/d = R2 with c >= b+1, d > c, d < best_max.
    # This is a 2-term Egyptian fraction problem for remainder R2.
    # 1/c + 1/d = R2 => d = 1 / (R2 - 1/c) = 1 / ((R2*c - 1)/c) = c / (R2*c - 1).
    # So c / (R2*c - 1) must be an integer.
    # Let R2 = p/q. Then d = c / ((p/q)*c - 1) = cq / (pc - q).
    # So (pc - q) must divide cq.
    # Since gcd(c, pc-q) = gcd(c, q), this is a number theoretic condition.
    
    # Alternative: Iterate c and check if d is integer.
    # c must satisfy: 1/c < R2 => c > 1/R2.
    # Also c >= b+1.
    # And d = c / (R2*c - 1). Since d > c, we need 1/(R2*c - 1) > 1/c => c > R2*c - 1 => 1 > c(R2 - 1).
    # Since R2 < 1, R2 - 1 is negative, so this is always true for positive c.
    # Wait, d > c means 1/d < 1/c.
    # d = 1 / (R2 - 1/c).
    # For d to be > c, we need 1/d < 1/c => R2 - 1/c < 1/c => R2 < 2/c => c < 2/R2.
    # So c is in range [ceil(1/R2), floor(2/R2) - 1].
    
    # This range might be small, making the search very fast.
    
    for a in range(7, min(30, best_max)):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        
        # Bounds for b
        # 1/b < r1 => b > 1/r1
        # b >= a+1
        b_min = max(a + 1, (r1.denominator + r1.numerator - 1) // r1.numerator)
        # b < best_max
        # Also, we need enough remainder for c and d.
        # Min sum of 1/c + 1/d with c >= b+1, d > c is roughly 1/(b+1) + 1/(b+2).
        # So r1 - 1/b > 1/(b+1) + 1/(b+2)
        # This gives an upper bound for b.
        
        for b in range(b_min, min(b_min + 100, best_max - 1)):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                continue
            
            # Now solve 1/c + 1/d = r2
            # c_min = max(b+1, ceil(1/r2))
            c_min = max(b + 1, (r2.denominator + r2.numerator - 1) // r2.numerator)
            # c_max: d > c implies c < 2/r2. So c_max = floor(2/r2) - 1
            c_max = (2 * r2.denominator) // r2.numerator - 1
            
            if c_max < c_min:
                continue
                
            # Also d must be < best_max.
            # d = 1 / (r2 - 1/c).
            # As c increases, r2 - 1/c increases, so d decreases.
            # So we need d < best_max => 1 / (r2 - 1/c) < best_max => r2 - 1/c > 1/best_max => 1/c < r2 - 1/best_max => c > 1 / (r2 - 1/best_max).
            # This gives another lower bound for c.
            
            for c in range(c_min, min(c_max + 1, best_max - 1)):
                if time.time() > deadline:
                    break
                rem = r2 - Fraction(1, c)
                if rem <= 0:
                    continue
                # d = 1 / rem
                if rem.denominator % rem.numerator != 0:
                    continue
                d = rem.denominator // rem.numerator
                if d <= c:
                    continue
                if d >= best_max:
                    continue
                if d > 10000000:
                    continue
                
                sol = [a, b, c, d]
                # Check distinctness
                if len(set(sol)) == 4:
                    # Verify sum
                    if sum(Fraction(1, x) for x in sol) == target:
                        mx = max(sol)
                        if mx < best_max:
                            best_max = mx
                            best_solution = sol
                            # If we found a very good one, we can tighten bounds or stop
                            if best_max < 50:
                                return best_solution

    # 3. Try 5-term solutions if 4-term didn't find anything better?
    # Given the 3-term solution has max 248, and 4-term search is fast, 
    # it's unlikely 5-term will beat 248 significantly unless 4-term fails completely.
    # But let's add a quick greedy completion for 2-term starts to be safe.
    
    # Try starting with a, then greedily complete with max_den = best_max - 1
    for a in range(7, min(20, best_max)):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        sol_rest = can_complete_greedy(r1, a + 1, best_max - 1)
        if sol_rest is not None:
            full_sol = [a] + sol_rest
            if len(full_sol) <= 256:
                m = max(full_sol)
                if m < best_max:
                    best_max = m
                    best_solution = full_sol

    # Fallback
    if best_solution is None:
        # Should not happen, but just in case
        r = target
        greedy_vals = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            greedy_vals.append(d)
            r -= Fraction(1, d)
        if r == 0 and len(greedy_vals) <= 256 and max(greedy_vals) <= 10000000:
            return greedy_vals
        # If all else fails, return the known 3-term solution
        return [8, 31, 248]

    return best_solution

# EVOLVE-BLOCK-END
