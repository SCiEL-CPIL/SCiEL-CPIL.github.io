import time
from fractions import Fraction
import math

def construct():
    target = Fraction(5, 31)
    
    # Known solution from seed
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s kill

    def verify(solution):
        if not solution:
            return False
        if len(solution) < 1 or len(solution) > 256:
            return False
        if len(set(solution)) != len(solution):
            return False
        if any(d < 1 or d > 10000000 for d in solution):
            return False
        s = sum(Fraction(1, d) for d in solution)
        return s == target

    # 1. Exhaustive search for 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # This is fast and might find a better solution than the seed.
    # a < b < c
    # 1/a < 5/31 => a > 31/5 = 6.2 => a >= 7
    # 3/a > 5/31 => a < 93/5 = 18.6 => a <= 18
    for a in range(7, 19):
        rem1 = Fraction(5, 31) - Fraction(1, a)
        if rem1 <= 0:
            continue
        
        # 1/b < rem1 => b > 1/rem1
        # 1/b + 1/c = rem1, c > b => 1/c < 1/b => 2/b > rem1 => b < 2/rem1
        b_low = int(1 / rem1) + 1
        b_high = int(2 / rem1)
        b_low = max(b_low, a + 1)
        
        if b_low > b_high:
            continue
        
        for b in range(b_low, b_high + 1):
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            # c = 1 / rem2
            if rem2.denominator == 1:
                c = rem2.numerator
            else:
                if rem2.numerator % rem2.denominator != 0:
                    continue
                c = rem2.numerator // rem2.denominator
            
            if c <= b:
                continue
            if c < best_max_denom:
                best_solution = [a, b, c]
                best_max_denom = c

    # 2. If we found a better 3-term solution, we might still find a better 4-term solution.
    # But first, let's see if we can beat the current best with a 4-term search.
    # The current best is likely 434 (from [7, 62, 434] or similar) or 1736.
    # Let's try to find a 4-term solution with max < best_max_denom.
    
    # Dijkstra-like / BFS approach for 4-term solutions?
    # Or just a bounded DFS.
    
    # Let's try a DFS for 4-term solutions with max < best_max_denom.
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # a < b < c < d < best_max_denom
    
    # a < 4 * 31 / 5 = 24.8 => a <= 24
    # a > 31/5 = 6.2 => a >= 7
    
    def dfs_4(remainder, min_denom, current_set, depth, max_denom_limit):
        nonlocal best_solution, best_max_denom
        
        if time.time() - start_time > time_limit:
            return False
            
        if depth == 4:
            if remainder == 0:
                # Found a solution
                # The max denominator is the last one added (since sorted)
                # But we are constrained by max_denom_limit which is < best_max_denom
                # So this is an improvement.
                best_solution = list(current_set)
                best_max_denom = max(current_set)
                return True
            return False
            
        if remainder <= 0:
            return False
            
        # Lower bound for next denominator d
        # 1/d <= remainder => d >= 1/remainder
        lower_bound = (1 / remainder).numerator // (1 / remainder).denominator
        if lower_bound < min_denom + 1:
            lower_bound = min_denom + 1
            
        # Upper bound is max_denom_limit - 1 (since we need distinct and increasing, and max is < limit)
        # Actually, we just need d < best_max_denom.
        # And we want to find ANY solution, so we can stop at the first one found?
        # Yes, because any solution found will have max < best_max_denom, which is an improvement.
        # And since we are minimizing the max, finding one with max < current best is good.
        # We don't need the absolute minimum max among 4-term solutions if it's already < 3-term best.
        
        # Pruning: Can we complete the sum with (4 - depth) terms?
        # The maximum possible sum of (4-depth) terms with denominators > min_denom and < best_max_denom
        # is roughly sum(1/d for d in range(min_denom+1, best_max_denom)).
        # This is an upper bound on the sum we can add.
        # If this sum < remainder, we can prune.
        # Approximation: ln(best_max_denom) - ln(min_denom)
        # But this is a loose bound.
        # A tighter bound: The sum of reciprocals of the smallest (4-depth) integers > min_denom.
        # But we can just iterate.
        
        # To make it faster, we can bound d by the fact that the remaining sum must be achievable.
        # If we pick d, the new remainder is remainder - 1/d.
        # The remaining terms must sum to remainder - 1/d.
        
        # Let's just iterate d.
        # d goes from lower_bound to best_max_denom - 1.
        # But if best_max_denom is large (e.g. 1736), this loop is big.
        # However, a is small (7..24).
        # Let's try.
        
        for d in range(lower_bound, best_max_denom):
            if time.time() - start_time > time_limit:
                return False
                
            next_rem = remainder - Fraction(1, d)
            if next_rem < 0:
                break # 1/d is decreasing as d increases, so next_rem increases.
                      # Wait. If next_rem < 0, it means 1/d > remainder.
                      # As d increases, 1/d decreases, so next_rem will become positive.
                      # So we should NOT break. We should continue.
                      # But if d is small, 1/d is large.
                      # If 1/d > remainder, we skip.
                      # Eventually 1/d < remainder.
                      # So we continue.
                      # However, if d is too small, we waste time.
                      # The lower_bound calculation ensures 1/d <= remainder.
                      # So next_rem should be >= 0.
                      # Let's double check.
                      # lower_bound = ceil(1/remainder).
                      # So 1/lower_bound <= remainder.
                      # So next_rem >= 0.
                      # So this break is never triggered if lower_bound is correct.
                      # But floating point errors? We use Fraction.
                      # So next_rem >= 0 is guaranteed.
                      # So we can remove the break or keep it as a safety.
                      # Actually, if next_rem == 0 and depth < 4, we can't stop, we need more terms?
                      # No, if remainder becomes 0, we have a solution.
                      # But we need exactly 4 terms?
                      # The problem allows 1..256 terms.
                      # So if we reach 0 with 3 terms, that's a valid solution.
                      # But we are in a 4-term search loop.
                      # If we find a 3-term solution inside the 4-term search, it's also valid.
                      # But we already did an exhaustive 3-term search.
                      # So we shouldn't find a new 3-term solution.
                      # Unless the 3-term search was incomplete?
                      # It was exhaustive for a in [7,18].
                      # So we can assume next_rem > 0.
                      
            if next_rem == 0:
                # This would mean 1/a + 1/b + 1/c = 5/31 (if depth was 3 before this step)
                # But we are looking for 4 terms.
                # If we find a solution with fewer terms, it's still valid.
                # But we already checked 3 terms.
                # So this case should not happen if the 3-term search was correct.
                # If it does, it's a solution with max d.
                # Is d < best_max_denom? Yes, by loop bound.
                # So it's an improvement.
                best_solution = list(current_set) + [d]
                best_max_denom = d
                return True
                
            # Pruning: Can we complete the rest?
            # Remaining terms: 4 - depth - 1 = 3 - depth
            # Min denominator for next: d
            # Max denominator: best_max_denom - 1
            # Max sum of (3-depth) terms with denominators > d and < best_max_denom.
            # If this max sum < next_rem, prune.
            
            # Approximation:
            # If next_rem is very small, it's easy to complete.
            # If next_rem is large, it's hard.
            
            # Let's just recurse.
            res = dfs_4(next_rem, d, current_set + [d], depth + 1, best_max_denom)
            if res:
                return True
                
        return False

    # Try 4-term search
    # We want to find a solution with max < best_max_denom.
    # So we pass best_max_denom as the limit.
    # a in [7, 24]
    for a in range(7, 25):
        if time.time() - start_time > time_limit:
            break
        rem1 = Fraction(5, 31) - Fraction(1, a)
        if rem1 <= 0:
            continue
            
        # Call DFS for the remaining 3 terms
        # min_denom = a
        # depth = 1 (we picked 'a')
        # We need 3 more terms.
        # The function dfs_4 expects to pick the NEXT term.
        # So we call it with remainder=rem1, min_denom=a, current_set=[a], depth=1.
        # It will pick b, c, d.
        if dfs_4(rem1, a, [a], 1, best_max_denom):
            # Found a solution with max < best_max_denom
            # best_solution and best_max_denom are updated
            # We can break because we found an improvement.
            # Is it the best possible?
            # Not necessarily. There might be a 4-term solution with even smaller max.
            # Or a 3-term solution with smaller max (but we already searched 3-term).
            # Or another 4-term solution with smaller max.
            # Since we iterate a upwards, and for each a, we iterate b upwards...
            # The first solution found might not have the smallest max.
            # However, the max is bounded by the loop limit.
            # To find the absolute minimum max for 4 terms, we would need to search more.
            # But for now, finding ANY solution with max < current best is a win.
            # We can try to continue searching to see if we can find a smaller max.
            # But the time limit is a concern.
            # Let's break for now and return the improvement.
            break

    if verify(best_solution):
        return best_solution
        
    # Fallback
    return [8, 28, 1736]
