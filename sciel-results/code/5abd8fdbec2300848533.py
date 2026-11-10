"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    start_time = time.time()
    # Deadline guard: 90s limit, leave 5s buffer for safety/return overhead
    deadline = start_time + 85.0

    best_solution = None
    best_max_den = 10_000_000  # Upper bound from problem statement

    def check_time():
        return time.time() >= deadline

    def dfs(remaining, min_den, max_den, used, depth, max_depth):
        """
        Depth-first search to find a valid Egyptian fraction expansion.
        
        Args:
            remaining: Fraction remaining to be decomposed.
            min_den: Minimum allowed denominator (must be > last used).
            max_den: Maximum allowed denominator (current bound M).
            used: Set of used denominators (for distinctness check, though min_den handles ordering).
            depth: Current depth.
            max_depth: Maximum allowed terms.
            
        Returns:
            List of denominators if a solution is found, else None.
        """
        if check_time():
            return None

        if remaining == 0:
            return []

        if depth >= max_depth:
            return None

        # Pruning 1: The remaining sum must be achievable by the max number of terms
        # using the smallest possible denominators (min_den, min_den+1, ...)
        # Max possible sum with (max_depth - depth) terms starting from min_den:
        # S_max = sum(1/d for d in range(min_den, min_den + (max_depth - depth)))
        # We approximate this with an integral or just check if remaining is too large.
        # A tighter bound: The largest possible sum of k distinct reciprocals starting from min_den
        # is roughly ln((min_den+k)/min_den) or sum directly.
        # For speed, we can check if remaining > sum(1/d for d in range(min_den, min_den + (max_depth - depth)))
        # But computing this sum every step is expensive. 
        # A simpler necessary condition: 
        # remaining <= (max_depth - depth) / min_den
        # And remaining >= 1 / (max_den * (max_depth - depth)) ? No, that's for smallest sum.
        
        # Stronger Pruning:
        # 1. remaining must be positive.
        if remaining <= 0:
            return None
            
        # 2. The largest possible sum of (max_depth - depth) terms starting from min_den
        #    is sum_{i=0}^{k-1} 1/(min_den+i). If remaining > this, impossible.
        #    We can approximate or calculate. Since max_depth <= 256, calculating is fast.
        terms_left = max_depth - depth
        if terms_left == 0:
            return None
            
        # Calculate max possible sum quickly
        # If min_den is small, this is significant.
        # For performance, we might skip the exact sum and use a loose bound first,
        # or just rely on the min_den constraint in the loop.
        
        # 3. The smallest possible sum of (max_depth - depth) terms ending at max_den
        #    is sum_{i=0}^{k-1} 1/(max_den-i). If remaining < this, impossible.
        #    This is a very strong pruning condition.
        #    Let's check if remaining is too small.
        #    Min sum = sum(1/d for d in range(max_den - terms_left + 1, max_den + 1))
        #    We can approximate: terms_left / max_den
        #    If remaining < terms_left / (max_den + terms_left):
        #        return None
        
        # Let's use the exact min-sum check for strong pruning when close to max_den
        # But it's O(terms_left). terms_left <= 256.
        # We will do a quick check first.
        if remaining < terms_left / (max_den + terms_left):
            return None

        # 4. Determine the range for the next denominator d.
        # d must be >= min_den.
        # d must satisfy 1/d <= remaining => d >= 1/remaining.
        lower_bound = max(min_den, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        # d must also be such that the remaining sum (remaining - 1/d) can be formed by the rest.
        # The rest must be positive, so 1/d < remaining => d > 1/remaining.
        # Also, the rest must be at least the sum of the smallest possible reciprocals for the remaining terms.
        # Let's iterate d from lower_bound to max_den.
        
        # Optimization: We can limit the upper bound of d.
        # If we pick d, the remaining sum is R' = remaining - 1/d.
        # R' must be >= min_sum(terms_left - 1, d+1, max_den)
        # And R' must be <= max_sum(terms_left - 1, d+1)
        
        # To keep it fast, we iterate d.
        # However, if max_den is large (e.g., 10000), iterating is slow.
        # But we binary search M. We hope to find a small M quickly.
        
        # Heuristic: Try smaller d first? Or larger?
        # For Egyptian fractions, smaller d's are usually "harder" to fit into the tail?
        # Actually, greedy chooses smallest d. 
        # To minimize max denominator, we are already bounding by M.
        # We just need *any* solution.
        
        # Let's try d from lower_bound upwards.
        # But if lower_bound is small and max_den is large, this is a huge loop.
        # We can prune d if even the *largest* possible sum of the remaining terms
        # (using the smallest available denominators d+1, d+2, ...) is less than R'.
        # Wait, R' = remaining - 1/d.
        # Max sum of remaining terms (terms_left-1) starting from d+1 is S_max(d+1).
        # We need R' <= S_max(d+1).
        # remaining - 1/d <= S_max(d+1)
        # 1/d >= remaining - S_max(d+1)
        
        # This is complex to compute dynamically.
        # Let's stick to a simpler, effective DFS with time limits.
        
        # To speed up, we can jump d.
        # If d is too small, R' is large, might be impossible to cover with small terms? No, small d -> large 1/d -> small R'.
        # If d is too large, R' is close to remaining, might be impossible to cover with large terms?
        
        # Let's just iterate. The binary search will keep M small if possible.
        
        for d in range(lower_bound, max_den + 1):
            if check_time():
                return None
                
            # Pruning: If 1/d >= remaining, we can't subtract it and keep positive (unless it's the last term)
            if Fraction(1, d) > remaining:
                break # Since d is increasing, 1/d is decreasing. Wait.
                # If 1/d > remaining, then remaining - 1/d < 0.
                # Since d increases, 1/d decreases. So for larger d, 1/d becomes smaller.
                # So we should NOT break. We should continue.
                # The condition 1/d <= remaining is d >= 1/remaining.
                # lower_bound already ensures this.
                
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                continue
                
            # Recursive step
            # min_den for next is d+1
            # max_den is same M
            res = dfs(new_rem, d + 1, max_den, used | {d}, depth + 1, max_depth)
            if res is not None:
                return [d] + res
                
        return None

    # Binary search for the minimal M
    # Lower bound: The smallest denominator in any expansion of 5/31.
    # 1/d <= 5/31 => d >= 31/5 = 6.2 => d >= 7.
    # So M must be at least 7.
    # Let's start low and go up.
    
    # We know a solution exists for M=10,000,000.
    # Let's try to find a smaller one.
    
    # To be safe and fast, we can start with a known good M from a quick greedy run,
    # or just start from a reasonable lower bound and double up, then binary search.
    
    # Quick Greedy to get an upper bound
    r = target
    greedy_max = 0
    greedy_sol = []
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d <= 0:
            break
        greedy_sol.append(d)
        r -= Fraction(1, d)
        greedy_max = max(greedy_max, d)
        
    if greedy_sol:
        best_solution = greedy_sol
        best_max_den = greedy_max
    
    # Now search for smaller M
    # Start from a lower bound.
    # The sum of reciprocals of 1..M is approx ln(M).
    # We need sum to be 5/31 ~ 0.16.
    # ln(M) >= 0.16 => M >= 1.17.
    # But distinct denominators > 1.
    # 1/7 + 1/8 + ... 
    # Let's just start M from 10 and go up.
    
    # Actually, a better strategy:
    # Try M values: 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000...
    # Until we find a solution, then binary search between the last failed and first success.
    
    low = 10
    high = best_max_den
    
    # If best_max_den is already small, we might skip binary search.
    # Let's try a few small M values directly first.
    
    candidate_ms = [10, 15, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000]
    
    # Filter to those < current best
    candidate_ms = [m for m in candidate_ms if m < best_max_den]
    
    # If current best is very good (e.g. < 100), we might want to search more finely.
    # But let's try the coarse steps first.
    
    found_m = None
    for m in candidate_ms:
        if check_time():
            break
        # Run DFS with max_den = m
        # We need to pass the target and constraints.
        # The DFS function needs to be called with the initial state.
        sol = dfs(target, 1, m, set(), 0, 256)
        if sol is not None:
            found_m = m
            # Check if this solution is valid and distinct
            # The DFS enforces distinctness via min_den increasing.
            # Check sum
            s = sum(Fraction(1, d) for d in sol)
            if s == target:
                best_solution = sol
                best_max_den = m
                # We can stop here if we want the first found, 
                # but we want the MINIMAL M.
                # So we should continue to smaller M? 
                # No, candidate_ms is sorted ascending.
                # So the first found is the smallest in this list.
                # But there might be a smaller M not in the list.
                # We should binary search between previous_failed and found_m.
                break
    else:
        # No solution found in candidate_ms, keep current best
        pass

    # If we found a solution at M_found, we can try to refine between the previous value and M_found
    # But for simplicity and time, the coarse search might be enough if the gap is small.
    # Let's try a binary search refinement if we found a solution.
    if found_m is not None:
        # Determine the lower bound for binary search
        # It's the largest M in candidate_ms that is smaller than found_m and failed, 
        # or just found_m - 1.
        # Let's just do a few steps down.
        # Or binary search between max(10, found_m//2) and found_m.
        
        # Actually, if found_m is 100, maybe 80 works?
        # Let's do a quick binary search in [found_m//2, found_m]
        lo = max(10, found_m // 2)
        hi = found_m
        
        while lo < hi and not check_time():
            mid = (lo + hi) // 2
            sol = dfs(target, 1, mid, set(), 0, 256)
            if sol is not None:
                s = sum(Fraction(1, d) for d in sol)
                if s == target:
                    best_solution = sol
                    best_max_den = mid
                    hi = mid - 1
            else:
                lo = mid + 1

    if best_solution is None:
        # Fallback to greedy if DFS failed completely (should not happen if greedy worked)
        return greedy_sol
    else:
        return best_solution

# EVOLVE-BLOCK-END
