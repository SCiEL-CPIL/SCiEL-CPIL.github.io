import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # We know k >= 8.
    # We search for k = 8, 9, ...
    
    def find_solution(k, max_den=10000000):
        """
        Find k distinct denominators in [5, max_den] summing to 1.
        Uses DFS with pruning.
        """
        def dfs(remaining, k_left, min_den, current):
            if time.time() > deadline:
                return None
            
            if k_left == 0:
                if remaining == 0:
                    return current[:]
                return None
            
            if k_left == 1:
                # 1/x = remaining => x = 1/remaining
                if remaining > 0:
                    num, den = remaining.numerator, remaining.denominator
                    if den % num == 0:
                        x = den // num
                        if x > min_den and x <= max_den:
                            return current + [x]
                return None
            
            # Bounds for next denominator a
            # 1/a < remaining => a > 1/remaining
            # Also, we need to fit k_left terms.
            # Max sum of k_left terms starting from a is 1/a + 1/(a+1) + ...
            # This must be >= remaining.
            # Min sum of k_left terms starting from a is 1/a + 1/(a+1) ... (actually min is with large denominators, but we are bounded by max_den)
            # Better bound:
            # remaining <= 1/a + 1/(a+1) + ... + 1/(a+k_left-1)
            # remaining >= 1/a + 1/(a+1) ... (this is not tight)
            # Use: remaining < k_left / a => a < k_left / remaining
            # And: remaining > 1/a => a > 1/remaining
            
            lower_a = max(min_den, int(1 / remaining) + 1)
            upper_a = min(max_den, int(k_left / remaining) + 1)
            
            if lower_a > upper_a:
                return None
            
            # Prune: if even taking the smallest possible denominators (a, a+1, ...) the sum is too small
            # sum_{i=0}^{k_left-1} 1/(a+i) < remaining
            # We can approximate this or just check the first few terms in the loop.
            
            for a in range(lower_a, upper_a + 1):
                if time.time() > deadline:
                    return None
                
                # Prune: max possible sum from this point
                # If 1/a + 1/(a+1) + ... + 1/(a+k_left-1) < remaining, break
                # This is the maximum sum we can get with distinct denominators starting at a.
                max_possible = 0.0
                for i in range(k_left):
                    max_possible += 1.0 / (a + i)
                if max_possible < float(remaining) - 1e-15:
                    break
                
                new_remaining = remaining - Fraction(1, a)
                if new_remaining <= 0:
                    continue
                
                result = dfs(new_remaining, k_left - 1, a + 1, current + [a])
                if result is not None:
                    return result
            
            return None
        
        return dfs(Fraction(1), k, 5, [])

    # Try k=8 first
    for k in range(8, 20):
        if time.time() > deadline - 5:
            break
        sol = find_solution(k, max_den=10000000)
        if sol is not None:
            # Verify
            s = sum(Fraction(1, x) for x in sol)
            if s == 1 and len(set(sol)) == len(sol) and all(5 <= x <= 10000000 for x in sol):
                return sol
        
        # If k=8 takes too long, we might need to interrupt and try a different strategy or accept k=9
        # But k=8 search space is small enough.
        
    # Fallback: Greedy algorithm (will produce a valid, but possibly non-minimal, solution)
    # This ensures we return *something* valid if DFS fails.
    def greedy_solution():
        remaining = Fraction(1)
        denoms = []
        last = 4
        while remaining > 0:
            # Find smallest d > last such that 1/d <= remaining
            d = last + 1
            while Fraction(1, d) > remaining:
                d += 1
            denoms.append(d)
            remaining -= Fraction(1, d)
            last = d
            if len(denoms) > 256:
                break
        return denoms

    sol = greedy_solution()
    s = sum(Fraction(1, x) for x in sol)
    if s == 1 and len(set(sol)) == len(sol) and all(5 <= x <= 10000000 for x in sol):
        return sol

    raise RuntimeError("No solution found")
