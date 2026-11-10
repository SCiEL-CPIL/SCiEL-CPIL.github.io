import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds limit
    
    # Fallback solution from seed program logic (just in case search fails or takes too long)
    # The seed program returns a list of denominators.
    # We will try to find a better one (fewer terms).
    
    def dfs(k, depth, current_sum, last_denom, path):
        if time.time() > deadline:
            return None
            
        if depth == k:
            if current_sum == Fraction(1, 1):
                return path[:]
            return None
        
        remaining = k - depth
        target = Fraction(1, 1) - current_sum
        
        # Lower bound for next denominator
        # 1/d <= target => d >= 1/target
        min_d = max(last_denom + 1, int(1 / target) + 1 if target > 0 else last_denom + 1)
        
        # If target is 0, we can't add more terms unless remaining is 0
        if target == 0:
            if remaining == 0:
                return path[:]
            else:
                return None
        
        # Upper bound for next denominator
        # We need to fit 'remaining' terms.
        # The sum of the remaining terms (including current) must be 'target'.
        # The minimum sum of 'remaining' distinct integers starting from min_d is approximated.
        # Actually, we can just iterate d up to a reasonable limit.
        # A safe upper bound: d <= 1/target * remaining is not quite right.
        # If we pick d, we have target - 1/d left for remaining-1 terms.
        # The max sum of remaining-1 terms with denominators > d is bounded.
        # But a simpler heuristic: d cannot be larger than 1/target * some_factor.
        # Let's use the constraint that the sum of the smallest 'remaining' denominators starting from min_d must be >= target.
        # If sum(1/(min_d + i) for i in range(remaining)) < target, then no solution.
        
        min_possible_sum = sum(Fraction(1, min_d + i) for i in range(remaining))
        if min_possible_sum < target:
            return None
            
        # Iterate possible next denominators
        # d must be such that 1/d <= target
        max_d = int(1 / target)
        
        # We also need to ensure we can form the rest.
        # For each d, we check if the rest is possible.
        
        for d in range(min_d, max_d + 1):
            if d > 10000000:
                break
            next_sum = current_sum + Fraction(1, d)
            if next_sum > 1:
                break # Since d is increasing, sum will only get smaller, but wait.
                # 1/d is decreasing. So next_sum is decreasing as d increases.
                # If next_sum > 1, we might still have solutions for larger d?
                # No, if next_sum > 1, we have exceeded 1. But we can't subtract.
                # So if next_sum > 1, we must increase d to reduce 1/d.
                # So we continue.
                
            # Pruning: Can we complete the sum?
            # Remaining target: 1 - next_sum
            # Remaining terms: remaining - 1
            # Min sum of remaining-1 terms starting from d+1:
            if remaining > 1:
                rem_target = 1 - next_sum
                if rem_target < 0:
                    continue
                min_rem_sum = sum(Fraction(1, d + 1 + i) for i in range(remaining - 1))
                if min_rem_sum < rem_target:
                    continue
            
            path.append(d)
            res = dfs(k, depth + 1, next_sum, d, path)
            if res:
                return res
            path.pop()
            
        return None

    best_solution = None
    
    # Try k from 3 upwards
    for k in range(3, 11):
        res = dfs(k, 0, Fraction(0, 1), 4, [])
        if res:
            best_solution = res
            break # Found minimal k
        
        if time.time() > deadline:
            break

    if best_solution:
        return best_solution
    
    # Fallback: Run the seed algorithm quickly to get a valid solution
    # The seed algorithm uses DP with target 2520.
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
            
    raise RuntimeError("no solution found")
