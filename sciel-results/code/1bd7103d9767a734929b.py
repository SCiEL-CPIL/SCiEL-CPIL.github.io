import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    target = Fraction(5, 31)
    
    # Known good solution with max 248
    best_solution = [8, 31, 248]
    best_max = 248
    
    # Search for smaller max denominator
    # We iterate M from 1 upwards
    # If we find a solution with max M, we can stop if we are sure it's minimal?
    # The problem asks to minimize the LARGEST denominator.
    # So we want the smallest M such that a solution exists.
    
    # However, we must return the denominators.
    # We can just find the first M that works.
    
    for M in range(1, 100):
        if time.time() > deadline - 2:
            break
            
        # Precompute suffix sums of reciprocals for 1..M
        # suffix_sum[i] = sum(1/j for j in range(i, M+1))
        suffix_sum = [Fraction(0, 1)] * (M + 2)
        for i in range(M, 0, -1):
            suffix_sum[i] = suffix_sum[i+1] + Fraction(1, i)
            
        # DFS
        def dfs(current_sum, next_d, path):
            if time.time() > deadline:
                return None
            if current_sum == target:
                return list(path)
            if next_d > M:
                return None
            if current_sum > target:
                return None
            # Pruning: max possible sum from remaining
            if current_sum + suffix_sum[next_d] < target:
                return None
            
            # Option 1: Include next_d
            # Check if 1/next_d can be added without exceeding target
            rem = target - current_sum
            if Fraction(1, next_d) <= rem:
                path.append(next_d)
                res = dfs(current_sum + Fraction(1, next_d), next_d + 1, path)
                if res is not None:
                    return res
                path.pop()
                
            # Option 2: Exclude next_d
            res = dfs(current_sum, next_d + 1, path)
            if res is not None:
                return res
            
            return None

        res = dfs(Fraction(0, 1), 1, [])
        if res is not None:
            # Verify
            if len(set(res)) == len(res) and all(1 <= d <= M for d in res):
                s = sum(Fraction(1, d) for d in res)
                if s == target:
                    best_solution = res
                    best_max = M
                    # Since we iterate M upwards, the first valid M is the minimum.
                    # We can break.
                    break

    return best_solution
