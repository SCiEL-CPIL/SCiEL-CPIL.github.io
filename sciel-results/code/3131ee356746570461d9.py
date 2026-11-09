"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = 5040
    target = 4979

    # Get all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # Sort divisors in descending order for greedy initialization
    divisors_desc = sorted(divisors, reverse=True)
    num_divisors = len(divisors)  # 60 divisors

    # Greedy initial solution
    remainder = target
    chosen = []
    for d in divisors_desc:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
    
    assert remainder == 0, "Greedy failed to find a solution"
    best_count = len(chosen)
    best_solution = chosen[:]
    
    # If we already have a small solution, try to improve via DP
    # DP: dp[s] = minimum number of divisors to achieve sum s
    # We also track the actual set of divisors used
    
    # Since target is 4979 and we have 60 divisors, we can do DP
    # dp[s] stores the minimum count, and parent[s] stores (prev_sum, divisor)
    
    INF = float('inf')
    dp = [INF] * (target + 1)
    parent = [(-1, -1)] * (target + 1)
    dp[0] = 0
    
    # For each divisor, update DP in reverse to avoid using same divisor twice
    for d in divisors:
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
    
    # Check if we found a better solution
    if dp[target] < best_count:
        # Reconstruct solution
        s = target
        solution = []
        while s > 0:
            prev_s, d = parent[s]
            solution.append(d)
            s = prev_s
        best_solution = solution
        best_count = dp[target]
    
    # If DP found the optimal, return it
    # Otherwise, try local search to improve further
    
    # Try to reduce the solution size by swapping
    # If we have a solution with k divisors, try to replace some with fewer
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = set(best_solution)
        current_sum = target
        
        # Try removing each divisor and seeing if we can replace it with fewer
        for d in list(current):
            new_sum = current_sum - d
            # Try to find a subset of remaining divisors that sums to d
            # with fewer than 1 element (impossible) or try to merge
            # Actually, let's try: remove d, then try to add other divisors
            # to reach target with fewer total elements
            
            # Try replacing d with a combination of smaller divisors
            # that sums to d, using divisors not in current
            available = [x for x in divisors if x not in current and x < d]
            
            # Check if d can be represented as sum of 0 divisors (no) 
            # or if we can find a representation with fewer elements
            # Since d is a single element, we need to find sum=d with 0 elements (impossible)
            # So this approach won't reduce count by removing one element
            
            # Alternative: try to find two divisors a, b in current such that
            # a + b can be replaced by a single divisor c
            pass
        
        # Try merging: find a, b in current where a+b is a divisor and not in current
        for i, a in enumerate(best_solution):
            for b in best_solution[i+1:]:
                s = a + b
                if s in divisors and s not in current:
                    # Check if replacing a, b with s still gives valid sum
                    # New sum = target - a - b + s = target
                    new_solution = [x for x in best_solution if x != a and x != b]
                    new_solution.append(s)
                    if len(set(new_solution)) == len(new_solution) and sum(new_solution) == target:
                        if len(new_solution) < len(best_solution):
                            best_solution = new_solution
                            best_count = len(best_solution)
                            improved = True
                            break
            if improved:
                break
    
    # Final verification
    assert sum(best_solution) == target
    assert len(set(best_solution)) == len(best_solution)
    assert all(x in divisors for x in best_solution)
    assert len(best_solution) <= 256
    
    return best_solution
# EVOLVE-BLOCK-END
