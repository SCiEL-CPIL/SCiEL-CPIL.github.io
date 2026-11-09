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
    
    # DP: dp[s] = minimum number of divisors to achieve sum s
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
    
    # Local search: try to reduce solution size by merging pairs
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = set(best_solution)
        
        # Try merging: find a, b in current where a+b is a divisor and not in current
        for i, a in enumerate(best_solution):
            for b in best_solution[i+1:]:
                s = a + b
                if s in divisors and s not in current:
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
