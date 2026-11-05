import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin
    
    N = factorial(12)
    
    # Generate all divisors of N
    divisors = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
        i += 1
        
    # Filter divisors > 12 and sort ascending
    valid_divisors = sorted([d for d in divisors if d > 12])
    
    best_max = N + 1
    best_solution = None
    
    # Greedy heuristic to find an initial valid solution and set a tight bound
    def greedy():
        rem = N
        factors = []
        last = 12
        while rem > 1:
            found = False
            for d in valid_divisors:
                if d <= last:
                    continue
                # If d is the last factor, it must equal rem
                if d == rem:
                    if rem > last and rem > 12:
                        factors.append(rem)
                        return factors
                    else:
                        break
                # If d is not the last factor, it must divide rem and be smaller
                if d > rem:
                    break
                if rem % d == 0:
                    factors.append(d)
                    rem //= d
                    last = d
                    found = True
                    break
            if not found:
                # If we didn't find a divisor, maybe rem itself is the last factor
                if rem > last and rem > 12:
                    factors.append(rem)
                    return factors
                return None
        return factors if factors else None
    
    g_sol = greedy()
    if g_sol:
        best_max = max(g_sol)
        best_solution = g_sol
    
    # DFS to minimize the largest factor
    def dfs(remaining, min_idx, path):
        nonlocal best_max, best_solution
        
        if time.time() > deadline:
            return
            
        if remaining == 1:
            if path:
                current_max = path[-1]
                if current_max < best_max:
                    best_max = current_max
                    best_solution = list(path)
            return
            
        if len(path) >= 12:
            return
            
        # Iterate through possible next factors
        for i in range(min_idx, len(valid_divisors)):
            d = valid_divisors[i]
            
            # Pruning: if d >= best_max, no further d can improve best_max
            if d >= best_max:
                break
                
            # Pruning: if d > remaining, no further d can divide remaining
            if d > remaining:
                break
                
            if remaining % d != 0:
                continue
                
            path.append(d)
            dfs(remaining // d, i + 1, path)
            path.pop()
            
            # If we found the theoretical minimum (smallest possible max factor), we can stop
            # The smallest factor is 14. The max factor must be at least 14.
            # If best_max is 14, it's optimal.
            if best_max == 14:
                return

    dfs(N, 0, [])
    
    if best_solution:
        return best_solution
    else:
        # Fallback: 12! itself
        return [N]
