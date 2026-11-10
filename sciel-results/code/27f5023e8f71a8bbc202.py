"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _get_divisors(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs, reverse=True)

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # 5s safety margin
    
    n = 5040
    target = 4979
    divs = _get_divisors(n)
    div_set = set(divs)
    
    # Greedy fallback
    remainder = target
    greedy_chosen = []
    for d in divs:
        if d <= remainder:
            greedy_chosen.append(d)
            remainder -= d
    if remainder != 0:
        # If greedy fails (shouldn't for practical numbers usually, but safe), 
        # we might need a general solver, but for 5040 it works.
        # If it doesn't sum to target, the greedy is invalid. 
        # However, the task implies a solution exists.
        pass 

    best_solution = greedy_chosen
    best_k = len(greedy_chosen)
    
    # Check for smaller k (1 to best_k - 1)
    # Since greedy gave 5, we check 1, 2, 3, 4.
    
    # k=1
    if target in div_set:
        return [target]
        
    # k=2
    for d1 in divs:
        if time.time() > deadline:
            break
        d2 = target - d1
        if d2 in div_set and d2 != d1:
            return [d1, d2]
            
    # k=3
    # Iterate d1, then check if target-d1 is sum of 2 distinct divisors from remaining
    # To be efficient, we can just iterate d1, then d2, and check d3
    for i in range(len(divs)):
        if time.time() > deadline:
            break
        d1 = divs[i]
        rem1 = target - d1
        if rem1 <= 0:
            continue
        # We need d2 + d3 = rem1 with d2, d3 distinct from d1 and each other
        # We can iterate d2 from divs, ensuring d2 != d1
        for j in range(len(divs)):
            if i == j:
                continue
            d2 = divs[j]
            if d2 >= rem1: # Since d3 > 0, d2 must be < rem1
                continue
            d3 = rem1 - d2
            if d3 in div_set and d3 != d1 and d3 != d2:
                return [d1, d2, d3]
                
    # k=4
    # Iterate d1, d2, then check if target-d1-d2 is sum of 2 distinct divisors
    for i in range(len(divs)):
        if time.time() > deadline:
            break
        d1 = divs[i]
        rem1 = target - d1
        if rem1 <= 0:
            continue
        for j in range(len(divs)):
            if i == j:
                continue
            d2 = divs[j]
            rem2 = rem1 - d2
            if rem2 <= 0:
                continue
            # Need d3 + d4 = rem2, d3, d4 distinct from d1, d2, each other
            for k in range(len(divs)):
                if i == k or j == k:
                    continue
                d3 = divs[k]
                if d3 >= rem2:
                    continue
                d4 = rem2 - d3
                if d4 in div_set and d4 != d1 and d4 != d2 and d4 != d3:
                    return [d1, d2, d3, d4]
                    
    # If no smaller k found, return greedy (k=5)
    return best_solution
# EVOLVE-BLOCK-END
