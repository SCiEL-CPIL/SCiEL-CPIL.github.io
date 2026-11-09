import itertools
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin out of 90s
    
    n = 5040
    target = 4979
    
    # Generate all divisors of n
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in descending order to potentially find large values first?
    # For combinations, order doesn't matter for correctness, but might help if we were pruning.
    # Here we just enumerate.
    divisors.sort(reverse=True)
    
    num_divs = len(divisors)
    
    # We know a solution of size 4 exists (from greedy).
    # We search for minimum k starting from 1.
    for k in range(1, 5):
        if time.time() > deadline:
            # Fallback to greedy if somehow we hit deadline (unlikely for k<=4)
            remainder = target
            chosen = []
            for d in divisors:
                if d <= remainder:
                    chosen.append(d)
                    remainder -= d
                    if remainder == 0:
                        return chosen
            # If greedy fails (it shouldn't for 5040), return empty or whatever
            return chosen
            
        # Check all combinations of size k
        for combo in itertools.combinations(divisors, k):
            if time.time() > deadline:
                break
            if sum(combo) == target:
                return list(combo)
        
        if time.time() > deadline:
            break
            
    # If no solution found up to k=4 (should not happen), fallback to greedy
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                return chosen
                
    # If even greedy fails (mathematically impossible for 5040 and 4979? 
    # 5040 is practical number, so all numbers up to 5040 can be represented as sum of distinct divisors.
    # 4979 < 5040, so a solution always exists.)
    return chosen
