"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Generate all positive divisors of 5040
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    divisors.sort(reverse=True)
    
    # Try to find a representation with minimum number of terms
    # Start with k=1 and increase until we find a solution
    
    # For each k, try combinations of k divisors
    # Since 4979 is close to 5040, and 5040 is a divisor, 
    # maybe we can do it with few terms
    
    # Let's think about this more carefully:
    # 4979 = 5040 - 61
    # 61 is not a divisor of 5040
    # 5040's divisors include: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 28, 30, 35, 36, 40, 42, 45, 48, 56, 60, 63, 70, 72, 80, 84, 90, 105, 112, 120, 126, 140, 144, 168, 180, 210, 224, 240, 252, 280, 315, 336, 360, 420, 504, 560, 630, 720, 840, 1008, 1260, 1680, 2520, 5040
    
    # Can we express 4979 as sum of distinct divisors?
    # 4979 is odd. Sum of distinct divisors... 
    # If we use 5040, we'd need negative 61, which isn't possible.
    # So we can't use 5040.
    
    # Let's try: 2520 + 1680 = 4200, need 779 more
    # 779 is prime? 779 = 19 * 41, not a divisor
    # 2520 + 1260 = 3780, need 1199
    # 1199 = 11 * 109, not a divisor
    
    # Let me just do a systematic search
    
    # For k terms, we need to find k distinct divisors that sum to target
    # Try small k first
    
    # k=1: just check if target is a divisor
    if target in divisors:
        return [target]
    
    # k=2: check all pairs
    div_set = set(divisors)
    for i in range(len(divisors)):
        d1 = divisors[i]
        if d1 >= target:
            continue
        d2 = target - d1
        if d2 in div_set and d2 != d1:
            return [d1, d2]
    
    # k=3: check all triples - this could be expensive, use meet-in-the-middle
    # For k=3, we can do: for each pair (a,b), check if target-a-b is a divisor not in {a,b}
    
    # Build a dictionary for fast lookup
    # For efficiency, let's use a different approach for k=3,4,...
    
    # Let's use dynamic programming or backtracking with pruning
    
    # Backtracking approach:
    best_solution = None
    best_k = len(divisors) + 1
    
    def backtrack(idx, remaining, current, k_limit):
        nonlocal best_solution, best_k
        if time.time() > deadline:
            return
        
        if remaining == 0:
            if len(current) < best_k:
                best_k = len(current)
                best_solution = list(current)
            return
        
        if remaining < 0:
            return
        
        if len(current) >= best_k:
            return
        
        if idx >= len(divisors):
            return
        
        # Pruning: if we can't reach remaining with the remaining divisors
        # The maximum we can add is sum of divisors[idx:]
        # But that's expensive to compute, so let's use a simpler bound
        
        # Try including divisors[idx]
        if divisors[idx] <= remaining:
            current.append(divisors[idx])
            backtrack(idx + 1, remaining - divisors[idx], current, k_limit)
            current.pop()
        
        # Try excluding divisors[idx]
        backtrack(idx + 1, remaining, current, k_limit)
    
    # Start backtracking
    backtrack(0, target, [], len(divisors))
    
    if best_solution is not None:
        return best_solution
    
    # If backtracking didn't find a good solution within time, 
    # try a greedy approach as fallback
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    
    if remainder == 0:
        return chosen
    
    # This shouldn't happen, but return something
    return chosen
# EVOLVE-BLOCK-END
