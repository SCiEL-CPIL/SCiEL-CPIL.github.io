"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time
from itertools import combinations

def _factorize(n):
    """Return set of prime factors of n."""
    if n <= 1:
        return set()
    primes = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            primes.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        primes.add(n)
    return primes

def _get_cost(numbers):
    """Calculate the number of distinct prime factors across all pairwise sums."""
    all_primes = set()
    n = len(numbers)
    for i in range(n):
        for j in range(i + 1, n):
            s = numbers[i] + numbers[j]
            all_primes |= _factorize(s)
    return len(all_primes)

def _generate_random_set(count=12, max_val=10000):
    """Generate a random set of distinct positive integers."""
    return random.sample(range(1, max_val + 1), count)

def construct():
    """
    Use Simulated Annealing to find 12 distinct integers <= 10000
    that minimize the number of distinct prime factors in their pairwise sums.
    """
    start_time = time.time()
    time_limit = 50.0  # Leave buffer for overhead
    
    # Initial state: Try a few heuristic starts
    # 1. Multiples of 2 (all even -> sums even, factor 2 guaranteed)
    # 2. Multiples of 6
    # 3. Random
    initial_candidates = [
        list(range(2, 26, 2)), # 2, 4, ..., 24
        list(range(6, 78, 6)), # 6, 12, ..., 72
        list(range(1, 13)),    # 1..12
        _generate_random_set()
    ]
    
    best_numbers = None
    best_cost = float('inf')
    
    for start_nums in initial_candidates:
        # Validate start
        if len(set(start_nums)) != 12 or any(x < 1 or x > 10000 for x in start_nums):
            continue
            
        current_numbers = start_nums[:]
        current_cost = _get_cost(current_numbers)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_numbers = current_numbers[:]
        
        # Simulated Annealing
        T = 10.0 # Initial temperature (cost units, but cost is int, so this scales acceptance)
        cooling_rate = 0.995
        min_T = 1e-5
        
        # We want to run SA for a portion of the time for each start, or just one long SA from the best start.
        # Let's run a short SA for each start to see if it improves, then continue from the best found.
        
        # Actually, a single long SA from the best initial candidate is often better.
        # Let's just run SA from the best initial candidate found so far.
        if best_numbers is None:
            continue
            
        current_numbers = best_numbers[:]
        current_cost = best_cost
        
        while T > min_T and (time.time() - start_time) < time_limit:
            # Generate a neighbor
            new_numbers = current_numbers[:]
            idx = random.randint(0, 11)
            old_val = new_numbers[idx]
            
            # Try to replace with a random value
            # Bias towards values that might share factors? 
            # For now, uniform random replacement.
            new_val = random.randint(1, 10000)
            
            # Ensure distinctness
            if new_val in new_numbers and new_val != old_val:
                # Collision, try a few times or skip
                # To keep it simple, just pick a different one
                for _ in range(10):
                    new_val = random.randint(1, 10000)
                    if new_val not in new_numbers or new_val == old_val:
                        break
            
            if new_val == old_val:
                continue
                
            new_numbers[idx] = new_val
            
            # Check distinctness again strictly
            if len(set(new_numbers)) != 12:
                continue
                
            new_cost = _get_cost(new_numbers)
            
            # Acceptance criterion
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_numbers = new_numbers
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_numbers = current_numbers[:]
                    # Optional: Reset temperature if we found a new global best? 
                    # No, keep cooling.
            
            T *= cooling_rate
            
            # If we hit a very low cost, we can slow down cooling or stop
            if best_cost <= 3: # 3 is very good (e.g. {2,3,5} or similar)
                # We might break early if we are confident
                # But let's run a bit more to see if 2 or 1 is possible (unlikely)
                pass

    if best_numbers is None:
        return list(range(1, 13))
        
    return best_numbers
# EVOLVE-BLOCK-END
