"""Combined program: greedy + integer verification + randomized restarts."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 58.0  # Safety margin for 60s limit
    
    universe = list(range(1, 23))
    # LCM of 1..22 for integer arithmetic
    lcm_val = 1
    for i in range(1, 23):
        lcm_val = lcm_val * i // gcd(lcm_val, i)
    
    # Precompute reciprocal values as integers: 1/x * lcm_val
    recip_int = {x: lcm_val // x for x in universe}
    
    def is_dissociated_int(S):
        """Check if all subset sums of reciprocals of S are distinct using integer arithmetic."""
        if not S:
            return True
        sums = {0}
        for x in S:
            val = recip_int[x]
            # Check collision before updating
            for s in sums:
                if (s + val) in sums:
                    return False
            # Update sums
            new_sums = [s + val for s in sums]
            sums.update(new_sums)
        return True

    def try_add_int(S, x):
        """Check if S union {x} is dissociated using integer arithmetic."""
        return is_dissociated_int(S + [x])
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy with different deterministic orders
    # Key insight: larger numbers have smaller reciprocals, less likely to cause collisions
    orders = [
        sorted(universe, reverse=True),  # Large first (small reciprocals)
        sorted(universe),  # Small first
        [n for n in universe if n >= 12],  # Large numbers only
        [n for n in universe if n % 2 == 0],  # Evens
        [n for n in universe if n % 2 == 1],  # Odds
        [n for n in universe if n >= 10],  # Very large
        [n for n in universe if n in [2,3,5,7,11,13,17,19,23]],  # Primes
        [n for n in universe if n in [4,6,8,9,10,12,14,15,16,18,20,21,22]],  # Composites
    ]
    
    for order in orders:
        if time.time() - start_time > timeout:
            break
        current_set = []
        for x in order:
            if try_add_int(current_set, x):
                current_set.append(x)
        
        # Try to expand with remaining elements
        remaining = [x for x in universe if x not in current_set]
        added = True
        while added and time.time() - start_time < timeout:
            added = False
            for x in remaining[:]:
                if try_add_int(current_set, x):
                    current_set.append(x)
                    remaining.remove(x)
                    added = True
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Strategy 2: Randomized restarts with weighted randomization
    random.seed(42)
    restart_count = 0
    while time.time() - start_time < timeout:
        restart_count += 1
        order = universe[:]
        random.shuffle(order)
        
        current_set = []
        for x in order:
            if try_add_int(current_set, x):
                current_set.append(x)
        
        # Try to expand with remaining elements
        remaining = [x for x in universe if x not in current_set]
        added = True
        while added and time.time() - start_time < timeout:
            added = False
            for x in remaining[:]:
                if try_add_int(current_set, x):
                    current_set.append(x)
                    remaining.remove(x)
                    added = True
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            # If we found a very large set, we can stop early
            if best_size >= 15:
                break
    
    # Strategy 3: Local search - try to improve current best by swapping
    if best_set and time.time() - start_time < timeout - 5:
        for _ in range(100):
            if time.time() - start_time > timeout:
                break
            improved = False
            for i in range(len(best_set)):
                for j in range(len(universe)):
                    if universe[j] not in best_set:
                        new_set = best_set[:i] + best_set[i+1:] + [universe[j]]
                        if is_dissociated_int(new_set) and len(new_set) > best_size:
                            best_set = new_set
                            best_size = len(new_set)
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
    
    # Fallback to a known valid set if search fails
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
# EVOLVE-BLOCK-END
