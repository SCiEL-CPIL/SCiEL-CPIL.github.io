"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _compute_lcm_upto(n):
    l = 1
    for i in range(1, n + 1):
        l = _lcm(l, i)
    return l

# Precompute LCM of 1..22
L = _compute_lcm_upto(22)
# Integer values corresponding to 1/i
# val[i] = L // i
VALS = [0] * 23
for i in range(1, 23):
    VALS[i] = L // i

def construct():
    """
    Find a nonempty subset of [1, 22] with maximum cardinality such that
    all reciprocal subset sums (including empty sum = 0) are distinct.
    
    We use integer arithmetic by scaling with L = lcm(1..22).
    The condition that subset sums of reciprocals are distinct is equivalent
    to subset sums of {L/i} being distinct.
    """
    candidates = list(range(1, 23))
    best_set = []
    best_size = 0
    
    # Define the greedy check using integer sums
    def try_greedy(order):
        current = []
        current_sums = {0}
        
        for candidate in order:
            v = VALS[candidate]
            # Check if adding v creates collisions
            # New sums = current_sums U {s + v for s in current_sums}
            # If len(new_sums) < 2 * len(current_sums), there is a collision
            
            # Optimization: if v is 0 (not possible here) or if we can detect collision faster
            # For now, generate new sums
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + v)
            
            if len(new_sums) == 2 * len(current_sums):
                current.append(candidate)
                current_sums = new_sums
        
        return current

    # Strategy 1: Descending order (largest values first, i.e., smallest indices)
    # This tends to create large gaps, making collisions less likely initially?
    # Actually, large values are "harder" to fit if they are close to existing sums.
    # But let's try both.
    orderings = []
    
    # Descending index (1, 2, 3...) -> Largest reciprocals first
    orderings.append(sorted(candidates))
    
    # Ascending index (22, 21, ...) -> Smallest reciprocals first
    orderings.append(sorted(candidates, reverse=True))
    
    # Random shuffles
    rng = random.Random(42)
    for _ in range(50):
        order = candidates[:]
        rng.shuffle(order)
        orderings.append(order)
        
    # Heuristic: Sort by value descending (largest L/i first)
    # This is same as index ascending
    # Heuristic: Sort by value ascending (smallest L/i first)
    # This is same as index descending
    
    # Let's also try a "mixed" strategy: prioritize elements that are coprime to the current product?
    # Or just rely on random restarts with more iterations.
    
    for order in orderings:
        current_set = try_greedy(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    # If we still have time, we can try local search or more random restarts
    # But 60s limit. Integer ops are fast.
    # Let's add more random restarts if best_size is small
    if best_size < 10:
        for _ in range(100):
            order = candidates[:]
            rng.shuffle(order)
            current_set = try_greedy(order)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]

    if not best_set:
        best_set = [1]
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
