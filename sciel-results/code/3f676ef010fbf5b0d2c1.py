"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    start_time = time.time()
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    def check_dissociated(elements):
        """Check if all subset sums of reciprocals are distinct using integer arithmetic."""
        k = len(elements)
        if k == 0:
            return True
        
        # Compute LCM of all elements
        L = reduce(lcm, elements)
        
        # Convert reciprocals to integers: 1/x -> L/x
        int_vals = [L // x for x in elements]
        
        # Generate all subset sums and check for duplicates
        sums = {0}
        for v in int_vals:
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        
        return len(sums) == (1 << k)

    candidates = list(range(1, 23))
    best_set = []
    best_size = 0
    
    def run_greedy(order):
        current = []
        for elem in order:
            if time.time() > deadline - 5:
                break
            if elem in current:
                continue
            test_set = current + [elem]
            if check_dissociated(test_set):
                current = test_set
        return current

    # Define smart orderings
    smart_orders = [
        sorted(candidates, reverse=True),
        sorted(candidates),
        [2, 3, 5, 7, 11, 13, 17, 19, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22],
        [1, 2, 4, 8, 16, 3, 6, 12, 5, 10, 20, 7, 14, 21, 9, 18, 11, 22, 13, 15, 17, 19],
        [22, 1, 21, 2, 20, 3, 19, 4, 18, 5, 17, 6, 16, 7, 15, 8, 14, 9, 13, 10, 12, 11]
    ]
    
    # Run smart orders first
    for order in smart_orders:
        res = run_greedy(order)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res[:]

    # Run random orders
    # We will run as many as possible within the time limit
    while time.time() < deadline - 5:
        order = candidates[:]
        random.shuffle(order)
        res = run_greedy(order)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res[:]
            
            # If we found a very large set, we might want to try to improve it locally
            # But for now, just keep searching for even larger random sets.
            
    # Final local improvement attempt on the best set
    if best_set and time.time() < deadline - 2:
        # Try to add any remaining element
        for elem in candidates:
            if time.time() > deadline - 2:
                break
            if elem in best_set:
                continue
            test_set = best_set + [elem]
            if check_dissociated(test_set):
                best_set = test_set
                best_size += 1

    if not best_set:
        best_set = [1]

    return best_set
# EVOLVE-BLOCK-END
