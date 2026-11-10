"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s kill (evaluator limit)
    
    candidates = list(range(1, 23))
    reciprocals = {x: Fraction(1, x) for x in candidates}
    
    best_set = []
    best_size = 0
    
    def try_add(sums, elem):
        """
        Checks if elem can be added to the set of subset sums.
        Returns new_sums if successful, None otherwise.
        """
        r = reciprocals[elem]
        new_sums = set(sums)
        for s in sums:
            ns = s + r
            if ns in new_sums:
                return None
            new_sums.add(ns)
        return new_sums

    def greedy_construct(order):
        """
        Greedily builds a set by adding elements if they don't cause collisions.
        """
        current_sums = {Fraction(0, 1)}
        current_set = []
        
        for elem in order:
            new_sums = try_add(current_sums, elem)
            if new_sums is not None:
                current_sums = new_sums
                current_set.append(elem)
        
        return current_set

    # Define several orderings to try
    orders_to_try = []
    
    # 1. Ascending
    orders_to_try.append(sorted(candidates))
    
    # 2. Descending
    orders_to_try.append(sorted(candidates, reverse=True))
    
    # 3. Powers of 2 first, then others
    pows = [1, 2, 4, 8, 16]
    rest = [x for x in candidates if x not in pows]
    orders_to_try.append(pows + rest)
    
    # 4. Interleaved small/large
    inter = []
    for i in range(11):
        inter.append(i + 1)
        inter.append(22 - i)
    orders_to_try.append(inter)
    
    # 5-14. Randomized orders
    rng = random.Random(42)
    for _ in range(10):
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        orders_to_try.append(rand_order)

    # Run greedy for each ordering
    for order in orders_to_try:
        if time.time() > deadline:
            break
            
        current_set = greedy_construct(order)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
            # If we found a very large set (e.g., >= 11), we can stop early
            # as it's hard to beat significantly with simple heuristics
            if best_size >= 11:
                # We could try to polish, but let's ensure we don't time out
                # A set of size 11 is a strong result.
                pass

    # Fallback
    if not best_set:
        best_set = [1, 2, 4, 8, 16]

    return best_set
# EVOLVE-BLOCK-END
