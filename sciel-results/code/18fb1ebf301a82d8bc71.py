"""Optimized program: search for maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import random
import itertools
from fractions import Fraction

# Precompute LCM of 1..22
# LCM(1..22) = 232792560
_LCM = 232792560

def _verify_recip_dissociated(subset):
    """
    Return True if all 2^len(subset) reciprocal subset sums are distinct.
    Uses integer arithmetic with LCM as denominator.
    """
    if not subset:
        return True
    
    # Convert each element i to LCM // i
    # Subset sums are sums of these integers.
    # We need to check if all 2^k subset sums are distinct.
    # k is at most 22, so 2^22 is 4M, which is feasible but we only check subsets of size <= 11 or so.
    # Actually, we just need to check the specific subset passed in.
    
    k = len(subset)
    if k > 15:
        # Optimization: if k is large, 2^k is huge. 
        # But the task asks to maximize cardinality. 
        # If k=11, 2^11 = 2048. Very fast.
        # If k=15, 2^15 = 32768. Still fast.
        # If k=20, 2^20 = 1M. Might be slow in Python loop, but numpy could help.
        # Let's stick to standard Python set for now, it's fast enough for k<=15.
        pass

    values = [_LCM // i for i in subset]
    
    # Generate all subset sums
    # Using a set to store sums. If len(set) < 2^k, there's a collision.
    sums = set()
    sums.add(0)
    
    # Efficient subset sum generation
    # We can iterate through masks or use a doubling approach.
    # Doubling approach:
    # Start with {0}
    # For each v in values:
    #   new_sums = {s + v for s in sums}
    #   if any s in new_sums is already in sums -> collision? No, that's not right.
    #   We need to check if the set of ALL sums has size 2^k.
    #   Actually, if we maintain the set of sums, and we add v, the new sums are old_sums + v.
    #   If (old_sums + v) intersects old_sums, does that mean a collision?
    #   Yes, because a collision means sum(A) = sum(B) for distinct A, B.
    #   This is equivalent to sum(A \ B) = sum(B \ A) + sum(A \cap B) - sum(A \cap B) ... 
    #   Wait, simpler: sum(A) = sum(B) iff sum(A \ (A∩B)) = sum(B \ (A∩B)).
    #   Let X = A \ (A∩B), Y = B \ (A∩B). X and Y are disjoint subsets of the set.
    #   sum(X) = sum(Y).
    #   This is a linear dependency.
    
    # The most robust way: generate all 2^k sums and check length.
    # For k=11, 2048 sums. For k=15, 32768.
    
    # Let's use a recursive or iterative generation.
    # Iterative:
    current_sums = [0]
    for v in values:
        # New sums are existing sums + v
        # We can just add them to the list and check uniqueness at the end, 
        # or maintain a set.
        # Maintaining a set is O(2^k * k) roughly.
        new_sums = []
        for s in current_sums:
            ns = s + v
            new_sums.append(ns)
        current_sums.extend(new_sums)
        
    # Check if all unique
    return len(set(current_sums)) == len(current_sums)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    # Greedy approach first
    # Start with empty set, add elements if valid
    # Try to add in various orders?
    # Randomized Hill Climbing is better.
    
    candidates = list(range(1, 23))
    
    # Try to find a set of size 10, then 11, etc.
    # We can use a heuristic: elements with large LCM//i (i.e., small i) are "heavy" and more likely to cause collisions?
    # Actually, small i means large weight. Large weights are less likely to sum to small weights?
    # But 1/1 = 1. 1/2 + 1/3 + ... 
    # Let's just do random search.
    
    random.seed(42) # For reproducibility if needed, but random restarts help.
    
    # Strategy:
    # 1. Start with a random subset of size 10.
    # 2. If valid, try to add elements to make it size 11.
    # 3. If invalid, try to swap elements or remove/add.
    
    # Simpler: Randomized Greedy
    # Try many random orders of adding elements.
    
    for attempt in range(200):
        if time.time() > deadline:
            break
            
        # Shuffle candidates
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        # Try to build a set greedily
        for i in shuffled:
            if len(current_set) >= best_size:
                # Try to extend
                new_set = current_set + [i]
                if _verify_recip_dissociated(new_set):
                    current_set = new_set
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set[:]
                        # If we reached 12 or more, we can stop or continue
                        if best_size >= 12:
                            # 12 is very likely max or close to it
                            pass
            else:
                # Just add if valid
                new_set = current_set + [i]
                if _verify_recip_dissociated(new_set):
                    current_set = new_set
        
        # After greedy pass, try local search on best_set
        if time.time() > deadline:
            break
            
        # Local search: try to add any missing element
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in candidates:
                if i not in best_set:
                    new_set = best_set + [i]
                    if _verify_recip_dissociated(new_set):
                        best_set = new_set
                        best_size = len(best_set)
                        improved = True
                        break # Restart loop to try adding another

    return best_set
