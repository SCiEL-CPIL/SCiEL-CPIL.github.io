"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd

def _is_coprime_to_set(a, current_set):
    """Check if a is coprime to all elements in current_set."""
    for x in current_set:
        if gcd(a, x) > 1:
            return False
    return True

def _local_search(initial_set, n=512, time_limit=10.0):
    """Perform local search (hill climbing) starting from initial_set."""
    start_time = time.time()
    current_set = list(initial_set)
    current_score = sum(1.0 / (n - a) for a in current_set)
    
    # Candidates: all numbers 1..n-1
    all_candidates = list(range(1, n))
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        
        # Try adding a single element
        best_add_candidate = None
        best_add_gain = 0.0
        for a in all_candidates:
            if a not in current_set and _is_coprime_to_set(a, current_set):
                gain = 1.0 / (n - a)
                if gain > best_add_gain:
                    best_add_gain = gain
                    best_add_candidate = a
        
        if best_add_candidate is not None:
            current_set.append(best_add_candidate)
            current_score += best_add_gain
            improved = True
            continue
            
        # Try swapping one element
        # Sort current set by weight ascending (remove low weight first)
        # Sort candidates not in set by weight descending (add high weight first)
        current_sorted = sorted(current_set, key=lambda x: 1.0 / (n - x))
        candidates_not_in_set = [a for a in all_candidates if a not in current_set]
        candidates_sorted = sorted(candidates_not_in_set, key=lambda x: 1.0 / (n - x), reverse=True)
        
        # Limit search space for swaps to avoid O(N^2) being too slow if N is large, but N=511 is small.
        # We check all pairs (remove, add)
        swap_found = False
        for a_remove in current_sorted:
            if swap_found:
                break
            weight_remove = 1.0 / (n - a_remove)
            for a_add in candidates_sorted:
                # Optimization: if gain from add is less than loss from remove, we can break if candidates are sorted by weight?
                # No, because validity depends on gcd. But we can break if potential max gain is too low?
                # Let's just check.
                weight_add = 1.0 / (n - a_add)
                if weight_add <= weight_remove:
                    # Since candidates are sorted descending, if this one is not better, none after are (in terms of weight).
                    # But we might still want to swap for structural reasons? No, objective is sum of weights.
                    # If weight_add <= weight_remove, the score won't increase.
                    break
                
                # Check if a_add is coprime to (current_set \ {a_remove})
                temp_set = [x for x in current_set if x != a_remove]
                if _is_coprime_to_set(a_add, temp_set):
                    # Valid swap
                    new_score = current_score - weight_remove + weight_add
                    current_set = temp_set
                    current_set.append(a_add)
                    current_score = new_score
                    improved = True
                    swap_found = True
                    break
    
    return current_set, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    n = 512
    best_set = []
    best_score = 0.0
    
    # Strategy: Randomized Greedy + Local Search
    # Restart multiple times with different randomizations
    
    while time.time() < deadline:
        # 1. Greedy Construction with Randomization
        # Order candidates by weight descending (a descending), but add some noise
        candidates = list(range(1, n))
        
        # Randomized ordering: Sort by (weight - noise)
        # Or simply: Pick from top K random candidates
        # Let's try: Sort by weight, then shuffle the top 50% to allow diversity
        candidates_sorted = sorted(candidates, key=lambda x: 1.0 / (n - x), reverse=True)
        
        # Random perturbation: Shuffle a subset
        if random.random() < 0.5:
            # Swap some adjacent elements in the sorted list
            for _ in range(10):
                i = random.randint(0, len(candidates_sorted) - 2)
                candidates_sorted[i], candidates_sorted[i+1] = candidates_sorted[i+1], candidates_sorted[i]
        else:
            # More aggressive randomization: Shuffle the first 100
            top_k = 100
            top_part = candidates_sorted[:top_k]
            random.shuffle(top_part)
            candidates_sorted = top_part + candidates_sorted[top_k:]
            
        current_set = []
        for a in candidates_sorted:
            if _is_coprime_to_set(a, current_set):
                current_set.append(a)
        
        # 2. Local Search
        refined_set, refined_score = _local_search(current_set, n=n, time_limit=5.0)
        
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined_set
    
    return best_set
# EVOLVE-BLOCK-END
