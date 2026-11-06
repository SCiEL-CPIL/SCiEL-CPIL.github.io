"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Denominators are 2..48, indices 0..47
    n = 47
    denominators = list(range(2, 49))
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # Initial greedy solution: pick as many as possible without reaching 1
    # We want to minimize deficit = 1 - sum, so we want sum as close to 1 from below as possible.
    # Greedy: sort by value descending? No, we want to fill up to 1.
    # Standard greedy for subset sum: sort by value descending, pick if fits.
    # But here we want to get close to 1.
    # Let's try a few heuristics to get a good starting point.
    
    best_indices = []
    best_sum = Fraction(0)
    
    # Heuristic 1: Greedy by descending value (largest reciprocals first)
    # This tends to overshoot quickly, but we skip if it reaches 1.
    # Actually, for "strictly below 1", we want to fill the space.
    # Let's try: sort by value descending.
    idx_by_val_desc = sorted(range(n), key=lambda i: reciprocals[i], reverse=True)
    curr_indices = []
    curr_sum = Fraction(0)
    for i in idx_by_val_desc:
        if curr_sum + reciprocals[i] < 1:
            curr_indices.append(i)
            curr_sum += reciprocals[i]
    if len(curr_indices) > len(best_indices):
        best_indices = curr_indices
        best_sum = curr_sum
    
    # Heuristic 2: Greedy by ascending value (smallest reciprocals first)
    # This picks many small items.
    idx_by_val_asc = sorted(range(n), key=lambda i: reciprocals[i])
    curr_indices = []
    curr_sum = Fraction(0)
    for i in idx_by_val_asc:
        if curr_sum + reciprocals[i] < 1:
            curr_indices.append(i)
            curr_sum += reciprocals[i]
    if len(curr_indices) > len(best_indices):
        best_indices = curr_indices
        best_sum = curr_sum
        
    # Heuristic 3: Randomized local search / Hill Climbing
    # Start from best found so far.
    # Try to add/remove/swap elements to improve sum (get closer to 1 from below).
    
    def get_sum(indices):
        s = Fraction(0)
        for i in indices:
            s += reciprocals[i]
        return s

    # Use a set for fast lookup
    best_set = set(best_indices)
    
    # Local Search: Randomly perturb and improve
    # We will run a simple hill-climbing with random restarts or just local moves.
    # Move types:
    # 1. Add an index not in set if sum < 1
    # 2. Remove an index in set (to allow adding larger ones? or just to explore)
    # 3. Swap an index in set with one not in set
    
    # To make it fast, we track current sum.
    curr_set = set(best_indices)
    curr_sum = best_sum
    
    # Precompute reciprocals for fast access
    # We have reciprocals list.
    
    # Optimization: Use floating point for quick filtering, exact for final check?
    # Fractions are slow. Let's use floats for heuristic search, then verify with Fraction.
    float_recips = [1.0/d for d in denominators]
    
    def float_sum(indices):
        return sum(float_recips[i] for i in indices)
    
    # Restart local search from best_set
    # We'll do a simple random local search.
    import random
    random.seed(42)
    
    max_iter = 100000
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Choose a random move
        # 1. Try to add a random element
        # 2. Try to remove a random element
        # 3. Try to swap
        
        r = random.random()
        
        if r < 0.5:
            # Add
            available = [i for i in range(n) if i not in curr_set]
            if not available:
                continue
            i = random.choice(available)
            new_sum_f = float_sum(curr_set) + float_recips[i]
            if new_sum_f < 1.0:
                # Check exact
                new_sum_e = curr_sum + reciprocals[i]
                if new_sum_e < 1:
                    # Accept if better (closer to 1)
                    if new_sum_e > curr_sum:
                        curr_set.add(i)
                        curr_sum = new_sum_e
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_set = set(curr_set)
        elif r < 0.75:
            # Remove
            if not curr_set:
                continue
            i = random.choice(list(curr_set))
            new_sum_e = curr_sum - reciprocals[i]
            # Removing always decreases sum, so it's never "better" in terms of maximizing sum.
            # But it might allow adding a larger one later?
            # For pure hill climbing on sum, removal is only useful if followed by addition.
            # Let's skip pure removal in this simple loop, or use it for restarts.
            pass
        else:
            # Swap
            if not curr_set:
                continue
            i_in = random.choice(list(curr_set))
            available = [i for i in range(n) if i not in curr_set]
            if not available:
                continue
            i_out = random.choice(available)
            
            # Check if swap is valid and improves sum
            delta_f = float_recips[i_out] - float_recips[i_in]
            new_sum_f = float_sum(curr_set) + delta_f
            if new_sum_f < 1.0:
                new_sum_e = curr_sum + reciprocals[i_out] - reciprocals[i_in]
                if new_sum_e < 1:
                    if new_sum_e > curr_sum:
                        curr_set.remove(i_in)
                        curr_set.add(i_out)
                        curr_sum = new_sum_e
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_set = set(curr_set)
    
    # Final verification and return
    # Ensure best_set is valid
    final_sum = Fraction(0)
    for i in best_set:
        final_sum += reciprocals[i]
        
    if final_sum >= 1:
        # Fallback to a valid small set if something went wrong
        # e.g. {0} -> 1/2
        return [0]
        
    return list(best_set)

# EVOLVE-BLOCK-END
