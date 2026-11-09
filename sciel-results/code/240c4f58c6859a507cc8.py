"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We stop at 130s to be safe.
    deadline = start_time + 130.0
    
    max_n = 511
    # Weights: 1 / (512 - a)
    weights = [0.0] * (max_n + 1)
    for a in range(1, max_n + 1):
        weights[a] = 1.0 / (512.0 - a)
        
    # Precompute prime factors to speed up gcd checks? 
    # Actually, gcd is fast. But let's precompute a coprime matrix? 
    # 511x511 matrix of booleans is ~260k entries. 
    # Python list of lists is slow. 
    # Let's stick to gcd but optimize the loop.
    
    def is_coprime_to_all(a, others):
        for b in others:
            if gcd(a, b) != 1:
                return False
        return True

    # 1. Greedy Initialization
    # Pick largest weights (largest a) first
    chosen = [1]
    for a in range(max_n, 1, -1):
        if is_coprime_to_all(a, chosen):
            chosen.append(chosen) # Bug in thought process, append a
            chosen.append(a)
            # Wait, I wrote append twice in thought. 
            # Let's rewrite clean.
            
    # Correct Greedy
    chosen = [1]
    for a in range(max_n, 1, -1):
        if is_coprime_to_all(a, chosen):
            chosen.append(a)
            
    best_solution = chosen
    best_score = sum(weights[a] for a in best_solution)
    
    def get_score(sol):
        return sum(weights[a] for a in sol)
        
    # Local Search
    # We will try 1-for-1 and 1-for-2 swaps.
    
    while time.time() < deadline:
        improved = False
        current = best_solution
        current_score = best_score
        
        # --- 1-for-1 Swaps ---
        # For each element in current, try to replace it with a better one
        # Sort current by weight descending to prioritize replacing low-weight elements?
        # Or just iterate all.
        
        for i, a_out in enumerate(current):
            # Remove a_out
            temp_set = [x for j, x in enumerate(current) if j != i]
            
            # Find best b_in to add
            best_gain = 0
            best_b_in = None
            
            # We can restrict search to b_in with weight > a_out weight
            # Since we want gain > 0
            for b_in in range(2, max_n + 1):
                if b_in in temp_set:
                    continue
                if weights[b_in] <= weights[a_out]:
                    continue
                if is_coprime_to_all(b_in, temp_set):
                    gain = weights[b_in] - weights[a_out]
                    if gain > best_gain:
                        best_gain = gain
                        best_b_in = b_in
                        
            if best_b_in is not None:
                new_set = temp_set + [best_b_in]
                new_score = current_score + best_gain
                if new_score > best_score:
                    best_score = new_score
                    best_solution = new_set
                    improved = True
                    # Break inner loop, restart outer with new best
                    break
        
        if improved:
            continue
            
        # --- 1-for-2 Swaps ---
        # Remove one element, add two
        for i, a_out in enumerate(current):
            temp_set = [x for j, x in enumerate(current) if j != i]
            
            # Find best pair (b1, b2)
            best_gain = 0
            best_pair = None
            
            # Optimization: Only consider pairs where sum of weights > weight of a_out
            # And b1, b2 must be coprime to each other and to temp_set
            
            # To speed up, we can iterate b1, then b2
            # This is O(N^2) per removal.
            # N=100, so 10k checks per removal. 100 removals -> 1M checks.
            
            for b1 in range(2, max_n + 1):
                if b1 in temp_set:
                    continue
                if weights[b1] <= weights[a_out]:
                    # Even with b2=0, gain is negative? 
                    # Not necessarily, if b2 is huge. But b2 max is 1.
                    # weights[b1] + weights[b2] > weights[a_out]
                    # So we can prune if weights[b1] + 1.0 <= weights[a_out]
                    if weights[b1] + 1.0 <= weights[a_out]:
                        continue
                
                if not is_coprime_to_all(b1, temp_set):
                    continue
                
                for b2 in range(b1 + 1, max_n + 1):
                    if b2 in temp_set:
                        continue
                    if weights[b1] + weights[b2] <= weights[a_out]:
                        continue
                    if not is_coprime_to_all(b2, temp_set):
                        continue
                    if gcd(b1, b2) != 1:
                        continue
                    
                    gain = (weights[b1] + weights[b2]) - weights[a_out]
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (b1, b2)
                        
            if best_pair is not None:
                new_set = temp_set + [best_pair[0], best_pair[1]]
                new_score = current_score + best_gain
                if new_score > best_score:
                    best_score = new_score
                    best_solution = new_set
                    improved = True
                    break
        
        if not improved:
            # Stuck in local optimum. Try random perturbation.
            # Remove 2 random, add 2 random valid
            if len(best_solution) > 4:
                # Remove 2
                idxs = random.sample(range(len(best_solution)), 2)
                to_remove = {best_solution[idxs[0]], best_solution[idxs[1]]}
                perturbed = [x for x in best_solution if x not in to_remove]
                
                # Add 2
                added = 0
                # Try random candidates
                candidates = random.sample(range(2, max_n + 1), 200)
                for b_in in candidates:
                    if b_in in perturbed:
                        continue
                    if is_coprime_to_all(b_in, perturbed):
                        perturbed.append(b_in)
                        added += 1
                        if added == 2:
                            break
                
                perturbed_score = get_score(perturbed)
                if perturbed_score > best_score:
                    best_score = perturbed_score
                    best_solution = perturbed
                    # Don't break, continue loop to see if we can improve from here
                else:
                    # If perturbation failed, we might be in a very good spot.
                    # To avoid infinite loop of failed perturbations, 
                    # we can break or reduce frequency.
                    # For now, just break if no improvement in perturbation?
                    # No, let's just break out of the while loop if we've been stuck a while?
                    # Simpler: just break.
                    break
                    
    return best_solution
# EVOLVE-BLOCK-END
