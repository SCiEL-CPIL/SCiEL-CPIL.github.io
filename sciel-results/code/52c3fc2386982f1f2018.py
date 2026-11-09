"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
import time
import random

def construct():
    N = 512
    size = 511  # numbers 1..511
    # Indices 0..510 represent numbers 1..511
    
    # Precompute coprime bitmasks
    # mask[i] is a bitset where bit j is 1 if gcd(i+1, j+1) == 1
    masks = [0] * size
    for i in range(size):
        a = i + 1
        m = 0
        for j in range(size):
            b = j + 1
            if gcd(a, b) == 1:
                m |= (1 << j)
        masks[i] = m

    # Score function
    def get_score(chosen_indices):
        s = 0.0
        for idx in chosen_indices:
            a = idx + 1
            s += 1.0 / (N - a)
        return s

    # Initialization: Greedy from high to low
    chosen_set = set()
    current_mask = 0
    for i in range(size - 1, -1, -1):
        # Check if i is coprime to all in chosen_set
        # This is equivalent to checking if masks[i] covers all bits in current_mask
        if (masks[i] & current_mask) == current_mask:
            chosen_set.add(i)
            current_mask |= (1 << i)
    
    current_score = get_score(list(chosen_set))
    best_set = set(chosen_set)
    best_score = current_score

    # Simulated Annealing
    start_time = time.time()
    T = 1.0
    T_min = 1e-6
    alpha = 0.9999
    max_time = 110.0 # Leave 10s buffer

    while T > T_min and (time.time() - start_time) < max_time:
        # Generate a move
        # 1. Pick a random element to remove (if set not empty)
        # 2. Pick a random element to add (if not in set)
        # Try Swap: Remove r, Add a
        if len(chosen_set) == 0:
            # Add random coprime
            candidates = [j for j in range(size) if j not in chosen_set]
            if not candidates:
                break
            a = random.choice(candidates)
            if (masks[a] & current_mask) == current_mask:
                chosen_set.add(a)
                current_mask |= (1 << a)
                current_score += 1.0 / (N - (a + 1))
        else:
            # Pick random r from chosen_set
            r = random.choice(list(chosen_set))
            # Pick random a from not chosen_set
            # To avoid generating too many invalid candidates, we can just pick random a and check
            a = random.randint(0, size - 1)
            
            # Check if a is already in set
            if a in chosen_set:
                continue
                
            # Check if a is coprime to all in (chosen_set - {r})
            # new_mask_needed = current_mask without bit r
            temp_mask = current_mask & ~(1 << r)
            
            if (masks[a] & temp_mask) == temp_mask:
                # Valid move
                old_val = 1.0 / (N - (r + 1))
                new_val = 1.0 / (N - (a + 1))
                delta_score = new_val - old_val
                
                # Metropolis acceptance
                if delta_score > 0 or random.random() < np.exp(delta_score / T):
                    # Accept
                    chosen_set.remove(r)
                    chosen_set.add(a)
                    current_mask = temp_mask | (1 << a)
                    current_score += delta_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(chosen_set)
            else:
                # Try Add only (if we removed r, maybe we can add a different one? 
                # Actually, swap is the main driver. Let's also allow pure Add/Remove to change set size)
                pass
            
            # Occasionally try pure Add or Remove to adjust size?
            # The problem doesn't constrain the size, only that they are distinct and pairwise coprime.
            # Larger sets generally have smaller terms (denominator 512-a).
            # To maximize sum, we want large 'a' (small denominator).
            # Greedy picks large 'a'. 
            # Sometimes removing a small 'a' (large denominator, small contribution) allows adding a large 'a' that was blocked.
            # The swap move handles this.
            # What if we can just ADD a number without removing one?
            # Check if there is any number not in set that is coprime to all in set.
            # This is expensive to check every step. 
            # However, if we start with a maximal independent set (greedy), we might not be able to add more.
            # But maybe we can remove a "bad" number and add TWO "good" numbers?
            # The current swap is 1-for-1. 
            # Let's stick to 1-for-1 swaps and 1-for-0 adds if possible.
            
            # Let's add a check for pure add with low probability to escape size traps
            if random.random() < 0.1:
                # Try to add a random candidate
                a = random.randint(0, size - 1)
                if a not in chosen_set:
                    if (masks[a] & current_mask) == current_mask:
                        chosen_set.add(a)
                        current_mask |= (1 << a)
                        current_score += 1.0 / (N - (a + 1))
                        if current_score > best_score:
                            best_score = current_score
                            best_set = set(chosen_set)
            
            # Try to remove a random element if it has low value? 
            # Usually removing lowers score, so only accept if it allows future adds.
            # SA will handle this if we allow negative delta.
            # But we need a move that removes.
            if random.random() < 0.1:
                r = random.choice(list(chosen_set))
                old_val = 1.0 / (N - (r + 1))
                delta_score = -old_val
                if delta_score > 0 or random.random() < np.exp(delta_score / T):
                    chosen_set.remove(r)
                    current_mask &= ~(1 << r)
                    current_score += delta_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(chosen_set)

        T *= alpha

    # Final greedy polish on best_set
    # Try to add any remaining coprime numbers to best_set
    best_list = list(best_set)
    best_mask = 0
    for idx in best_list:
        best_mask |= (1 << idx)
        
    # Try to add more
    for i in range(size):
        if i not in best_set:
            if (masks[i] & best_mask) == best_mask:
                best_set.add(i)
                best_mask |= (1 << i)
                
    # Try to improve via simple swaps on best_set
    # Re-run a quick local search on the best found set
    improved = True
    while improved:
        improved = False
        best_list = list(best_set)
        for r in best_list:
            temp_mask = best_mask & ~(1 << r)
            best_r_val = 1.0 / (N - (r + 1))
            best_gain = 0
            best_a = -1
            for a in range(size):
                if a in best_set:
                    continue
                if (masks[a] & temp_mask) == temp_mask:
                    gain = 1.0 / (N - (a + 1)) - best_r_val
                    if gain > best_gain:
                        best_gain = gain
                        best_a = a
            if best_a != -1 and best_gain > 1e-12:
                best_set.remove(r)
                best_set.add(best_a)
                best_mask = temp_mask | (1 << best_a)
                improved = True
                break # restart

    return [int(i + 1) for i in best_set]
# EVOLVE-BLOCK-END
