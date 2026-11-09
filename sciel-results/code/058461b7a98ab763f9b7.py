"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a // gcd(a, b) * b

def _lcm_list(lst):
    return reduce(_lcm, lst)

def construct():
    """
    Return a nonempty subset of [1,22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    
    Strategy: 
    1. Precompute integer weights w_i = LCM(1..22) / i.
    2. Use a randomized local search (Simulated Annealing) to maximize the size 
       of a dissociated subset. 
    3. Track the best valid set found.
    """
    N = 22
    universe = list(range(1, N + 1))
    
    # Precompute LCM of 1..22
    L = _lcm_list(universe)
    
    # Integer weights: w_i = L // i
    weights = {i: L // i for i in universe}

    def evaluate_set(subset):
        """
        Evaluate a subset.
        Returns:
        (is_valid, size, min_gap)
        """
        k = len(subset)
        if k == 0:
            return True, 0, float('inf')
        
        sums = {0}
        for idx in subset:
            w = weights[idx]
            # Optimization: if len(sums) is already > 2^k, it's invalid? 
            # No, we build it up.
            new_sums = set()
            for s in sums:
                ns = s + w
                # Early termination: if ns is already in sums or new_sums, 
                # we have a collision. 
                # However, checking `in` for a set is O(1).
                # But we need to finish the loop to know if it's valid?
                # Actually, if we find a collision, we can return False immediately.
                if ns in sums or ns in new_sums:
                    return False, k, 0
                new_sums.add(ns)
            sums.update(new_sums)
            
        # If we reach here, no collisions found in the construction process?
        # Wait, the collision check above only checks if `ns` collides with 
        # existing `sums` or `new_sums`. 
        # This is sufficient to detect if all subset sums are distinct.
        # If a collision happens at any step, the set is not dissociated.
        
        # Calculate min gap
        sorted_sums = sorted(sums)
        min_gap = float('inf')
        for i in range(len(sorted_sums) - 1):
            gap = sorted_sums[i+1] - sorted_sums[i]
            if gap < min_gap:
                min_gap = gap
        
        return True, k, min_gap

    # Greedy initialization
    def greedy_construct(order):
        S = []
        for i in order:
            candidate = S + [i]
            valid, _, _ = evaluate_set(candidate)
            if valid:
                S = candidate
        return S

    # Find a good initial set using multiple random greedy passes
    best_set = []
    best_size = 0
    best_gap = 0
    
    for trial in range(30):
        order = universe[:]
        random.shuffle(order)
        S = greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            _, _, gap = evaluate_set(S)
            best_gap = gap
        elif len(S) == best_size:
            _, _, gap = evaluate_set(S)
            if gap > best_gap:
                best_set = S[:]
                best_gap = gap

    current_set = best_set[:]
    current_size = len(current_set)
    current_gap = best_gap
    
    # Simulated Annealing
    start_time = time.time()
    timeout = 55.0  # Leave 5s buffer
    
    T = 10.0
    cooling = 0.999
    
    while time.time() - start_time < timeout:
        if T < 0.1:
            break
            
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Add
            candidates_to_add = [i for i in universe if i not in current_set]
            if not candidates_to_add:
                continue
            i = random.choice(candidates_to_add)
            new_set = current_set + [i]
        elif move_type < 0.6:
            # Remove
            if len(current_set) <= 1:
                continue
            i = random.choice(current_set)
            new_set = [x for x in current_set if x != i]
        else:
            # Swap
            if len(current_set) == 0:
                continue
            i_remove = random.choice(current_set)
            candidates_to_add = [i for i in universe if i not in current_set]
            if not candidates_to_add:
                continue
            i_add = random.choice(candidates_to_add)
            new_set = [x for x in current_set if x != i_remove] + [i_add]
            
        # Evaluate new set
        new_valid, new_size, new_gap = evaluate_set(new_set)
        
        if new_valid:
            # Update best if better
            if new_size > best_size or (new_size == best_size and new_gap > best_gap):
                best_size = new_size
                best_set = new_set[:]
                best_gap = new_gap
                
            # Acceptance criteria for current state
            if new_size > current_size:
                current_set = new_set
                current_size = new_size
                current_gap = new_gap
            elif new_size == current_size:
                if new_gap > current_gap:
                    current_set = new_set
                    current_gap = new_gap
                else:
                    delta = current_gap - new_gap
                    if random.random() < pow(2.71828, -delta / T):
                        current_set = new_set
                        current_gap = new_gap
            else:
                # Size decreased
                delta = (current_size - new_size) * 1000
                if random.random() < pow(2.71828, -delta / T):
                    current_set = new_set
                    current_size = new_size
                    current_gap = new_gap
        else:
            # Invalid set: Reject
            pass
            
        T *= cooling

    # Return the best set found
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
# EVOLVE-BLOCK-END
