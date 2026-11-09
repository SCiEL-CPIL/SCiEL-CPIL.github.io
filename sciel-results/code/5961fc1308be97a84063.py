"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # seconds, safety margin before 60s kill
    
    def elapsed():
        return time.time() - start_time

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Precompute conflict matrix as a boolean numpy array
    # conflict[i, j] is True if lcm(numbers[i], numbers[j]) <= 256
    conflict = np.zeros((n, n), dtype=bool)
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict[i, j] = True
                conflict[j, i] = True
                
    # Weights: reciprocal sum. w[i] = 1.0 / numbers[i]
    weights = np.array([1.0 / x for x in numbers])
    
    # Greedy Initial Solution
    # Sort by weight descending (i.e., number ascending)
    order = np.argsort(-weights)
    
    selected_mask = np.zeros(n, dtype=bool)
    current_sum = 0.0
    
    for i in order:
        # Check if i conflicts with any selected node
        # conflict[i] gives a boolean vector of conflicts
        if not np.any(conflict[i] & selected_mask):
            selected_mask[i] = True
            current_sum += weights[i]
            
    best_sum = current_sum
    best_mask = selected_mask.copy()
    
    # Simulated Annealing
    # Parameters
    T_start = 0.05
    T_end = 0.001
    alpha = 0.9995  # Cooling rate
    T = T_start
    
    # Precompute neighbors? No, random sampling is better for SA.
    # To speed up conflict checking:
    # When adding i: check conflict[i] & selected_mask
    # When removing i: just flip bit
    # When swapping u, v: 
    #   Remove u: mask' = mask & ~bit_u
    #   Add v: check conflict[v] & mask'
    
    rng = np.random.default_rng(42)
    
    while T > T_end and elapsed() < DEADLINE:
        # Generate a random move
        # Types:
        # 1. Add a random unselected node
        # 2. Remove a random selected node
        # 3. Swap a selected node with an unselected node
        
        # Count selected and unselected
        sel_count = np.sum(selected_mask)
        unsel_count = n - sel_count
        
        # Choose move type based on availability
        if sel_count == 0:
            move_type = 1
        elif unsel_count == 0:
            move_type = 2
        else:
            # 50% add, 20% remove, 30% swap?
            # Or just random choice among available
            r = rng.random()
            if r < 0.5:
                move_type = 1
            elif r < 0.7:
                move_type = 2
            else:
                move_type = 3
        
        # Execute move
        if move_type == 1:
            # Add
            unsel_indices = np.where(~selected_mask)[0]
            i = unsel_indices[rng.integers(len(unsel_indices))]
            
            # Check validity
            if np.any(conflict[i] & selected_mask):
                continue # Invalid move, skip (or accept with 0 prob? Usually skip)
            
            new_mask = selected_mask.copy()
            new_mask[i] = True
            delta_sum = weights[i]
            
        elif move_type == 2:
            # Remove
            sel_indices = np.where(selected_mask)[0]
            i = sel_indices[rng.integers(len(sel_indices))]
            
            new_mask = selected_mask.copy()
            new_mask[i] = False
            delta_sum = -weights[i]
            
        else:
            # Swap
            sel_indices = np.where(selected_mask)[0]
            unsel_indices = np.where(~selected_mask)[0]
            
            u = sel_indices[rng.integers(len(sel_indices))]
            v = unsel_indices[rng.integers(len(unsel_indices))]
            
            # Check if v can be added to (selected \ {u})
            temp_mask = selected_mask.copy()
            temp_mask[u] = False
            if np.any(conflict[v] & temp_mask):
                continue # Invalid swap
            
            new_mask = selected_mask.copy()
            new_mask[u] = False
            new_mask[v] = True
            delta_sum = weights[v] - weights[u]
            
        new_sum = current_sum + delta_sum
        
        # Acceptance criterion
        if new_sum > current_sum:
            selected_mask = new_mask
            current_sum = new_sum
            if current_sum > best_sum:
                best_sum = current_sum
                best_mask = selected_mask.copy()
        else:
            # Metropolis acceptance
            # delta_sum is negative
            p = np.exp(delta_sum / T)
            if rng.random() < p:
                selected_mask = new_mask
                current_sum = new_sum
                
        T *= alpha
        
    # Return best found
    result = [numbers[i] for i in np.where(best_mask)[0]]
    return result
# EVOLVE-BLOCK-END
