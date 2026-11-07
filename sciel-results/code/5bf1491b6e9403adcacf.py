"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes a heuristic matrix based on 'Lookahead Greedy Packing Efficiency'.
    
    Strategy:
    For each pair (i, j) where item i is already in a bin and we consider adding item j:
    1. Check if i + j fits in capacity.
    2. If it fits, simulate how many more items from the 'remaining' set (excluding i and j) 
       could fit into the remaining space of this bin using a simple greedy First Fit strategy.
    3. The heuristic value is the total number of items in this bin (1 for i, 1 for j, plus k more).
    
    This encourages the ACO to choose pairs that not only fit together but also allow for 
    dense packing of the bin with subsequent items, reducing the total number of bins needed.
    
    To handle the 'remaining set' dependency (which changes as the bin fills), we approximate 
    the 'remaining set' for the heuristic calculation as the set of all items NOT equal to i 
    or j, sorted in descending order. This provides a static but highly informative estimate 
    of the potential density of the bin starting with the pair (i, j).
    """
    n = len(demands)
    demands = np.asarray(demands, dtype=float)
    
    # Initialize heuristic matrix with a very small value (effectively invalid/low priority)
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # Sort items in descending order for the greedy lookahead simulation
    # We need to map original indices to sorted positions or just use values.
    # Since items are distinct indices but values might repeat, we track values.
    # For the simulation, we just need the multiset of available items.
    
    # Pre-compute sorted list of all item values for quick access in simulation
    sorted_demands = np.sort(demands)[::-1]
    
    # To speed up the O(N^2 * N) simulation, we can pre-compute a 'greedy fill' table?
    # No, because the available set changes based on i and j.
    # However, N=120. N^2 = 14400 pairs.
    # For each pair, simulation is O(N log N) or O(N) if we just scan sorted list.
    # Total ops ~ 14400 * 120 = 1.7 million. This is very fast in Python/NumPy if vectorized or optimized.
    # Let's implement an efficient loop.
    
    # We want to maximize the number of items in the bin.
    # Base items: i and j.
    # Remaining capacity: capacity - demands[i] - demands[j].
    # Available items: all demands except demands[i] and demands[j].
    # Greedy fill: take largest available item that fits, repeat.
    
    # Optimization: 
    # Instead of removing i and j from the full list every time (O(N)), 
    # we can just iterate through the sorted list and skip indices i and j.
    # Wait, indices i and j refer to positions in the original array. 
    # If there are duplicate values, we must be careful. 
    # The task says "demands are integers... passed as floats". It doesn't say unique.
    # But "item-by-item matrix... placing item j right after item i". 
    # Usually in these tasks, each index is a distinct physical item.
    # So if demands[0] == demands[5], they are different items.
    
    # Let's create an array of indices sorted by demand value descending.
    # sort_indices[k] is the original index of the k-th largest item.
    sort_indices = np.argsort(demands)[::-1]
    sorted_values = demands[sort_indices]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            sum_ij = demands[i] + demands[j]
            if sum_ij > capacity:
                continue
            
            remaining_cap = capacity - sum_ij
            
            # Simulate greedy fill with remaining items
            # We iterate through the sorted list of all items.
            # We skip item i and item j.
            # We pick an item if its value <= remaining_cap.
            
            count = 2 # i and j are already in
            
            # To make this faster, we can break early if the smallest remaining item > remaining_cap?
            # The smallest item is sorted_values[-1].
            # If sorted_values[-1] > remaining_cap, we can stop.
            
            # However, we are skipping i and j.
            # Let's just loop. 120 items is small.
            
            for k in range(n):
                idx = sort_indices[k]
                if idx == i or idx == j:
                    continue
                
                val = sorted_values[k]
                if val <= remaining_cap:
                    remaining_cap -= val
                    count += 1
                    # If remaining_cap is 0, we can break
                    if remaining_cap < 1e-9:
                        break
                else:
                    # Since sorted descending, if this doesn't fit, smaller ones might.
                    # We cannot break here. We must continue checking smaller items.
                    # But if we are at the end, we stop.
                    pass
            
            # The heuristic is the total count of items in this bin.
            # Larger is better.
            eta[i, j] = float(count)
            
    # Ensure diagonal is low (self-pairing not allowed/meaningless in this context)
    np.fill_diagonal(eta, 1e-9)
    
    # Normalize? The ACO uses pheromone * heuristic.
    # If counts are 2 to 10, the relative differences are preserved.
    # We might want to scale them up to have more impact against pheromone (which starts at 1).
    # If pheromone is 1 and heuristic is 5, prob ~ 5.
    # If pheromone is 1 and heuristic is 1e-9, prob ~ 0.
    # This looks fine.
    
    return eta
# EVOLVE-BLOCK-END
