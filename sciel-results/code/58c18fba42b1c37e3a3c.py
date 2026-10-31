"""Rewritten program: Dual-view optimization via Hill Climbing."""
import numpy as np
import math
import random

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0 to maximize
    the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1).
    
    Strategy: Use a "dual-view" optimization approach. 
    We view the problem as optimizing the ratio of difference set size to sum set size 
    relative to the diameter. We use a Hill Climbing / Simulated Annealing approach 
    to refine initial candidate sets, exploring the space of sets with fixed or 
    varying sizes and diameters.
    """
    
    def compute_score(U):
        if not U or 0 not in U:
            return 0.0
        U_set = set(U)
        max_u = max(U_set)
        if max_u == 0:
            return 0.0
        # Compute |U+U| and |U-U|
        sum_set = set()
        diff_set = set()
        for a in U_set:
            for b in U_set:
                sum_set.add(a + b)
                diff_set.add(a - b)
        n_diff = len(diff_set)
        n_sum = len(sum_set)
        if n_sum == 0:
            return 0.0
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        return 1.0 + math.log(n_diff / n_sum) / log_denom

    def local_search(initial_U, max_steps=500, max_val=200, min_size=3, max_size=20):
        """
        Refine a set U using local search.
        Moves: Add, Remove, Swap.
        """
        current_U = set(initial_U)
        current_score = compute_score(current_U)
        
        # Ensure 0 is in the set
        if 0 not in current_U:
            current_U.add(0)
        
        for step in range(max_steps):
            best_neighbor = None
            best_neighbor_score = current_score
            
            # Generate candidates for neighbors
            # 1. Add a random element
            for _ in range(10):
                candidate = set(current_U)
                if len(candidate) < max_size:
                    new_elem = random.randint(1, max_val)
                    if new_elem not in candidate:
                        candidate.add(new_elem)
                        score = compute_score(candidate)
                        if score > best_neighbor_score:
                            best_neighbor_score = score
                            best_neighbor = candidate
            
            # 2. Remove a random element (not 0)
            for _ in range(10):
                candidate = set(current_U)
                removable = [x for x in candidate if x != 0]
                if removable and len(candidate) > min_size:
                    rem_elem = random.choice(removable)
                    candidate.remove(rem_elem)
                    score = compute_score(candidate)
                    if score > best_neighbor_score:
                        best_neighbor_score = score
                        best_neighbor = candidate
            
            # 3. Swap an element
            for _ in range(10):
                candidate = set(current_U)
                removable = [x for x in candidate if x != 0]
                if removable:
                    old_elem = random.choice(removable)
                    new_elem = random.randint(1, max_val)
                    if new_elem not in candidate:
                        candidate.remove(old_elem)
                        candidate.add(new_elem)
                        score = compute_score(candidate)
                        if score > best_neighbor_score:
                            best_neighbor_score = score
                            best_neighbor = candidate

            if best_neighbor is not None:
                current_U = best_neighbor
                current_score = best_neighbor_score
            else:
                # No improvement, try a random jump (Simulated Annealing style)
                # or break if stuck for too long (simplified: break)
                # To avoid infinite loops, we break if no improvement found in a batch
                # A more robust SA would accept worse moves with probability exp(-delta/T)
                # Here we just break to save time, relying on multiple restarts
                break
                
        return sorted(current_U), current_score

    # Generate diverse initial seeds
    seeds = []
    
    # 1. Arithmetic Progressions
    for n in range(5, 25):
        for step in [1, 2, 3, 5, 10]:
            seeds.append(list(range(0, n * step, step)))
            
    # 2. Powers of 2
    seeds.append([2**i for i in range(10)])
    seeds.append([2**i for i in range(12)])
    
    # 3. Random sets
    random.seed(42)
    for _ in range(20):
        n = random.randint(5, 20)
        max_v = random.randint(50, 300)
        # Ensure 0 is included
        elems = {0}
        while len(elems) < n:
            elems.add(random.randint(1, max_v))
        seeds.append(sorted(elems))
        
    # 4. "Sidon-like" or structured sets
    # Sequence: 0, 1, 4, 10, 18, 23, 31...
    seeds.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86])
    seeds.append([0, 1, 3, 7, 13, 21, 31, 43, 57, 73])
    
    best_U = [0, 1, 2]
    best_score = 0.0
    
    for seed in seeds:
        # Filter invalid seeds
        if 0 not in seed:
            continue
            
        # Run local search with different parameters
        # We can vary max_val to explore different diameters
        for max_val in [100, 200, 500]:
            U, score = local_search(seed, max_steps=200, max_val=max_val)
            if score > best_score:
                best_score = score
                best_U = U
                
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
