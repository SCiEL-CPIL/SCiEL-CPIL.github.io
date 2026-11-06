"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    n = 16
    # 1. Generate all possible triples (indices 0..559)
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # 2. Generate all K(2,2,2) configurations
    triple_to_ks = [[] for _ in range(num_triples)]
    all_ks_masks = []
    
    # Hardcoded 15 matchings of indices 0-5
    matchings = [
        [(0,1),(2,3),(4,5)],
        [(0,1),(2,4),(3,5)],
        [(0,1),(2,5),(3,4)],
        [(0,2),(1,3),(4,5)],
        [(0,2),(1,4),(3,5)],
        [(0,2),(1,5),(3,4)],
        [(0,3),(1,2),(4,5)],
        [(0,3),(1,4),(2,5)],
        [(0,3),(1,5),(2,4)],
        [(0,4),(1,2),(3,5)],
        [(0,4),(1,3),(2,5)],
        [(0,4),(1,5),(2,3)],
        [(0,5),(1,2),(3,4)],
        [(0,5),(1,3),(2,4)],
        [(0,5),(1,4),(2,3)]
    ]
    
    # Precompute bits for each triple index
    triple_bits = [1 << i for i in range(num_triples)]
    
    for six_points in combinations(range(n), 6):
        for matching in matchings:
            p1 = (six_points[matching[0][0]], six_points[matching[0][1]])
            p2 = (six_points[matching[1][0]], six_points[matching[1][1]])
            p3 = (six_points[matching[2][0]], six_points[matching[2][1]])
            
            # Generate 8 triple indices
            t_indices = []
            mask = 0
            for x1 in p1:
                for x2 in p2:
                    for x3 in p3:
                        t = tuple(sorted((x1, x2, x3)))
                        idx = triple_to_idx[t]
                        t_indices.append(idx)
                        mask |= triple_bits[idx]
            
            k_idx = len(all_ks_masks)
            all_ks_masks.append(mask)
            
            for ti in t_indices:
                triple_to_ks[ti].append(k_idx)
    
    num_ks = len(all_ks_masks)
    
    # Helper to check if a triple can be added
    def can_add(idx, current_mask):
        bit = triple_bits[idx]
        for k_idx in triple_to_ks[idx]:
            k_mask = all_ks_masks[k_idx]
            if (k_mask & current_mask) == (k_mask ^ bit):
                return False
        return True

    # Helper to perform greedy fill
    def greedy_fill(indices, current_mask):
        for idx in indices:
            if can_add(idx, current_mask):
                current_mask |= triple_bits[idx]
        return current_mask

    # Max-SAT style local search with clause count tracking
    # State: boolean array x[i] = 1 if triple i is selected
    # Objective: maximize sum(x[i]) subject to no clause fully satisfied
    
    # Precompute for each triple, the list of clauses it appears in
    # and for each clause, the 8 triples it contains
    
    # clause_triples[k] = list of 8 triple indices in clause k
    clause_triples = []
    for k in range(num_ks):
        mask = all_ks_masks[k]
        triples_in_clause = []
        for i in range(num_triples):
            if mask & triple_bits[i]:
                triples_in_clause.append(i)
        clause_triples.append(triples_in_clause)
    
    # Initialize with greedy
    best_mask = 0
    best_count = 0
    
    # Multiple greedy restarts
    while time.time() < deadline - 30.0:
        current_mask = 0
        indices = list(range(num_triples))
        random.shuffle(indices)
        current_mask = greedy_fill(indices, current_mask)
        
        count = current_mask.bit_count()
        if count > best_count:
            best_count = count
            best_mask = current_mask
    
    # Now do local search: try to add triples that don't violate constraints
    # and remove triples that are "blocking" many others
    
    # For efficient local search, track for each unselected triple how many
    # clauses it's blocked in
    
    # Simulated annealing with kick restarts
    temperature = 100.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if best_count == 0:
            break
            
        current_mask = best_mask
        
        # Compute blocked counts for each unselected triple
        # blocked_count[i] = number of clauses where adding i would complete the clause
        blocked_count = [0] * num_triples
        for i in range(num_triples):
            if not (current_mask & triple_bits[i]):
                for k_idx in triple_to_ks[i]:
                    k_mask = all_ks_masks[k_idx]
                    # Check if all other 7 triples in this clause are already selected
                    other_mask = k_mask ^ triple_bits[i]
                    if (other_mask & current_mask) == other_mask:
                        blocked_count[i] += 1
        
        # Try single flips: add an unblocked triple, or remove a triple to unblock others
        improved = False
        
        # Try adding unblocked triples
        addable = [i for i in range(num_triples) if not (current_mask & triple_bits[i]) and blocked_count[i] == 0]
        if addable:
            random.shuffle(addable)
            for idx in addable:
                new_mask = current_mask | triple_bits[idx]
                # Check validity (should be valid since blocked_count was 0)
                # But re-check for safety
                valid = True
                for k_idx in triple_to_ks[idx]:
                    k_mask = all_ks_masks[k_idx]
                    if (k_mask & new_mask) == k_mask:
                        valid = False
                        break
                if valid:
                    current_mask = new_mask
                    improved = True
                    break
        
        # If no improvement, try a kick: remove some triples and re-greedy
        if not improved:
            kick_size = random.randint(3, 12)
            selected_indices = [i for i in range(num_triples) if (current_mask & triple_bits[i])]
            
            if len(selected_indices) < kick_size:
                continue
            
            kick_indices = random.sample(selected_indices, kick_size)
            
            new_mask = current_mask
            for idx in kick_indices:
                new_mask ^= triple_bits[idx]
            
            free_indices = [i for i in range(num_triples) if not (new_mask & triple_bits[i])]
            random.shuffle(free_indices)
            new_mask = greedy_fill(free_indices, new_mask)
            
            new_count = new_mask.bit_count()
            if new_count > best_count:
                best_count = new_count
                best_mask = new_mask
            else:
                # Accept with probability based on temperature (simulated annealing)
                delta = best_count - new_count
                if delta > 0:
                    accept_prob = min(1.0, temperature / max(1, delta))
                    if random.random() < accept_prob:
                        best_mask = new_mask
                        best_count = new_count
        
        temperature *= cooling_rate
        if temperature < 1.0:
            temperature = 1.0
    
    # Final polish: try to add any remaining valid triples
    for i in range(num_triples):
        if not (best_mask & triple_bits[i]):
            if can_add(i, best_mask):
                best_mask |= triple_bits[i]
                best_count = best_mask.bit_count()
    
    # Convert best_mask back to list of triples
    result = []
    for i in range(num_triples):
        if best_mask & triple_bits[i]:
            result.append(list(all_triples[i]))
            
    return result
# EVOLVE-BLOCK-END
