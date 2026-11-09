"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n = 16
    points = list(range(n))
    
    # Generate all C(16,3) = 560 triples
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Map triple to index
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute constraints (K(2,2,2)s)
    # A constraint is a set of 8 triples forming a K(2,2,2)
    # We represent each constraint as a bitset (integer)
    # We also need a list of constraints for each triple to quickly check conflicts
    
    constraints = []
    triple_to_constraints = [[] for _ in range(num_triples)]
    
    # Generate all 6-point subsets
    # For each 6-point subset, generate all 15 pairings into 3 pairs
    # For each pairing, form the 8 transversal triples
    
    # Optimization: Precompute pairings for 6 points?
    # The points are dynamic, so we just iterate.
    
    # To speed up, we can just iterate over all 3 disjoint pairs directly?
    # C(16,2) * C(14,2) * C(12,2) / 6 is large.
    # C(16,6) * 15 is better.
    
    for combo in combinations(range(n), 6):
        # Generate all 15 perfect matchings of 6 points
        # Points: p0, p1, p2, p3, p4, p5
        # p0 must pair with one of p1..p5 (5 choices)
        # p1 (if not paired with p0) pairs with one of remaining 3 (3 choices)
        # p2 (if not paired) pairs with last one (1 choice)
        
        # We can generate these recursively or hardcoded.
        # Let's use a simple generator for matchings of 6 elements.
        # Elements: 0,1,2,3,4,5
        # Matching is a set of 3 pairs.
        
        # Hardcoding the 15 matchings for indices 0-5 is efficient.
        # 0-1, 2-3, 4-5
        # 0-1, 2-4, 3-5
        # 0-1, 2-5, 3-4
        # 0-2, 1-3, 4-5
        # 0-2, 1-4, 3-5
        # 0-2, 1-5, 3-4
        # 0-3, 1-2, 4-5
        # 0-3, 1-4, 2-5
        # 0-3, 1-5, 2-4
        # 0-4, 1-2, 3-5
        # 0-4, 1-3, 2-5
        # 0-4, 1-5, 2-3
        # 0-5, 1-2, 3-4
        # 0-5, 1-3, 2-4
        # 0-5, 1-4, 2-3
        
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
        
        for m in matchings:
            # Map local indices to actual points
            # m contains pairs of indices 0..5.
            # We need to get the actual points.
            actual_pairs = []
            for idx1, idx2 in m:
                pt1 = combo[idx1]
                pt2 = combo[idx2]
                actual_pairs.append((pt1, pt2))
            
            # Now we have 3 pairs: (a1, a2), (b1, b2), (c1, c2)
            (a1, a2), (b1, b2), (c1, c2) = actual_pairs
            
            # The 8 transversal triples are:
            # (a1, b1, c1), (a1, b1, c2), (a1, b2, c1), (a1, b2, c2)
            # (a2, b1, c1), (a2, b1, c2), (a2, b2, c1), (a2, b2, c2)
            
            transversals = [
                (a1, b1, c1),
                (a1, b1, c2),
                (a1, b2, c1),
                (a1, b2, c2),
                (a2, b1, c1),
                (a2, b1, c2),
                (a2, b2, c1),
                (a2, b2, c2)
            ]
            
            # Convert to sorted tuples to match triple_index
            t_indices = []
            valid = True
            for t in transversals:
                st = tuple(sorted(t))
                if st in triple_index:
                    t_indices.append(triple_index[st])
                else:
                    valid = False
                    break
            
            if valid:
                # Create bitset
                mask = 0
                for idx in t_indices:
                    mask |= (1 << idx)
                
                constraints.append(mask)
                c_idx = len(constraints) - 1
                for idx in t_indices:
                    triple_to_constraints[idx].append(c_idx)
    
    # Greedy Initialization
    # Sort triples by degree (number of constraints they are in) or just random?
    # Lower degree might be better to add first? Or higher?
    # Usually in Max Independent Set on hypergraphs, adding vertices with fewer conflicts is good.
    # But here we want to MAXIMIZE size, so we want to avoid completing constraints.
    
    # Let's try a simple greedy:
    # 1. Random shuffle
    # 2. Add if valid
    
    def is_valid_add(t_idx, selected_mask):
        # Check if adding t_idx completes any constraint
        for c_idx in triple_to_constraints[t_idx]:
            c_mask = constraints[c_idx]
            # If all other bits in c_mask are set in selected_mask, then adding t_idx completes it
            # c_mask has t_idx set.
            # We want to check if (c_mask & ~selected_mask) == (1 << t_idx)
            # i.e., the only bit in c_mask not in selected_mask is t_idx itself.
            if (c_mask & ~selected_mask) == (1 << t_idx):
                return False
        return True

    # Initial Greedy
    best_mask = 0
    best_size = 0
    
    # Try a few random greedy orders
    for i in range(20):
        if time.time() > deadline - 30: break
        random.seed(i)
        order = list(range(num_triples))
        random.shuffle(order)
        
        curr_mask = 0
        curr_size = 0
        for t_idx in order:
            if is_valid_add(t_idx, curr_mask):
                curr_mask |= (1 << t_idx)
                curr_size += 1
        
        if curr_size > best_size:
            best_size = curr_size
            best_mask = curr_mask
    
    # Simulated Annealing
    # State: current_mask
    # Move: Add a random triple, Remove a random triple, Swap
    
    curr_mask = best_mask
    curr_size = best_size
    best_mask_sa = best_mask
    best_size_sa = best_size
    
    T = 10.0
    T_min = 0.1
    cooling = 0.99
    
    # Precompute list of triples not in mask? 
    # We can just pick random triple indices and check if they are in mask.
    
    steps = 0
    while time.time() < deadline - 5 and T > T_min:
        steps += 1
        
        # Choose a move
        move_type = random.random()
        if move_type < 0.3:
            # Add
            t_idx = random.randint(0, num_triples - 1)
            if curr_mask & (1 << t_idx):
                continue
            new_mask = curr_mask | (1 << t_idx)
            # Check validity
            valid = is_valid_add(t_idx, curr_mask)
            if valid:
                delta = 1
                accept = True
            else:
                # If invalid, we don't usually accept in strict IS, but in SA for Max Weight IS with penalties?
                # Here we want STRICT validity. 
                # So we only propose valid moves? 
                # If we only propose valid moves, we are doing local search, not SA.
                # To use SA, we need to allow invalid states and penalize them, OR
                # we define the objective as size - penalty * violations.
                # Given the strict requirement "no K(2,2,2)", we must return a valid set.
                # So we should stick to valid moves or use a penalty function and project back.
                # Easier: Only propose moves that keep the set valid.
                # This is effectively a random walk on the space of valid sets.
                # This is a local search (Random Walk / Metropolis on valid states).
                continue
        
        elif move_type < 0.6:
            # Remove
            # Pick a random triple from current set
            # To pick randomly from set bits, we can just pick random indices and check
            # Or maintain a list. For 560 items, picking random index and checking is fine.
            t_idx = random.randint(0, num_triples - 1)
            if not (curr_mask & (1 << t_idx)):
                continue
            new_mask = curr_mask & ~(1 << t_idx)
            # Removing is always valid
            delta = -1
            # We accept removals if they lead to a better state later? 
            # In pure maximization, we don't want to remove unless it helps add more.
            # Standard SA: accept if delta > 0 or with prob exp(delta/T).
            # Here delta is change in size.
            # If we remove, size decreases. We only accept if we can add more later.
            # So we should accept removals with some probability to escape local optima.
            accept = random.random() < (1.0 / T) # Simplified acceptance for removal
            # Better: accept with prob exp(-1/T)
            accept = random.random() < (2.71828 ** (-1.0 / T))
            
        else:
            # Swap
            # Pick one in, one out
            t_out = random.randint(0, num_triples - 1)
            if not (curr_mask & (1 << t_out)):
                continue
            t_in = random.randint(0, num_triples - 1)
            if curr_mask & (1 << t_in):
                continue
            
            # Check if adding t_in is valid after removing t_out?
            # Actually, we remove t_out, then check if t_in can be added.
            temp_mask = curr_mask & ~(1 << t_out)
            if is_valid_add(t_in, temp_mask):
                new_mask = temp_mask | (1 << t_in)
                delta = 0 # Size unchanged, but structure changed
                # We might want to accept swaps to explore.
                # In terms of size, delta is 0.
                # We accept swaps with probability 1.0 (or high) to explore.
                accept = True
            else:
                continue
        
        if accept:
            curr_mask = new_mask
            curr_size += delta
            
            if curr_size > best_size_sa:
                best_size_sa = curr_size
                best_mask_sa = curr_mask
        
        T *= cooling
        
        # If we get stuck (no valid moves found in many iterations), we might need to cool faster or restart
        # But the random selection should eventually find a move.
        
    # Final check: Ensure best_mask_sa is valid (it should be, as we only added valid triples)
    # However, if we used "accept removal" logic, we must ensure we don't end up with an invalid set.
    # Since we only add if valid, and removals don't create violations, the set is always valid.
    
    # Convert best_mask_sa to list of triples
    result = []
    for i in range(num_triples):
        if best_mask_sa & (1 << i):
            result.append(list(all_triples[i]))
            
    return result
# EVOLVE-BLOCK-END
