# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square.
    
    Strategy:
    1. Compute forbidden differences F = {x^2 mod 205 : x in 1..204} \ {0}.
    2. Build conflict graph using bitmasks for O(1) conflict checks.
    3. Use greedy initialization with multiple orderings (deterministic + random).
    4. Apply simulated annealing with Add/Swap/Remove moves.
    5. Final greedy fill to maximize set size.
    6. Deadline guard to ensure completion before 600s limit.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 580  # Safety margin of 20s
    
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = (x * x) % m
        if val != 0:
            forbidden.add(val)
    
    forbidden_set = forbidden  # for fast lookup
    
    # Precompute conflict masks: conflict_mask[i] is a bitmask of all j that conflict with i
    # Two elements i, j conflict if (i-j) mod m or (j-i) mod m is in forbidden
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden_set:
            j1 = (i - d) % m
            j2 = (i + d) % m
            if j1 != i:
                mask |= (1 << j1)
            if j2 != i:
                mask |= (1 << j2)
        conflict_mask[i] = mask
    
    # Helper: check if candidate can be added to a set represented by a bitmask
    def can_add_mask(cand, sel_mask):
        """Check if cand can be added to the set represented by sel_mask."""
        return (sel_mask & conflict_mask[cand]) == 0
    
    # Greedy initialization: try multiple orderings and take the best
    def greedy_init(order):
        selected_mask = 0
        selected_list = []
        for cand in order:
            if can_add_mask(cand, selected_mask):
                selected_mask |= (1 << cand)
                selected_list.append(cand)
        return selected_list, selected_mask
    
    # Try several deterministic and random orderings
    best_init_list = None
    best_init_mask = 0
    best_init_size = -1
    
    orders = [
        list(range(m)),
        list(range(m-1, -1, -1)),
        list(range(0, m, 2)),
        list(range(1, m, 2)),
        list(range(0, m, 3)),
        list(range(1, m, 3)),
        list(range(2, m, 3)),
        list(range(0, m, 5)),
        list(range(1, m, 5)),
        list(range(2, m, 5)),
        list(range(3, m, 5)),
        list(range(4, m, 5)),
        list(range(0, m, 7)),
        list(range(1, m, 7)),
        list(range(2, m, 7)),
        list(range(0, m, 11)),
        list(range(1, m, 11)),
        list(range(0, m, 13)),
        list(range(1, m, 13)),
        list(range(0, m, 17)),
        list(range(1, m, 17)),
        list(range(0, m, 19)),
        list(range(1, m, 19)),
    ]
    
    # Add random orderings
    for seed in range(100):
        rng = random.Random(seed)
        order = list(range(m))
        rng.shuffle(order)
        orders.append(order)
    
    for order in orders:
        if time.time() > deadline - 30:
            break
        selected_list, selected_mask = greedy_init(order)
        if len(selected_list) > best_init_size:
            best_init_size = len(selected_list)
            best_init_list = selected_list
            best_init_mask = selected_mask
    
    if best_init_list is None:
        best_init_list = [0]
        best_init_mask = 1
        best_init_size = 1
    
    # Start with the best greedy solution
    current_list = list(best_init_list)
    current_mask = best_init_mask
    current_size = best_init_size
    
    # Track best found so far
    best_list = list(current_list)
    best_mask = current_mask
    best_size = current_size
    
    # Phase 1: Greedy fill - repeatedly try to add any available element
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        for cand in range(m):
            if time.time() > deadline - 20:
                break
            if (current_mask & (1 << cand)) == 0:
                if can_add_mask(cand, current_mask):
                    current_mask |= (1 << cand)
                    current_list.append(cand)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_list = list(current_list)
                        best_mask = current_mask
                    break
    
    # Phase 2: Simulated annealing
    rng = random.Random(98765)
    temp = 3.0
    cooling = 0.9998
    min_temp = 0.005
    
    # Precompute list of all elements for faster random selection
    all_elements = list(range(m))
    
    iteration = 0
    while time.time() < deadline - 10 and temp > min_temp:
        iteration += 1
        
        # Generate a move
        move_type = rng.random()
        
        if move_type < 0.35:
            # Try to add a random element not in set
            # Find candidates to add efficiently
            added = False
            for _ in range(20):  # Try up to 20 random candidates
                cand = rng.choice(all_elements)
                if (current_mask & (1 << cand)) == 0:
                    if can_add_mask(cand, current_mask):
                        current_mask |= (1 << cand)
                        current_list.append(cand)
                        current_size += 1
                        added = True
                        if current_size > best_size:
                            best_size = current_size
                            best_list = list(current_list)
                            best_mask = current_mask
                        break
        
        elif move_type < 0.75:
            # Swap: remove one from set, add one from outside
            if not current_list:
                continue
            to_remove = rng.choice(current_list)
            
            # Find a candidate to add that doesn't conflict with (current_set \ {to_remove})
            temp_mask = current_mask & ~(1 << to_remove)
            swapped = False
            for _ in range(30):  # Try up to 30 random candidates
                to_add = rng.choice(all_elements)
                if (current_mask & (1 << to_add)) == 0:
                    if can_add_mask(to_add, temp_mask):
                        # Perform swap
                        current_mask = temp_mask | (1 << to_add)
                        current_list.remove(to_remove)
                        current_list.append(to_add)
                        # Size stays the same
                        swapped = True
                        break
        
        else:
            # Remove one (to break out of local optima)
            if not current_list:
                continue
            to_remove = rng.choice(current_list)
            current_mask &= ~(1 << to_remove)
            current_list.remove(to_remove)
            current_size -= 1
        
        # Cool down temperature
        temp *= cooling
        
        # Periodic greedy fill during annealing
        if iteration % 1000 == 0:
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for cand in range(m):
                    if (current_mask & (1 << cand)) == 0:
                        if can_add_mask(cand, current_mask):
                            current_mask |= (1 << cand)
                            current_list.append(cand)
                            current_size += 1
                            improved = True
                            if current_size > best_size:
                                best_size = current_size
                                best_list = list(current_list)
                                best_mask = current_mask
                            break
    
    # Phase 3: Final greedy fill on the best set found
    final_list = list(best_list)
    final_mask = best_mask
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for cand in range(m):
            if time.time() > deadline - 5:
                break
            if (final_mask & (1 << cand)) == 0:
                if can_add_mask(cand, final_mask):
                    final_mask |= (1 << cand)
                    final_list.append(cand)
                    improved = True
                    break
    
    # Verify the final set
    for i in range(len(final_list)):
        a = final_list[i]
        for j in range(i+1, len(final_list)):
            b = final_list[j]
            if (conflict_mask[a] >> b) & 1:
                # Should not happen, but if it does, rebuild conservatively
                final_list = [0]
                break
        if len(final_list) == 1 and i > 0:
            break
    
    return np.array(final_list, dtype=np.int64)

# EVOLVE-BLOCK-END
