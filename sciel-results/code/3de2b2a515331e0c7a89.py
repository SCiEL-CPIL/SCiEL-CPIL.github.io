# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    
    Strategy:
    1. Compute forbidden differences F = {x^2 mod 205 : x in 1..204} \ {0}.
    2. Use a combination of greedy initialization with multiple orderings,
       followed by simulated annealing with add/remove/swap moves,
       and final greedy fill to maximize set size.
    3. Include a deadline guard to ensure we finish well before the 600s limit.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 570  # Safety margin of 30s
    
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = (x * x) % m
        if val != 0:
            forbidden.add(val)
    
    forbidden_set = forbidden  # for fast lookup
    
    # Precompute conflict matrix: conflict[i][j] = True if (i-j) mod m or (j-i) mod m is forbidden
    # This is a 205x205 boolean matrix for fast lookup
    conflict = [[False] * m for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden_set or d2 in forbidden_set:
                conflict[i][j] = True
                conflict[j][i] = True
    
    def can_add(cand, sel_set):
        """Check if cand can be added to sel_set without conflicts."""
        for s in sel_set:
            if conflict[cand][s]:
                return False
        return True
    
    # Greedy initialization: try multiple orderings and take the best
    def greedy_init(order):
        selected = []
        selected_set = set()
        for cand in order:
            if can_add(cand, selected_set):
                selected.append(cand)
                selected_set.add(cand)
        return selected, selected_set
    
    # Try several deterministic and random orderings
    best_init = None
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
    for seed in range(50):
        rng = random.Random(seed)
        order = list(range(m))
        rng.shuffle(order)
        orders.append(order)
    
    for order in orders:
        selected, selected_set = greedy_init(order)
        if len(selected) > best_init_size:
            best_init_size = len(selected)
            best_init = (list(selected), set(selected))
    
    if best_init is None:
        best_init = ([0], {0})
        best_init_size = 1
    
    # Start with the best greedy solution
    current_set = list(best_init[0])
    current_set_set = set(best_init[1])
    current_size = len(current_set)
    
    # Track best found so far
    best_set = list(current_set)
    best_set_set = set(current_set_set)
    best_size = current_size
    
    # Phase 1: Greedy fill - repeatedly try to add the first available element
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for cand in range(m):
            if cand not in current_set_set:
                if can_add(cand, current_set_set):
                    current_set.append(cand)
                    current_set_set.add(cand)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_set = list(current_set)
                        best_set_set = set(current_set_set)
                    break
    
    # Phase 2: Simulated annealing
    rng = random.Random(98765)
    temp = 2.0
    cooling = 0.9995
    min_temp = 0.01
    
    iteration = 0
    while time.time() < deadline - 15 and temp > min_temp:
        iteration += 1
        
        if iteration % 50000 == 0:
            temp = max(temp * cooling, min_temp)
        
        # Generate a move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try to add a random element not in set
            candidates_to_add = [c for c in range(m) if c not in current_set_set]
            if not candidates_to_add:
                continue
            cand = rng.choice(candidates_to_add)
            if can_add(cand, current_set_set):
                current_set.append(cand)
                current_set_set.add(cand)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = list(current_set)
                    best_set_set = set(current_set_set)
        
        elif move_type < 0.75:
            # Swap: remove one from set, add one from outside
            if not current_set:
                continue
            to_remove = rng.choice(current_set)
            candidates_to_add = [c for c in range(m) if c not in current_set_set]
            if not candidates_to_add:
                continue
            to_add = rng.choice(candidates_to_add)
            
            # Check if to_add conflicts with any element in current_set_set \ {to_remove}
            new_set_set = current_set_set - {to_remove}
            can_swap = True
            for s in new_set_set:
                if conflict[to_add][s]:
                    can_swap = False
                    break
            
            if can_swap:
                current_set.remove(to_remove)
                current_set.append(to_add)
                current_set_set.discard(to_remove)
                current_set_set.add(to_add)
                # Size stays the same
                if current_size > best_size:
                    best_size = current_size
                    best_set = list(current_set)
                    best_set_set = set(current_set_set)
        
        else:
            # Remove one (to break out of local optima)
            if not current_set:
                continue
            to_remove = rng.choice(current_set)
            current_set.remove(to_remove)
            current_set_set.discard(to_remove)
            current_size -= 1
        
        # Cool down temperature
        temp *= cooling
    
    # Phase 3: Final greedy fill on the best set found
    final_set = list(best_set)
    final_set_set = set(best_set_set)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for cand in range(m):
            if cand not in final_set_set:
                if can_add(cand, final_set_set):
                    final_set.append(cand)
                    final_set_set.add(cand)
                    improved = True
                    break
    
    # Verify the final set
    for i in range(len(final_set)):
        for j in range(i+1, len(final_set)):
            a, b = final_set[i], final_set[j]
            if conflict[a][b]:
                # Should not happen, but if it does, rebuild
                final_set = [0]
                break
    
    return np.array(final_set, dtype=np.int64)

# EVOLVE-BLOCK-END
