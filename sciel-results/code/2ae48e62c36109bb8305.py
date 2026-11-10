# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem modulo 205.
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    
    Strategy:
    1. Compute forbidden differences F = {x^2 mod 205 : x in 1..204} \\ {0}.
    2. Use a hybrid approach:
       a. Deterministic greedy initialization with multiple orderings.
       b. Simulated annealing / local search to improve the set size.
       c. Multi-start with random perturbations to escape local optima.
    3. Include a deadline guard to ensure we finish well before the 600s limit.
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
    
    # Precompute conflict matrix for speed
    # conflict[i][j] = True if (i-j) % m is in forbidden_set
    # This is a 205x205 boolean matrix
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            d = (i - j) % m
            if d in forbidden_set:
                conflict[i][j] = True
    
    # Note: conflict[i][j] is True if i and j cannot coexist
    # We need to check if adding cand to set S is valid:
    # for all s in S, not conflict[cand][s]
    
    def can_add(cand, sel_set):
        """Check if cand can be added to sel_set."""
        for s in sel_set:
            if conflict[cand][s]:
                return False
        return True
    
    def count_conflicts(elem, sel_set):
        """Count how many elements in sel_set conflict with elem."""
        cnt = 0
        for s in sel_set:
            if conflict[elem][s]:
                cnt += 1
        return cnt
    
    # Greedy initialization: try multiple orderings and take the best
    def greedy_init(order):
        selected = []
        selected_set = set()
        for cand in order:
            ok = True
            for s in selected:
                if conflict[cand][s]:
                    ok = False
                    break
            if ok:
                selected.append(cand)
                selected_set.add(cand)
        return selected, selected_set
    
    # Try several deterministic orderings
    best_init = None
    best_init_size = -1
    
    orders = [
        list(range(m)),
        list(range(m-1, -1, -1)),
        # Random-ish but deterministic orderings
        list(range(0, m, 3)),
        list(range(1, m, 3)),
        list(range(2, m, 3)),
        list(range(0, m, 7)),
        list(range(1, m, 7)),
        list(range(2, m, 7)),
        list(range(0, m, 13)),
        list(range(1, m, 13)),
        list(range(2, m, 13)),
        list(range(0, m, 17)),
        list(range(1, m, 17)),
        list(range(2, m, 17)),
        list(range(0, m, 19)),
        list(range(1, m, 19)),
        list(range(2, m, 19)),
    ]
    
    # Also try interleaved orders
    rng = random.Random(42)
    for seed in range(30):
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
    current_set_size = best_init_size
    
    # Phase 1: Greedy fill - repeatedly try to add the first available element
    improved = True
    while improved:
        improved = False
        for cand in range(m):
            if cand not in current_set_set:
                if can_add(cand, current_set_set):
                    current_set.append(cand)
                    current_set_set.add(cand)
                    current_set_size += 1
                    improved = True
                    break
    
    # Track best solution found
    best_set = list(current_set)
    best_set_set = set(current_set_set)
    best_size = current_set_size
    
    # Phase 2: Simulated annealing-like local search
    rng2 = random.Random(12345)
    
    def make_move(current_set, current_set_set, current_set_size):
        """Generate a random move and apply it. Returns (new_set, new_set_set, new_size, improved)."""
        move_type = rng2.random()
        
        if move_type < 0.4:
            # Try to add a random element not in set
            candidates_to_add = [c for c in range(m) if c not in current_set_set]
            if not candidates_to_add:
                return current_set, current_set_set, current_set_size, False
            cand = rng2.choice(candidates_to_add)
            if can_add(cand, current_set_set):
                new_set = current_set + [cand]
                new_set_set = current_set_set | {cand}
                return new_set, new_set_set, current_set_size + 1, True
        
        elif move_type < 0.8:
            # Swap: remove one from set, add one from outside
            if not current_set:
                return current_set, current_set_set, current_set_size, False
            to_remove = rng2.choice(current_set)
            candidates_to_add = [c for c in range(m) if c not in current_set_set]
            if not candidates_to_add:
                return current_set, current_set_set, current_set_size, False
            to_add = rng2.choice(candidates_to_add)
            
            # Check if to_add conflicts with any element in set except to_remove
            new_set_set = current_set_set - {to_remove}
            if can_add(to_add, new_set_set):
                new_set = [x for x in current_set if x != to_remove] + [to_add]
                new_set_set = new_set_set | {to_add}
                return new_set, new_set_set, current_set_size, False
            else:
                return current_set, current_set_set, current_set_size, False
        
        else:
            # Remove one (to break out of local optima)
            if not current_set:
                return current_set, current_set_set, current_set_size, False
            to_remove = rng2.choice(current_set)
            new_set = [x for x in current_set if x != to_remove]
            new_set_set = current_set_set - {to_remove}
            return new_set, new_set_set, current_set_size - 1, False
    
    # Run local search with multiple restarts
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            # Start from current best
            cur_set = list(best_set)
            cur_set_set = set(best_set_set)
            cur_size = best_size
        else:
            # Start from a perturbed version of best
            cur_set = list(best_set)
            cur_set_set = set(best_set_set)
            cur_size = best_size
            
            # Perturb: remove 2-5 random elements
            num_remove = rng2.randint(2, min(5, cur_size))
            for _ in range(num_remove):
                if cur_set:
                    to_remove = rng2.choice(cur_set)
                    cur_set.remove(to_remove)
                    cur_set_set.discard(to_remove)
                    cur_size -= 1
        
        # Run local search for this restart
        local_best_set = list(cur_set)
        local_best_set_set = set(cur_set_set)
        local_best_size = cur_size
        
        # Simple hill climbing with occasional random moves
        for step in range(5000):
            if time.time() > deadline - 10:
                break
            
            new_set, new_set_set, new_size, _ = make_move(cur_set, cur_set_set, cur_size)
            
            # Accept if better or equal size (with some randomness for exploration)
            if new_size > cur_size:
                cur_set = new_set
                cur_set_set = new_set_set
                cur_size = new_size
                if cur_size > local_best_size:
                    local_best_size = cur_size
                    local_best_set = list(cur_set)
                    local_best_set_set = set(cur_set_set)
            elif new_size == cur_size and rng2.random() < 0.1:
                # Occasionally accept equal-size moves to explore
                cur_set = new_set
                cur_set_set = new_set_set
            elif new_size < cur_size and rng2.random() < 0.05:
                # Rarely accept worse moves to escape local optima
                cur_set = new_set
                cur_set_set = new_set_set
                cur_size = new_size
        
        # Update global best
        if local_best_size > best_size:
            best_size = local_best_size
            best_set = list(local_best_set)
            best_set_set = set(local_best_set_set)
    
    # Final greedy fill on the best set found
    final_set = list(best_set)
    final_set_set = set(best_set_set)
    improved = True
    while improved and time.time() < deadline - 2:
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
                final_set_set = {0}
                break
    
    return np.array(final_set, dtype=np.int64)

# EVOLVE-BLOCK-END
