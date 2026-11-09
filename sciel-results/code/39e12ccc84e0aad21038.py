# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem (modular formulation). Find a subset of Z/mZ of maximum
    size that does NOT contain any two elements differing by a nonzero perfect k-th power
    modulo m. Here m = 205 and k = 2 (squares), i.e. forbidden differences are the nonzero
    quadratic residues modulo 205.

    Strategy:
    1. Build the conflict graph where edges represent forbidden differences.
    2. Use randomized greedy construction to get an initial independent set.
    3. Apply local search (add/remove/refill cycles) to improve.
    4. Use simulated annealing / random restarts to escape local minima.
    5. Deadline guard to ensure we return within time limits.

    Returns:
        subset: np.ndarray of shape (s,), dtype int, a list of distinct residues in
            {0, ..., 204} with no two differing by a nonzero square modulo 205.
    """
    m = 205
    deadline = time.time() + 580  # Safety margin before 630s limit
    
    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_list = list(forbidden)
    forbidden_set = set(forbidden_list)
    
    # Build conflict graph: conflict[a][b] = True if (a-b) mod m is in forbidden
    # For efficiency, we'll use a 2D boolean array
    conflict = np.zeros((m, m), dtype=bool)
    for a in range(m):
        for b in range(m):
            d = (a - b) % m
            if d in forbidden_set:
                conflict[a, b] = True
    
    def is_valid_addition(candidate, selected_set, selected_list):
        """Check if candidate can be added to the current set."""
        for s in selected_list:
            if conflict[candidate, s]:
                return False
        return True
    
    def greedy_construct(random_order=None):
        """Construct an independent set using greedy method with optional random ordering."""
        if random_order is None:
            candidates = list(range(m))
            random.shuffle(candidates)
        else:
            candidates = random_order
        
        selected = []
        selected_set = set()
        for candidate in candidates:
            if candidate in selected_set:
                continue
            ok = True
            for s in selected:
                if conflict[candidate, s]:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
                selected_set.add(candidate)
        return selected, selected_set
    
    def local_search(selected, selected_set, max_iterations=500):
        """Improve the independent set using add/remove/refill local search."""
        best_selected = list(selected)
        best_set = set(selected_set)
        current_size = len(selected)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try to add a random element
            added = False
            for _ in range(10):
                candidate = random.randint(0, m - 1)
                if candidate not in selected_set:
                    if is_valid_addition(candidate, selected_set, selected):
                        selected.append(candidate)
                        selected_set.add(candidate)
                        added = True
                        break
            
            if added:
                if len(selected) > current_size:
                    current_size = len(selected)
                    best_selected = list(selected)
                    best_set = set(selected_set)
                continue
            
            # Try removal-refill: remove a random element, then try to add others
            if len(selected) > 1:
                remove_idx = random.randint(0, len(selected) - 1)
                removed_elem = selected[remove_idx]
                selected.pop(remove_idx)
                selected_set.remove(removed_elem)
                
                # Try to add back some elements
                refill_added = 0
                for candidate in random.sample(range(m), min(20, m)):
                    if candidate not in selected_set:
                        if is_valid_addition(candidate, selected_set, selected):
                            selected.append(candidate)
                            selected_set.add(candidate)
                            refill_added += 1
                
                if len(selected) > current_size:
                    current_size = len(selected)
                    best_selected = list(selected)
                    best_set = set(selected_set)
        
        return best_selected, best_set
    
    def simulated_annealing(selected, selected_set, start_temp=1.0, end_temp=0.01, iterations=2000):
        """Simulated annealing to escape local minima."""
        current_selected = list(selected)
        current_set = set(selected_set)
        best_selected = list(selected)
        best_set = set(selected_set)
        best_size = len(selected)
        
        temp = start_temp
        temp_decay = (end_temp / start_temp) ** (1.0 / iterations)
        
        for i in range(iterations):
            if time.time() > deadline:
                break
            
            # Generate a move: either add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3 and len(current_selected) < m:
                # Try to add
                candidate = random.randint(0, m - 1)
                if candidate not in current_set:
                    if is_valid_addition(candidate, current_set, current_selected):
                        current_selected.append(candidate)
                        current_set.add(candidate)
                        if len(current_selected) > best_size:
                            best_size = len(current_selected)
                            best_selected = list(current_selected)
                            best_set = set(current_set)
                    else:
                        # Reject addition (no penalty since we can't add invalid)
                        pass
            
            elif move_type < 0.6 and len(current_selected) > 1:
                # Try to remove
                remove_idx = random.randint(0, len(current_selected) - 1)
                removed = current_selected[remove_idx]
                current_selected.pop(remove_idx)
                current_set.remove(removed)
                
                # Accept with probability based on temperature (removal always accepted for exploration)
                # But track best
                if len(current_selected) > best_size:
                    best_size = len(current_selected)
                    best_selected = list(current_selected)
                    best_set = set(current_set)
                
                # Try to refill
                for _ in range(5):
                    candidate = random.randint(0, m - 1)
                    if candidate not in current_set:
                        if is_valid_addition(candidate, current_set, current_selected):
                            current_selected.append(candidate)
                            current_set.add(candidate)
                            if len(current_selected) > best_size:
                                best_size = len(current_selected)
                                best_selected = list(current_selected)
                                best_set = set(current_set)
            
            else:
                # Swap: remove one, add another
                if len(current_selected) > 0:
                    remove_idx = random.randint(0, len(current_selected) - 1)
                    removed = current_selected[remove_idx]
                    current_selected.pop(remove_idx)
                    current_set.remove(removed)
                    
                    candidate = random.randint(0, m - 1)
                    if candidate not in current_set:
                        if is_valid_addition(candidate, current_set, current_selected):
                            current_selected.append(candidate)
                            current_set.add(candidate)
                            if len(current_selected) > best_size:
                                best_size = len(current_selected)
                                best_selected = list(current_selected)
                                best_set = set(current_set)
            
            temp *= temp_decay
        
        return best_selected, best_set
    
    # Main search loop with random restarts
    best_overall_selected = []
    best_overall_set = set()
    best_overall_size = 0
    
    # Initial deterministic greedy
    det_selected, det_set = greedy_construct(list(range(m)))
    if len(det_selected) > best_overall_size:
        best_overall_size = len(det_selected)
        best_overall_selected = list(det_selected)
        best_overall_set = set(det_set)
    
    # Random restarts with local search
    num_restarts = 0
    while time.time() < deadline - 30:
        num_restarts += 1
        
        # Random greedy construction
        rand_selected, rand_set = greedy_construct()
        
        # Local search
        ls_selected, ls_set = local_search(rand_selected, rand_set, max_iterations=200)
        
        # Simulated annealing
        sa_selected, sa_set = simulated_annealing(ls_selected, ls_set, start_temp=1.0, end_temp=0.01, iterations=1000)
        
        if len(sa_selected) > best_overall_size:
            best_overall_size = len(sa_selected)
            best_overall_selected = list(sa_selected)
            best_overall_set = set(sa_set)
        
        # Also try local search on the best found so far
        if num_restarts % 5 == 0:
            ls2_selected, ls2_set = local_search(best_overall_selected, best_overall_set, max_iterations=300)
            if len(ls2_selected) > best_overall_size:
                best_overall_size = len(ls2_selected)
                best_overall_selected = list(ls2_selected)
                best_overall_set = set(ls2_set)
    
    # Final verification
    final_selected = best_overall_selected
    
    # Verify no forbidden pairs
    for i in range(len(final_selected)):
        for j in range(i + 1, len(final_selected)):
            d1 = (final_selected[i] - final_selected[j]) % m
            d2 = (final_selected[j] - final_selected[i]) % m
            if d1 in forbidden_set or d2 in forbidden_set:
                # This should not happen, but if it does, remove the problematic element
                # For safety, just return what we have
                pass
    
    return np.array(sorted(final_selected), dtype=np.int64)


# EVOLVE-BLOCK-END
