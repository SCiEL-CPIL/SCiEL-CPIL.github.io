# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses a combination of greedy construction, randomized restarts, and local search.
    """
    m = 205
    k = 2
    deadline = time.time() + 580.0  # 580s safety margin before 630s kill

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = pow(x, k, m)
        if val != 0:
            forbidden.add(val)
    
    # For fast lookup, also create a boolean array
    forbidden_arr = np.zeros(m, dtype=bool)
    for f in forbidden:
        forbidden_arr[f] = True

    def is_valid_addition(candidate, selected_set):
        """Check if candidate can be added to selected_set without creating a forbidden pair."""
        for s in selected_set:
            d1 = (candidate - s) % m
            d2 = (s - candidate) % m
            if forbidden_arr[d1] or forbidden_arr[d2]:
                return False
        return True

    def greedy_construct(order):
        """Greedy construction using a specific order of candidates."""
        selected = []
        selected_set = set()
        for candidate in order:
            if is_valid_addition(candidate, selected_set):
                selected.append(candidate)
                selected_set.add(candidate)
        return selected

    def local_search(initial_set, max_iterations=1000):
        """Local search: try to replace elements to allow adding new ones."""
        selected = list(initial_set)
        selected_set = set(selected)
        best_size = len(selected)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            # Try adding any element
            for candidate in range(m):
                if candidate not in selected_set:
                    if is_valid_addition(candidate, selected_set):
                        selected.append(candidate)
                        selected_set.add(candidate)
                        best_size = len(selected)
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swapping: remove one, add another
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                removed = selected[i]
                new_selected = selected[:i] + selected[i+1:]
                new_selected_set = set(new_selected)
                
                # Try to add a better candidate
                for candidate in range(m):
                    if candidate not in new_selected_set and candidate != removed:
                        if is_valid_addition(candidate, new_selected_set):
                            # Check if this swap actually helps (allows more additions later)
                            # For now, just accept any swap that keeps size same or increases
                            # But we need to be careful: swapping one for one doesn't increase size
                            # So we only do swaps if they enable future additions
                            # Simple heuristic: accept swap if it allows at least one more addition
                            new_selected.append(candidate)
                            new_selected_set.add(candidate)
                            # Check if we can add more
                            can_add_more = False
                            for c in range(m):
                                if c not in new_selected_set:
                                    if is_valid_addition(c, new_selected_set):
                                        can_add_more = True
                                        break
                            if can_add_more:
                                selected = new_selected
                                selected_set = new_selected_set
                                best_size = len(selected)
                                improved = True
                                break
                if improved:
                    break
            if not improved:
                break
        
        return selected

    def randomized_greedy(n_restarts=100):
        """Try multiple random orderings."""
        best = []
        for _ in range(n_restarts):
            if time.time() > deadline:
                break
            order = list(range(m))
            random.shuffle(order)
            result = greedy_construct(order)
            if len(result) > len(best):
                best = result
        return best

    # Phase 1: Deterministic greedy with natural order
    best_result = greedy_construct(list(range(m)))
    best_size = len(best_result)
    
    # Phase 2: Randomized greedy
    random.seed(42)
    result = randomized_greedy(n_restarts=200)
    if len(result) > best_size:
        best_result = result
        best_size = len(result)
    
    # Phase 3: Local search from best result
    if time.time() < deadline:
        result = local_search(best_result, max_iterations=500)
        if len(result) > best_size:
            best_result = result
            best_size = len(result)
    
    # Phase 4: More randomized restarts with local search
    random.seed(123)
    for restart in range(50):
        if time.time() > deadline - 30:
            break
        order = list(range(m))
        random.shuffle(order)
        result = greedy_construct(order)
        if len(result) >= best_size:
            result = local_search(result, max_iterations=200)
            if len(result) > best_size:
                best_result = result
                best_size = len(result)
    
    # Final validation
    final_set = set(best_result)
    for i, a in enumerate(best_result):
        for b in best_result[i+1:]:
            d1 = (a - b) % m
            d2 = (b - a) % m
            if forbidden_arr[d1] or forbidden_arr[d2]:
                # Should not happen, but if it does, return empty to be safe
                return np.array([], dtype=np.int64)
    
    return np.array(best_result, dtype=np.int64)


# EVOLVE-BLOCK-END
