"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _reachable_sums(selected):
    """Compute all subset sums <= 200 for a given set."""
    reachable = {0}
    for x in selected:
        new = {s + x for s in reachable if s + x <= 200}
        reachable |= new
    return reachable

def _is_200_free(selected):
    """Check if no subset of selected sums to 200."""
    reachable = {0}
    for x in selected:
        new = {s + x for s in reachable if s + x <= 200}
        if 200 in new:
            return False
        reachable |= new
    return True

def _greedy_construct(order):
    """Greedy: add elements in given order if 200-free."""
    selected = []
    reachable = {0}
    for x in order:
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    return selected

def _local_search(selected, all_elements, deadline):
    """Try to improve by removing one and adding one or more others."""
    selected_set = set(selected)
    best = list(selected)
    best_size = len(selected)
    
    reachable = _reachable_sums(selected)
    
    # Try single swaps: remove one, add one
    for x in list(selected):
        if time.time() > deadline:
            break
        remaining = [y for y in selected if y != x]
        rem_reachable = _reachable_sums(remaining)
        
        for y in all_elements:
            if y in selected_set:
                continue
            new = {s + y for s in rem_reachable if s + y <= 200}
            if 200 not in new:
                # This swap is valid; size stays same, but might enable further improvements
                # Only keep if it helps (it doesn't change size, so skip unless we track quality)
                pass
    
    # Try remove-one, add-two: remove x, add y and z
    # This can increase size by 1
    for x in list(selected):
        if time.time() > deadline:
            break
        remaining = [y for y in selected if y != x]
        rem_reachable = _reachable_sums(remaining)
        
        for y in all_elements:
            if y in selected_set:
                continue
            new1 = {s + y for s in rem_reachable if s + y <= 200}
            if 200 in new1:
                continue
            for z in all_elements:
                if z in selected_set or z == y:
                    continue
                new2 = {s + z for s in new1 if s + z <= 200}
                if 200 not in new2:
                    candidate = remaining + [y, z]
                    if len(candidate) > best_size:
                        best = candidate
                        best_size = len(candidate)
                        selected = candidate
                        selected_set = set(selected)
                        reachable = _reachable_sums(selected)
                        break
    
    return best

def _try_multiple_greedy(deadline):
    """Try multiple greedy orders and pick the best."""
    all_elements = list(range(1, 101))
    
    orders = [
        list(range(100, 0, -1)),  # descending
        list(range(1, 101)),      # ascending
        list(range(50, 0, -1)) + list(range(100, 50, -1)),  # mid-high, low-high
        list(range(67, 101)) + list(range(66, 0, -1)),  # safe block first
        list(range(100, 66, -1)) + [66] + list(range(65, 0, -1)),
    ]
    
    best = []
    best_size = 0
    for order in orders:
        if time.time() > deadline:
            break
        result = _greedy_construct(order)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Start with the known-good set {66, ..., 100} = 35 elements
    base_set = list(range(66, 101))
    assert _is_200_free(base_set), "Base set should be 200-free"
    
    best = base_set
    best_size = len(best)
    
    # Try multiple greedy constructions
    greedy_result = _try_multiple_greedy(deadline)
    if len(greedy_result) > best_size:
        best = greedy_result
        best_size = len(best)
    
    # Local search: try to increase size by removing one and adding two
    all_elements = set(range(1, 101))
    
    # More systematic local search
    selected = list(best)
    selected_set = set(selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(selected):
            if time.time() > deadline:
                break
            remaining = [y for y in selected if y != x]
            rem_reachable = _reachable_sums(remaining)
            
            # Find all pairs (y, z) not in selected such that remaining + {y, z} is 200-free
            candidates = []
            for y in range(1, 101):
                if y in selected_set:
                    continue
                new1 = {s + y for s in rem_reachable if s + y <= 200}
                if 200 in new1:
                    continue
                for z in range(y + 1, 101):
                    if z in selected_set:
                        continue
                    new2 = {s + z for s in new1 if s + z <= 200}
                    if 200 not in new2:
                        candidates.append((y, z))
            
            if candidates:
                # Pick the pair that maximizes future potential (largest elements)
                candidates.sort(key=lambda p: -(p[0] + p[1]))
                y, z = candidates[0]
                new_selected = remaining + [y, z]
                if _is_200_free(new_selected):
                    selected = new_selected
                    selected_set = set(selected)
                    if len(selected) > best_size:
                        best = selected
                        best_size = len(best)
                    improved = True
                    break
    
    return best
# EVOLVE-BLOCK-END
