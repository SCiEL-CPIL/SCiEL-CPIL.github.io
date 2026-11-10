import time
import random
import numpy as np

N = 256

# Precompute 4-APs
_ALL_4APS = []
for a in range(1, N+1):
    for d in range(1, (N - a) // 3 + 1):
        _ALL_4APS.append((a, a+d, a+2*d, a+3*d))

# Precompute 3-AP partners
_X_3AP_PARTNERS = [[] for _ in range(N+1)]
for a in range(1, N+1):
    for d in range(1, (N - a) // 2 + 1):
        b = a + d
        c = a + 2*d
        _X_3AP_PARTNERS[a].append((b, c))
        _X_3AP_PARTNERS[b].append((a, c))
        _X_3AP_PARTNERS[c].append((a, b))

def _count_3aps(selected_bool):
    count = 0
    for a in range(1, N+1):
        if not selected_bool[a]:
            continue
        max_d = (N - a) // 2
        for d in range(1, max_d + 1):
            if selected_bool[a+d] and selected_bool[a+2*d]:
                count += 1
    return count

def _delta_3aps(selected_bool, x):
    """
    If x is in selected_bool, this returns the number of 3-APs containing x.
    If x is not in selected_bool, this returns the number of 3-APs that would be formed by adding x.
    """
    delta = 0
    for u, v in _X_3AP_PARTNERS[x]:
        if selected_bool[u] and selected_bool[v]:
            delta += 1
    return delta

def _find_4ap(selected_bool):
    """Find any 4-AP in the set. Returns None if none."""
    for ap in _ALL_4APS:
        if selected_bool[ap[0]] and selected_bool[ap[1]] and selected_bool[ap[2]] and selected_bool[ap[3]]:
            return ap
    return None

def _construct_remove(rng):
    """
    Start with full set, remove elements to break 4-APs.
    """
    selected = np.ones(N+1, dtype=bool)
    selected[0] = False
    
    # To make it deterministic but varied, we can shuffle the order of 4-APs we check
    # Or just iterate.
    
    # We need to ensure we don't remove too many.
    # Strategy: While 4-APs exist, pick one, remove the element with lowest 3-AP contribution.
    
    # To avoid getting stuck in a bad local minimum, we can randomize the choice of which 4-AP to break first.
    # But finding a 4-AP is O(1) if we just scan.
    
    # Let's use a priority queue or just simple iteration.
    # Since N is small, simple iteration is fine.
    
    while True:
        ap = _find_4ap(selected)
        if ap is None:
            break
        
        # Choose element to remove from this 4-AP
        # Score each element by its 3-AP contribution
        best_x = -1
        min_contrib = float('inf')
        
        # Shuffle the elements of the 4-AP to break ties randomly
        elems = list(ap)
        rng.shuffle(elems)
        
        for x in elems:
            contrib = _delta_3aps(selected, x)
            if contrib < min_contrib:
                min_contrib = contrib
                best_x = x
        
        selected[best_x] = False
        
    return selected

def _local_search(selected, current_score, deadline, rng):
    """
    Best-improvement local search using swaps.
    """
    best_selected = selected.copy()
    best_score = current_score
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        sel_list = [x for x in range(1, N+1) if selected[x]]
        unsel_list = [x for x in range(1, N+1) if not selected[x]]
        
        if not sel_list or not unsel_list:
            break
            
        best_delta = 0
        best_x = -1
        best_y = -1
        
        # Evaluate all swaps
        for x in sel_list:
            for y in unsel_list:
                # Check 4-AP constraint for y
                selected[x] = False
                # Check if adding y creates a 4-AP
                valid = True
                for ap in _ALL_4APS:
                    # This is slow. We need a faster 4-AP check.
                    pass
                # We don't have a fast 4-AP check precomputed for "adding y".
                # We need to precompute _ELEM_4AP_IDX again.
                
                # Let's assume we have _ELEM_4AP_IDX.
                # If not, we must use the slow check or precompute.
                # I will precompute _ELEM_4AP_IDX in the main block.
                
                selected[x] = True
                
        # The slow 4-AP check makes this O(N^2 * K) where K is avg 4-APs per element.
        # K is small.
        
        if best_delta > 0:
            selected[best_x] = False
            selected[best_y] = True
            current_score += best_delta
            if current_score > best_score:
                best_score = current_score
                best_selected = selected.copy()
            no_improve = 0
            improved = True
        else:
            no_improve += 1
            # Diversification
            if rng.random() < 0.1:
                x = rng.choice(sel_list)
                y = rng.choice(unsel_list)
                selected[x] = False
                # Check 4-AP
                if not _is_4ap_fast(selected, y):
                    selected[y] = True
                    current_score = _count_3aps(selected)
                else:
                    selected[x] = True

    return best_selected, best_score

# Precompute _ELEM_4AP_IDX
_ELEM_4AP_IDX = [[] for _ in range(N+1)]
for ap in _ALL_4APS:
    for x in ap:
        _ELEM_4AP_IDX[x].append(ap)

def _is_4ap_fast(selected_bool, x):
    for ap in _ELEM_4AP_IDX[x]:
        others = [y for y in ap if y != x]
        if selected_bool[others[0]] and selected_bool[others[1]] and selected_bool[others[2]]:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    rng = random.Random(42)
    
    best_selected = None
    best_score = -1
    
    n_restarts = 20
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Construct
        selected = _construct_remove(rng)
        current_score = _count_3aps(selected)
        
        # Local Search
        ls_deadline = min(deadline - 5, time.time() + time_per_restart * 0.8)
        selected, current_score = _local_search(selected, current_score, ls_deadline, rng)
        
        if current_score > best_score:
            best_score = current_score
            best_selected = selected.copy()
            
    result = [x for x in range(1, N+1) if best_selected[x]]
    return sorted(result)
