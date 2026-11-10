import time
import random
import numpy as np

N = 256

# Precompute all 4-APs as tuples of 4 elements
_ALL_4APS = []
for _a in range(1, N+1):
    for _d in range(1, (N - _a) // 3 + 1):
        _ALL_4APS.append((_a, _a+_d, _a+2*_d, _a+3*_d))

# Precompute for each x, the list of 4-APs it belongs to, with the other 3 elements
_X_4APS = [[] for _ in range(N+1)]
for _ap in _ALL_4APS:
    for _i in range(4):
        _x = _ap[_i]
        _others = [ap for j, ap in enumerate(_ap) if j != _i]
        _X_4APS[_x].append(tuple(_others))

# Precompute for each x, the list of (u, v) pairs such that {x, u, v} is a 3-AP
_X_3AP_PARTNERS = [[] for _ in range(N+1)]
for _a in range(1, N+1):
    for _d in range(1, (N - _a) // 2 + 1):
        _b = _a + _d
        _c = _a + 2*_d
        _X_3AP_PARTNERS[_a].append((_b, _c))
        _X_3AP_PARTNERS[_b].append((_a, _c))
        _X_3AP_PARTNERS[_c].append((_a, _b))

def _has_4ap_fast(selected_bool, x):
    """Check if adding x to selected_bool creates a 4-AP using precomputed data."""
    for others in _X_4APS[x]:
        if selected_bool[others[0]] and selected_bool[others[1]] and selected_bool[others[2]]:
            return True
    return False

def _count_3aps_fast(selected_bool):
    """Count 3-APs using boolean array."""
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
    """Count 3-APs that involve x and would be created by adding x."""
    delta = 0
    for u, v in _X_3AP_PARTNERS[x]:
        if selected_bool[u] and selected_bool[v]:
            delta += 1
    return delta

def _greedy_construct(rng, alpha=0.5):
    """GRASP construction with randomness controlled by alpha."""
    selected = np.zeros(N+1, dtype=bool)
    candidates = list(range(1, N+1))
    
    while candidates:
        # Filter candidates that don't create 4-AP
        valid_candidates = []
        for x in candidates:
            if not _has_4ap_fast(selected, x):
                # Calculate delta
                d = _delta_3aps(selected, x)
                valid_candidates.append((d, x))
        
        if not valid_candidates:
            break
            
        # Sort by delta descending
        valid_candidates.sort(key=lambda t: t[0], reverse=True)
        
        # Choose from top candidates with some randomness
        # If alpha is low, choose the best. If high, choose randomly from top.
        top_k = max(1, int(len(valid_candidates) * alpha))
        if top_k >= len(valid_candidates):
            top_k = len(valid_candidates)
            
        # Randomly pick from the top_k
        # To be more effective, pick from the top 10 or so
        pick_pool = valid_candidates[:min(top_k, 10)]
        _, x = rng.choice(pick_pool)
        
        selected[x] = True
        candidates.remove(x)
        
    return selected

def _local_search(selected, current_score, deadline, rng):
    """Local search to improve the score."""
    best_selected = selected.copy()
    best_score = current_score
    no_improve = 0
    max_no_improve = 40
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        
        sel_list = [x for x in range(1, N+1) if selected[x]]
        unsel_list = [x for x in range(1, N+1) if not selected[x]]
        
        if not sel_list or not unsel_list:
            break
            
        n_swaps = min(100, len(sel_list) * len(unsel_list))
        for _ in range(n_swaps):
            x = rng.choice(sel_list)
            y = rng.choice(unsel_list)
            
            selected[x] = False
            if not _has_4ap_fast(selected, y):
                selected[y] = True
                new_score = current_score - _delta_3aps(selected, x) + _delta_3aps(selected, y)
                if new_score > current_score:
                    actual_new_score = _count_3aps_fast(selected)
                    if actual_new_score > best_score:
                        best_selected = selected.copy()
                        best_score = actual_new_score
                        current_score = actual_new_score
                        no_improve = 0
                        improved = True
                    else:
                        selected[x] = True
                        selected[y] = False
                else:
                    selected[x] = True
                    selected[y] = False
            else:
                selected[x] = True
                
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
            
    return best_selected, best_score

def _sa_polish(selected, current_score, deadline, rng):
    """Simulated annealing polish to escape local optima."""
    best_selected = selected.copy()
    best_score = current_score
    temp = 3.0
    cooling = 0.97
    
    while time.time() < deadline and temp > 0.05:
        sel_list = [x for x in range(1, N+1) if selected[x]]
        unsel_list = [x for x in range(1, N+1) if not selected[x]]
        
        if not sel_list or not unsel_list:
            break
        
        x = rng.choice(sel_list)
        y = rng.choice(unsel_list)
        
        selected[x] = False
        if not _has_4ap_fast(selected, y):
            selected[y] = True
            new_score = _count_3aps_fast(selected)
            delta = new_score - current_score
            
            if delta > 0 or rng.random() < np.exp(delta / temp):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_selected = selected.copy()
            else:
                selected[x] = True
                selected[y] = False
        else:
            selected[x] = True
        
        temp *= cooling
    
    return best_selected, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    rng = random.Random(42)
    
    best_selected = None
    best_score = -1
    
    n_restarts = 15
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # GRASP construction
        selected = _greedy_construct(rng, alpha=0.3)
        current_score = _count_3aps_fast(selected)
        
        # Local search
        ls_deadline = min(deadline - 10, time.time() + time_per_restart * 0.5)
        selected, current_score = _local_search(selected, current_score, ls_deadline, rng)
        
        # SA polish
        sa_deadline = min(deadline - 5, time.time() + time_per_restart * 0.4)
        selected, current_score = _sa_polish(selected, current_score, sa_deadline, rng)
        
        if current_score > best_score:
            best_score = current_score
            best_selected = selected.copy()
    
    result = [x for x in range(1, N+1) if best_selected[x]]
    return sorted(result)
