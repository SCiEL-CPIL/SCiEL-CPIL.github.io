import time
import numpy as np
import random

N = 64
P = N * N
DEADLINE = time.time() + 320

# Precompute coordinates
xs = np.repeat(np.arange(N), N)
ys = np.tile(np.arange(N), N)

# Precompute squared distances D[i, j]
# Using broadcasting to compute all pairs
# This creates a P x P array
# Memory: 4096 * 4096 * 4 bytes (int32) = 64 MB. Safe.
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
D = (dx * dx + dy * dy).astype(np.int32)

def is_valid_add(p, selected_bool):
    """
    Check if adding point p to the set defined by selected_bool is valid.
    selected_bool is a numpy boolean array of length P.
    """
    sel_idx = np.nonzero(selected_bool)[0]
    if len(sel_idx) < 2:
        return True
    
    # Distances from p to all selected points
    d_pa = D[p, sel_idx]
    
    # Condition 1: No two points in S are equidistant from p
    # i.e., d(p, a) != d(p, b) for all a != b in S
    if len(np.unique(d_pa)) != len(d_pa):
        return False
        
    # Condition 2 & 3: No point a in S has d(p, a) == d(a, b) for any b in S
    # and no point b in S has d(p, b) == d(a, b) for any a in S
    # Due to symmetry of distance, checking d(p, a) == d(a, b) for all a, b covers it.
    # We loop over each a in S.
    for i in range(len(sel_idx)):
        a = sel_idx[i]
        dist_pa = d_pa[i]
        # Check if dist_pa is present in D[a, sel_idx]
        # D[a, sel_idx] contains distances from a to all points in S
        # If dist_pa is in there, then d(p, a) == d(a, b) for some b.
        # Note: If b == a, d(a, a) = 0. dist_pa is > 0 (since p not in S). 
        # So we don't need to explicitly exclude a, unless p is in S (which it isn't).
        if np.any(D[a, sel_idx] == dist_pa):
            return False
            
    return True

def greedy_construct(order, selected_bool):
    """
    Greedily add points from 'order' to selected_bool if valid.
    """
    for p in order:
        if selected_bool[p]:
            continue
        if is_valid_add(p, selected_bool):
            selected_bool[p] = True

def local_search(selected_bool, num_steps=1000):
    """
    Attempt to improve the solution by adding points or swapping.
    """
    best_selected = selected_bool.copy()
    best_count = np.sum(best_selected)
    current_selected = selected_bool.copy()
    current_count = np.sum(current_selected)
    
    # Indices of all points
    all_points = np.arange(P)
    
    for _ in range(num_steps):
        if time.time() > DEADLINE:
            break
            
        # Try adding a random unselected point
        unselected = np.nonzero(~current_selected)[0]
        if len(unselected) > 0:
            p = unselected[random.randint(0, len(unselected) - 1)]
            if is_valid_add(p, current_selected):
                current_selected[p] = True
                current_count += 1
                if current_count > best_count:
                    best_selected = current_selected.copy()
                    best_count = current_count
                continue
        
        # Try swapping: remove a random selected point, try to add a random unselected point
        if current_count > 0:
            selected_indices = np.nonzero(current_selected)[0]
            rem_idx = selected_indices[random.randint(0, len(selected_indices) - 1)]
            
            # Temporarily remove
            current_selected[rem_idx] = False
            current_count -= 1
            
            # Try to add a random unselected point
            unselected = np.nonzero(~current_selected)[0]
            improved = False
            if len(unselected) > 0:
                # Try a few random candidates to increase chance of finding a valid swap
                for _ in range(5):
                    p = unselected[random.randint(0, len(unselected) - 1)]
                    if is_valid_add(p, current_selected):
                        current_selected[p] = True
                        current_count += 1
                        improved = True
                        if current_count > best_count:
                            best_selected = current_selected.copy()
                            best_count = current_count
                        break
            
            if not improved:
                # Revert removal
                current_selected[rem_idx] = True
                current_count += 1
                
    return best_selected

def construct():
    random.seed(42)
    np.random.seed(42)
    
    best_result_bool = np.zeros(P, dtype=bool)
    best_count = 0
    
    # Precompute a base order? No, randomize each time.
    base_order = np.arange(P)
    
    # Run multiple restarts
    while time.time() < DEADLINE:
        # Randomize order
        order = base_order.copy()
        np.random.shuffle(order)
        
        # Greedy phase
        current_selected = np.zeros(P, dtype=bool)
        greedy_construct(order, current_selected)
        
        current_count = np.sum(current_selected)
        
        if current_count > best_count:
            best_result_bool = current_selected.copy()
            best_count = current_count
            
        # Local search phase
        # Limit local search steps to ensure we can do multiple restarts
        # or just let it run until deadline?
        # Better to limit per-restart and do more restarts.
        # Let's say 500 steps per restart.
        improved_selected = local_search(current_selected, num_steps=500)
        
        improved_count = np.sum(improved_selected)
        if improved_count > best_count:
            best_result_bool = improved_selected.copy()
            best_count = improved_count
            
    # Convert boolean array to list of (x, y) tuples
    result_points = []
    for i in range(P):
        if best_result_bool[i]:
            x = i % N
            y = i // N
            result_points.append((x, y))
            
    return result_points
