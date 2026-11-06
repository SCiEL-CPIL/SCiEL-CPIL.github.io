"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import itertools

# Precompute all points
ALL_POINTS = list(itertools.product(range(N), repeat=2))
POINT_INDEX = {p: i for i, p in enumerate(ALL_POINTS)}
NUM_POINTS = len(ALL_POINTS)

# Precompute squared distances between all pairs of points
_points_arr = np.array(ALL_POINTS)
diff = _points_arr[:, np.newaxis, :] - _points_arr[np.newaxis, :, :]
DIST2 = np.sum(diff**2, axis=2)

def _get_dist_set(chosen_indices):
    """Compute the set of all squared distances between pairs in chosen_indices."""
    dists = set()
    k = len(chosen_indices)
    for i in range(k):
        ci = chosen_indices[i]
        for j in range(i + 1, k):
            dists.add(DIST2[ci, cj])
    return dists

def _can_add_index(p_idx, chosen_indices, dist_set_S):
    """
    Check if point p_idx can be added to the set chosen_indices.
    dist_set_S: set of squared distances between all pairs in chosen_indices.
    """
    if len(chosen_indices) < 2:
        return True
    
    # Get distances from p_idx to all points in chosen_indices
    d_p_c = [DIST2[p_idx, c] for c in chosen_indices]
    
    # 1. Check for duplicate distances from p (p is apex)
    if len(set(d_p_c)) != len(d_p_c):
        return False
        
    # 2. Check against distances between chosen points
    d_p_c_set = set(d_p_c)
    if d_p_c_set.intersection(dist_set_S):
        return False
        
    return True

def _is_maximal(chosen_indices, dist_set_S):
    """
    Check if the set chosen_indices is maximal.
    """
    chosen_set = set(chosen_indices)
    for p_idx in range(NUM_POINTS):
        if p_idx not in chosen_set:
            if _can_add_index(p_idx, chosen_indices, dist_set_S):
                return False
    return True

def _greedy_construct(order_indices):
    """
    Greedy construction given an order of point indices.
    """
    chosen = []
    dist_set_S = set()
    for p_idx in order_indices:
        if _can_add_index(p_idx, chosen, dist_set_S):
            # Update dist_set_S
            for c in chosen:
                dist_set_S.add(DIST2[p_idx, c])
            chosen.append(p_idx)
    return chosen, dist_set_S

def _make_maximal(chosen_indices, dist_set_S, all_indices):
    """
    Ensure the set is maximal by adding points.
    Returns updated chosen_indices and dist_set_S.
    """
    chosen_set = set(chosen_indices)
    # We need to scan all points to see if any can be added
    # Since we want to be maximal, we add any that can be added.
    # Order doesn't matter for maximality, but affects the specific set.
    # We iterate all_indices.
    for p_idx in all_indices:
        if p_idx not in chosen_set:
            if _can_add_index(p_idx, chosen_indices, dist_set_S):
                # Add it
                for c in chosen_indices:
                    dist_set_S.add(DIST2[p_idx, c])
                chosen_indices.append(p_idx)
                chosen_set.add(p_idx)
    return chosen_indices, dist_set_S

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit
    
    random.seed(42)
    
    best_solution_indices = None
    best_size = float('inf')
    best_dist_set = None
    
    all_indices = list(range(NUM_POINTS))
    
    # 1. Initial Greedy Constructions
    # Try different random orders
    for _ in range(50):
        if time.time() > deadline:
            break
        order = all_indices[:]
        random.shuffle(order)
        current, current_dist_set = _greedy_construct(order)
        
        # Ensure maximality
        current, current_dist_set = _make_maximal(current, current_dist_set, all_indices)
        
        if len(current) < best_size:
            best_size = len(current)
            best_solution_indices = current[:]
            best_dist_set = current_dist_set.copy()
            
    if best_solution_indices is None:
        # Fallback
        current, current_dist_set = _greedy_construct(all_indices)
        current, current_dist_set = _make_maximal(current, current_dist_set, all_indices)
        best_solution_indices = current
        best_dist_set = current_dist_set
        best_size = len(current)
        
    current = best_solution_indices[:]
    current_dist_set = best_dist_set.copy()
    current_size = best_size
    
    # 2. Local Search
    # Moves: Removal, Swap
    
    while time.time() < deadline - 5:
        
        if current_size < best_size:
            best_size = current_size
            best_solution_indices = current[:]
            best_dist_set = current_dist_set.copy()
            
        move_type = random.random()
        
        if move_type < 0.3:
            # Try Removal
            if len(current) == 0:
                continue
            p_idx = random.choice(current)
            new_current = [x for x in current if x != p_idx]
            
            # Recompute dist set for new_current
            new_dist_set = _get_dist_set(new_current)
            
            if _is_maximal(new_current, new_dist_set):
                current = new_current
                current_dist_set = new_dist_set
                current_size -= 1
        elif move_type < 0.8:
            # Try Swap
            if len(current) == 0:
                continue
            p_idx = random.choice(current)
            # Pick q not in current
            q_idx = random.randint(0, NUM_POINTS - 1)
            if q_idx in current:
                # Retry a few times
                for _ in range(10):
                    q_idx = random.randint(0, NUM_POINTS - 1)
                    if q_idx not in current:
                        break
                else:
                    continue
            
            temp_current = [x for x in current if x != p_idx]
            temp_dist_set = _get_dist_set(temp_current)
            
            if _can_add_index(q_idx, temp_current, temp_dist_set):
                # Add q_idx to temp_current to form new_current
                new_current = temp_current + [q_idx]
                # Update dist set
                new_dist_set = temp_dist_set.copy()
                for c in temp_current:
                    new_dist_set.add(DIST2[q_idx, c])
                
                if _is_maximal(new_current, new_dist_set):
                    current = new_current
                    current_dist_set = new_dist_set
                    # Size remains current_size
        else:
            # Try 2-Swap or just pass
            pass

    # Final check
    if best_solution_indices is None:
        best_solution_indices = current
        
    return [ALL_POINTS[i] for i in best_solution_indices]

# EVOLVE-BLOCK-END
