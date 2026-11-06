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

def _can_add_index(p_idx, chosen_indices):
    """
    Check if point p_idx can be added to the set of chosen_indices without creating an isosceles triangle.
    chosen_indices: list of indices into ALL_POINTS.
    """
    if len(chosen_indices) < 2:
        return True
    
    p_idx = int(p_idx)
    # Distances from p to all chosen points
    d_p_c = DIST2[p_idx, chosen_indices]
    
    k = len(chosen_indices)
    
    # 1. Check for duplicate distances from p (p is apex)
    # Since k is small, use a set
    seen_dists = set()
    for d in d_p_c:
        if d in seen_dists:
            return False
        seen_dists.add(d)
        
    # 2 & 3. Check against distances between chosen points
    # For each pair i, j in chosen_indices:
    # d_ij = DIST2[chosen_indices[i], chosen_indices[j]]
    # if d_p_c[i] == d_ij or d_p_c[j] == d_ij: return False
    
    for i in range(k):
        ci = chosen_indices[i]
        dpi = d_p_c[i]
        for j in range(i + 1, k):
            cj = chosen_indices[j]
            dij = DIST2[ci, cj]
            if dpi == dij:
                return False
            dpj = d_p_c[j]
            if dpj == dij:
                return False
                
    return True

def _is_maximal(chosen_indices):
    """
    Check if the set of chosen_indices is maximal.
    A set is maximal if no other point can be added.
    """
    chosen_set = set(chosen_indices)
    for p_idx in range(NUM_POINTS):
        if p_idx not in chosen_set:
            if _can_add_index(p_idx, chosen_indices):
                return False
    return True

def _greedy_construct(order_indices):
    """
    Greedy construction given an order of point indices.
    Returns a list of indices.
    """
    chosen = []
    for p_idx in order_indices:
        if _can_add_index(p_idx, chosen):
            chosen.append(p_idx)
    return chosen

def _make_maximal(chosen_indices, all_indices):
    """
    Ensure the set is maximal by adding points.
    """
    chosen_set = set(chosen_indices)
    remaining = [p for p in all_indices if p not in chosen_set]
    random.shuffle(remaining)
    
    new_chosen = list(chosen_indices)
    new_chosen_set = set(new_chosen)
    
    for p_idx in remaining:
        if p_idx not in new_chosen_set:
            if _can_add_index(p_idx, new_chosen):
                new_chosen.append(p_idx)
                new_chosen_set.add(p_idx)
                
    return new_chosen

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit
    
    random.seed(42)
    
    best_solution_indices = None
    best_size = float('inf')
    
    all_indices = list(range(NUM_POINTS))
    
    # 1. Initial Greedy Constructions
    # Try different random orders
    for _ in range(10):
        if time.time() > deadline:
            break
        order = all_indices[:]
        random.shuffle(order)
        current = _greedy_construct(order)
        current = _make_maximal(current, all_indices)
        if len(current) < best_size:
            best_size = len(current)
            best_solution_indices = current[:]
            
    if best_solution_indices is None:
        # Fallback
        best_solution_indices = _make_maximal(_greedy_construct(all_indices), all_indices)
        best_size = len(best_solution_indices)
        
    current = best_solution_indices[:]
    current_size = best_size
    
    # 2. Local Search with Simulated Annealing
    # Moves:
    # - Removal: size - 1 (only if maximal)
    # - Swap: size 0 (explore)
    # - 2-Swap: size 0 (explore)
    
    temp = 1.0
    cooling = 0.999
    
    # Precompute list of non-chosen points for faster swap selection?
    # We'll just pick random indices and check membership.
    
    num_iterations = 50000
    for i in range(num_iterations):
        if time.time() > deadline - 5:
            break
            
        if current_size < best_size:
            best_size = current_size
            best_solution_indices = current[:]
            
        move_type = random.random()
        
        if move_type < 0.3:
            # Try Removal
            if len(current) == 0:
                continue
            p_idx = random.choice(current)
            new_current = [x for x in current if x != p_idx]
            if _is_maximal(new_current):
                current = new_current
                current_size -= 1
                temp = max(0.1, temp * cooling) # Reset or cool? Cooling is fine.
            else:
                # Rejection
                pass
        elif move_type < 0.7:
            # Try Swap
            if len(current) == 0:
                continue
            p_idx = random.choice(current)
            # Pick q not in current
            q_idx = random.randint(0, NUM_POINTS - 1)
            if q_idx in current:
                for _ in range(5):
                    q_idx = random.randint(0, NUM_POINTS - 1)
                    if q_idx not in current:
                        break
                else:
                    continue
            
            temp_current = [x for x in current if x != p_idx]
            if _can_add_index(q_idx, temp_current):
                new_current = temp_current + [q_idx]
                if _is_maximal(new_current):
                    # Accept swap. We don't update best_size as size is same.
                    # But we accept to explore.
                    current = new_current
                    # Size remains current_size
                else:
                    # Not maximal, reject swap
                    pass
            else:
                pass
        else:
            # Try 2-Swap
            if len(current) < 2:
                continue
            # Pick two distinct points from current
            p1_idx, p2_idx = random.sample(current, 2)
            # Pick two distinct points not in current
            q1_idx = random.randint(0, NUM_POINTS - 1)
            while q1_idx in current:
                q1_idx = random.randint(0, NUM_POINTS - 1)
            
            q2_idx = random.randint(0, NUM_POINTS - 1)
            while q2_idx in current or q2_idx == q1_idx:
                q2_idx = random.randint(0, NUM_POINTS - 1)
                
            temp_current = [x for x in current if x != p1_idx and x != p2_idx]
            
            # Check if q1 can be added
            if not _can_add_index(q1_idx, temp_current):
                continue
            temp_current_with_q1 = temp_current + [q1_idx]
            
            # Check if q2 can be added to temp_current_with_q1
            if not _can_add_index(q2_idx, temp_current_with_q1):
                continue
            new_current = temp_current_with_q1 + [q2_idx]
            
            # Check maximality
            if _is_maximal(new_current):
                current = new_current
                # Size remains current_size
                
        temp = max(0.01, temp * cooling)

    # Final check
    if best_solution_indices is None:
        best_solution_indices = current
        
    return [ALL_POINTS[i] for i in best_solution_indices]

# EVOLVE-BLOCK-END
