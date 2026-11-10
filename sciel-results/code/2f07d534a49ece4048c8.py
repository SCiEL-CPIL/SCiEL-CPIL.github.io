import time
import random
import math
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 330s limit
    
    # Point representation: index 0..4095
    # x = i // N, y = i % N
    num_points = N * N
    
    # Precompute coordinates for faster access if needed, 
    # but index arithmetic is very fast.
    # xs = np.array([i // N for i in range(num_points)], dtype=np.int32)
    # ys = np.array([i % N for i in range(num_points)], dtype=np.int32)
    
    # Data structures for incremental validity checks
    selected = []
    point_set = set()
    # dist_sets[q] = set of squared distances from q to other selected points
    dist_sets = {}
    
    def _dist2_idx(i, j):
        xi = i // N
        yi = i % N
        xj = j // N
        yj = j % N
        dx = xi - xj
        dy = yi - yj
        return dx*dx + dy*dy

    def _can_add(p_idx):
        if p_idx in point_set:
            return False
        k = len(selected)
        if k == 0:
            return True
        
        seen_dists = set()
        for q_idx in selected:
            d = _dist2_idx(p_idx, q_idx)
            # Condition 1: Distances from p must be unique (no isosceles with vertex p)
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
            # Condition 2: d must not be in dist_sets[q_idx] (no isosceles with vertex q)
            if d in dist_sets[q_idx]:
                return False
        
        return True

    def _add(p_idx):
        point_set.add(p_idx)
        selected.append(p_idx)
        dist_sets[p_idx] = set()
        
        for q_idx in selected[:-1]: # All except p_idx
            d = _dist2_idx(p_idx, q_idx)
            dist_sets[p_idx].add(d)
            dist_sets[q_idx].add(d)

    def _remove(p_idx):
        point_set.remove(p_idx)
        # Remove from list
        try:
            idx = selected.index(p_idx)
            selected.pop(idx)
        except ValueError:
            pass
        
        # Update dist_sets for all remaining points
        for q_idx in selected:
            d = _dist2_idx(p_idx, q_idx)
            if d in dist_sets[q_idx]:
                dist_sets[q_idx].remove(d)
        
        if p_idx in dist_sets:
            del dist_sets[p_idx]
            
    def _is_valid_swap(remove_idx, add_idx):
        """
        Check if replacing remove_idx with add_idx results in a valid set.
        We do not modify state, just check.
        """
        if add_idx in point_set:
            return False
        if remove_idx not in point_set:
            return False
            
        # The new set will be: (selected - {remove_idx}) + {add_idx}
        # 1. Check distances from add_idx to all points in (selected - {remove_idx})
        #    - They must be unique among themselves.
        #    - They must not conflict with existing distances in dist_sets for those points.
        
        seen_dists = set()
        
        for q_idx in selected:
            if q_idx == remove_idx:
                continue
                
            d = _dist2_idx(add_idx, q_idx)
            
            # Uniqueness among new point's connections
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
            # Conflict with existing distances of q_idx.
            # Note: dist_sets[q_idx] currently contains distance to remove_idx.
            # If d == dist2(q_idx, remove_idx), it will be removed from dist_sets[q_idx] 
            # in the final state, so it's NOT a conflict with the *remaining* points.
            # However, we are checking if d is in the set of distances to *other* points.
            # Let's be precise.
            # In the new state, dist_sets_new[q_idx] = (dist_sets[q_idx] - {dist2(q_idx, remove_idx)}) + {d}
            # We need d not to be in (dist_sets[q_idx] - {dist2(q_idx, remove_idx)}).
            
            d_remove = _dist2_idx(q_idx, remove_idx)
            
            # If d is in dist_sets[q_idx], we must ensure it's ONLY because of remove_idx.
            # Since sets don't track multiplicity, if d is in dist_sets[q_idx], 
            # it could be due to remove_idx OR another point r.
            # If it's due to another point r, then adding add_idx creates a duplicate distance d for q_idx.
            
            # Efficient check:
            # We know d_remove is in dist_sets[q_idx].
            # We need to know if d is in dist_sets[q_idx] \ {d_remove}.
            
            if d in dist_sets[q_idx]:
                if d != d_remove:
                    return False
                # If d == d_remove, we need to be sure no OTHER point has this distance.
                # This is hard to check without iterating. 
                # But wait, if d == d_remove, and there is another point r with dist(q, r) == d, 
                # then dist_sets[q_idx] would contain d (from r) and d (from remove). 
                # Since it's a set, it just contains d.
                # So we can't distinguish.
                
                # Let's do a strict check for this case:
                # If d == d_remove, check if any other point r (r != remove_idx, r != add_idx) has dist(q, r) == d.
                # This is O(K). Given K is small, it's acceptable if rare.
                # However, this case (d == d_remove) is actually GOOD for the swap 
                # IF no other point has that distance.
                # If another point has that distance, the swap is invalid.
                
                found_other = False
                for r_idx in selected:
                    if r_idx == remove_idx or r_idx == add_idx:
                        continue
                    if _dist2_idx(q_idx, r_idx) == d:
                        found_other = True
                        break
                if found_other:
                    return False
            # If d not in dist_sets[q_idx], it's fine (unless d == d_remove was handled above? 
            # No, if d == d_remove, it IS in dist_sets[q_idx]).
            
        # 2. Check if add_idx itself has unique distances (handled by seen_dists)
        
        return True

    def _save_state():
        return {
            'selected': selected[:],
            'point_set': point_set.copy(),
            'dist_sets': {k: set(v) for k, v in dist_sets.items()}
        }

    def _restore_state(state):
        selected.clear()
        selected.extend(state['selected'])
        point_set.clear()
        point_set.update(state['point_set'])
        dist_sets.clear()
        for k, v in state['dist_sets'].items():
            dist_sets[k] = set(v)

    rng = random.Random(12345)
    
    # 1. Greedy Initialization
    best_size = 0
    best_state = None
    
    candidates = list(range(num_points))
    rng_init = random.Random(42)
    rng_init.shuffle(candidates)
    for p_idx in candidates:
        if _can_add(p_idx):
            _add(p_idx)
            
    best_size = len(selected)
    best_state = _save_state()

    # 2. Simulated Annealing
    # We want to maximize size.
    # Moves:
    # A: Add random point. If valid, size+1. Always accept.
    # B: Remove random point. Size-1. Accept with prob exp(-1/T).
    # C: Swap random in/out. Size same. Accept if valid.
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-4
    
    # To speed up, we can batch operations.
    # But SA needs step-by-step temperature update.
    
    # We will run until deadline.
    # If we get stuck (no adds for a long time), we might restart from best.
    
    last_improvement_time = start_time
    restart_interval = 30.0 # seconds
    
    while time.time() < deadline:
        # Pick a move type
        # Bias towards Add if possible, but allow Remove/Swap to escape.
        # Let's try: 50% Add, 25% Remove, 25% Swap
        
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try Add
            p_idx = rng.randint(0, num_points - 1)
            if p_idx not in point_set:
                if _can_add(p_idx):
                    _add(p_idx)
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_state = _save_state()
                        last_improvement_time = time.time()
                        
        elif move_type < 0.75:
            # Try Remove
            if selected:
                p_idx = selected[rng.randint(0, len(selected) - 1)]
                # Accept with prob exp(-1/T)
                if rng.random() < math.exp(-1.0 / T):
                    _remove(p_idx)
                    
        else:
            # Try Swap
            if selected:
                p_in = selected[rng.randint(0, len(selected) - 1)]
                p_out = rng.randint(0, num_points - 1)
                if p_out not in point_set:
                    if _is_valid_swap(p_in, p_out):
                        # Perform swap
                        _remove(p_in)
                        _add(p_out)
                        # Size is same, so no best_size update needed unless we want to track
                        # but best_state is for max size.
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
        # Restart if stuck for too long
        if time.time() - last_improvement_time > restart_interval:
            # Restore best state and restart with higher T
            _restore_state(best_state)
            T = 1.0
            last_improvement_time = time.time()
            
    # Convert best_state selected indices back to (x, y) tuples
    result = []
    for idx in best_state['selected']:
        x = idx // N
        y = idx % N
        result.append((x, y))
        
    return result
# EVOLVE-BLOCK-END
