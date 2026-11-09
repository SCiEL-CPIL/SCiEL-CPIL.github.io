import random
import time
import numpy as np

N = 64
M = N * N

def construct():
    deadline = time.time() + 295
    
    # Precompute coordinates for fast access
    # Index i -> (i % N, i // N)
    xs = np.arange(M) % N
    ys = np.arange(M) // N
    
    # Helper to compute squared distance between indices i and j
    def sq_dist(i, j):
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        return dx*dx + dy*dy

    # We will maintain:
    # points: list of indices
    # dist_set: set of all pairwise squared distances in 'points'
    # This ensures all pairwise distances are distinct, which implies no isosceles triangles.
    
    def try_add(p, points, dist_set):
        if p in points:
            return False, None, None
            
        n = len(points)
        if n == 0:
            return True, [p], set()
            
        new_dists = []
        # Check against existing distances and new distances among themselves
        for q in points:
            d = sq_dist(p, q)
            # If d is already in dist_set, it means some pair (a,b) in points has distance d.
            # If d == d(p,q), then (p,a,b) or similar forms an isosceles triangle?
            # Actually, if all pairwise distances are distinct, then d(p,q) must not be in dist_set.
            # Also d(p,q) must be unique for all q.
            if d in dist_set:
                return False, None, None
            if d in new_dists:
                return False, None, None
            new_dists.append(d)
            
        # If we get here, all new distances are unique and not in old set.
        new_points = points + [p]
        new_dist_set = set(dist_set)
        for d in new_dists:
            new_dist_set.add(d)
            
        return True, new_points, new_dist_set

    def try_remove(p, points, dist_set):
        if p not in points:
            return points, dist_set
            
        idx = points.index(p)
        # Remove distances involving p
        for q in points:
            if q == p:
                continue
            d = sq_dist(p, q)
            if d in dist_set:
                dist_set.remove(d)
                
        new_points = points[:idx] + points[idx+1:]
        return new_points, dist_set

    # Greedy Initialization
    random.seed(42)
    points = []
    dist_set = set()
    
    # Shuffle all indices
    candidates = list(range(M))
    random.shuffle(candidates)
    
    for p in candidates:
        if time.time() > deadline - 10:
            break
        success, new_points, new_dist_set = try_add(p, points, dist_set)
        if success:
            points = new_points
            dist_set = new_dist_set

    best_points = points[:]
    best_size = len(points)
    
    # Simulated Annealing
    # State: current_points, current_dist_set
    # Note: dist_set is modified in place in try_remove, so we must be careful with rollback.
    # It's safer to copy or undo. 
    # Given the overhead of copying a set of size ~k^2, and k~100, size~10000, copying is okay.
    # But let's try to do it efficiently.
    
    T = 1.0
    cooling = 0.9999
    min_T = 0.01
    
    current_points = points[:]
    current_dist_set = set(dist_set)
    
    all_indices = list(range(M))
    
    while time.time() < deadline - 5:
        # Generate Move
        move_type = random.random()
        
        if move_type < 0.3:
            # Add
            p = random.choice(all_indices)
            if p in current_points:
                continue
            
            success, new_points, new_dist_set = try_add(p, current_points, current_dist_set)
            if success:
                new_size = len(new_points)
                delta = new_size - len(current_points)
                
                if delta > 0:
                    # Always accept improvement
                    current_points = new_points
                    current_dist_set = new_dist_set
                    if len(current_points) > best_size:
                        best_size = len(current_points)
                        best_points = current_points[:]
                else:
                    # Should not happen if we only add new points, but logic is safe
                    pass
        elif move_type < 0.7:
            # Remove
            if not current_points:
                continue
            p = random.choice(current_points)
            new_points, new_dist_set = try_remove(p, current_points, current_dist_set)
            new_size = len(new_points)
            delta = new_size - len(current_points)
            
            # Accept if random < exp(-delta/T)
            # delta is negative
            if random.random() < np.exp(-delta / T):
                current_points = new_points
                current_dist_set = new_dist_set
        else:
            # Swap: Remove one, Add one
            if not current_points:
                continue
            p_remove = random.choice(current_points)
            # Tentative remove
            # We need to restore state if we don't accept
            # Copying sets is expensive. 
            # Alternative: Just do remove, then try add. If add fails or not accepted, revert.
            
            # To avoid complex rollback, we can just evaluate the delta.
            # But try_add needs the state *after* removal.
            
            # Let's do:
            # 1. Remove p_remove
            # 2. Try add p_add
            # 3. If success, calculate delta.
            # 4. If accept, keep. Else, revert.
            
            # Reverting requires storing the state before remove.
            # Since k is small, copying the set of distances is feasible.
            old_points = current_points[:]
            old_dist_set = set(current_dist_set)
            
            new_points, new_dist_set = try_remove(p_remove, current_points, current_dist_set)
            
            p_add = random.choice(all_indices)
            if p_add in new_points:
                # Retry add with a different point
                # To save time, just skip this move if collision
                # Restore state
                current_points = old_points
                current_dist_set = old_dist_set
                continue
                
            success, final_points, final_dist_set = try_add(p_add, new_points, new_dist_set)
            
            if success:
                new_size = len(final_points)
                old_size = len(old_points)
                delta = new_size - old_size
                
                accept = False
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = random.random() < 0.5
                else:
                    accept = random.random() < np.exp(-delta / T)
                
                if accept:
                    current_points = final_points
                    current_dist_set = final_dist_set
                    if len(current_points) > best_size:
                        best_size = len(current_points)
                        best_points = current_points[:]
            else:
                # Add failed, revert remove
                current_points = old_points
                current_dist_set = old_dist_set
        
        T *= cooling
        if T < min_T:
            T = min_T
            
    # Convert best_points indices to coordinates
    result = []
    for idx in best_points:
        x = idx % N
        y = idx // N
        result.append((x, y))
        
    return result
