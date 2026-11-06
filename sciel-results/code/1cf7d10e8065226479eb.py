import itertools
import time
import random
import numpy as np

N = 16
V = N * N

def _precompute():
    """Precompute conflict bitmasks and distance structures."""
    points = list(itertools.product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # Pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2)
    
    # conflict[a, b] = bitmask of all points p such that {a, b, p} forms an isosceles triangle
    # i.e., d(a,p)==d(a,b) or d(b,p)==d(a,b) or d(a,p)==d(b,p)
    conflict = np.zeros((V, V), dtype=object)
    
    # For efficiency, precompute for each point a and each distance d, the bitmask of points at that distance
    # But let's do it more directly with numpy
    
    # Actually, let's use a different approach:
    # For each pair (a,b), compute the conflict mask
    # We can vectorize over b for fixed a
    
    for a in range(V):
        # For fixed a, we need conflict[a, b] for all b
        # d(a, b) for all b: dists[a, :]
        # For each b, conflict includes:
        #   m1: points p where d(a,p) == d(a,b)
        #   m2: points p where d(b,p) == d(a,b)
        #   m3: points p where d(a,p) == d(b,p)
        
        # Let's compute this for all b at once using broadcasting where possible
        # This is complex, so let's do it pair by pair but with numpy for the masks
        
        for b in range(a + 1, V):
            d_ab = dists[a, b]
            
            # m1: p where d(a,p) == d_ab
            m1 = (dists[a, :] == d_ab)
            
            # m2: p where d(b,p) == d_ab
            m2 = (dists[b, :] == d_ab)
            
            # m3: p where d(a,p) == d(b,p)
            m3 = (dists[a, :] == dists[b, :])
            
            combined = m1 | m2 | m3
            
            # Convert to bitmask
            # Use numpy to pack bits efficiently
            mask_val = 0
            indices = np.where(combined)[0]
            for p in indices:
                mask_val |= (1 << p)
            
            conflict[a, b] = mask_val
            conflict[b, a] = mask_val
    
    return points, conflict, dists

def _greedy_with_backtrack(points, conflict, dists, seed, max_backtrack_depth=3):
    """Greedy construction with limited backtracking."""
    random.seed(seed)
    order = list(range(V))
    random.shuffle(order)
    
    chosen = []
    chosen_set = set()
    blocked_mask = 0
    chosen_mask = 0
    
    # We'll use a simple approach: greedy, then try to remove points that allow adding others
    # Actually, let's do pure greedy first, then local improvement
    
    for p in order:
        if (chosen_mask | blocked_mask) & (1 << p):
            continue
        
        # Check if p can be added (it's not chosen and not blocked)
        # Add p
        chosen.append(p)
        chosen_set.add(p)
        chosen_mask |= (1 << p)
        
        # Update blocked mask
        for q in chosen:
            if q == p:
                continue
            blocked_mask |= conflict[p][q]
    
    # Local improvement: try to remove a point and see if we can add more
    # Actually, for minimality, we want to REMOVE points while maintaining maximality
    # This is harder. Let's instead try different orderings and keep the best.
    
    return [points[i] for i in chosen]

def _greedy_minimal(points, conflict, dists, seed):
    """
    Greedy construction that aims for a minimal (inclusion-wise) maximal set.
    Strategy: build the set, then try to remove each point and check if the set
    is still maximal. If removing a point still leaves it maximal, keep it removed.
    """
    random.seed(seed)
    order = list(range(V))
    random.shuffle(order)
    
    chosen = []
    chosen_set = set()
    blocked_mask = 0
    chosen_mask = 0
    
    for p in order:
        if (chosen_mask | blocked_mask) & (1 << p):
            continue
        
        chosen.append(p)
        chosen_set.add(p)
        chosen_mask |= (1 << p)
        
        for q in chosen:
            if q == p:
                continue
            blocked_mask |= conflict[p][q]
    
    # Now try to minimize: for each point in chosen, try removing it
    # and check if the set is still maximal
    # A set is maximal if no point can be added without creating an isosceles triangle
    # i.e., every unchosen point is blocked by the current set
    
    # Recompute blocked mask from scratch for the current chosen set
    def compute_blocked(chosen_list):
        bm = 0
        cl = list(chosen_list)
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                bm |= conflict[cl[i]][cl[j]]
        return bm
    
    # Try removing each point
    for idx in range(len(chosen) - 1, -1, -1):
        p = chosen[idx]
        
        # Temporarily remove p
        new_chosen = chosen[:idx] + chosen[idx+1:]
        
        if not new_chosen:
            continue
        
        # Check if new_chosen is still maximal
        # Compute blocked mask for new_chosen
        new_blocked = compute_blocked(new_chosen)
        
        # The set is maximal if every point not in new_chosen is blocked
        # i.e., for all q not in new_chosen, q is in new_blocked
        new_chosen_mask = 0
        for q in new_chosen:
            new_chosen_mask |= (1 << q)
        
        # All points should be either chosen or blocked
        all_points_mask = (1 << V) - 1
        if (new_blocked | new_chosen_mask) == all_points_mask:
            # Still maximal! Keep p removed
            chosen = new_chosen
            chosen_set.discard(p)
            chosen_mask &= ~(1 << p)
    
    return [points[i] for i in chosen]

def construct():
    start_time = time.time()
    points, conflict, dists = _precompute()
    
    best_set = None
    best_size = float('inf')
    
    deadline = start_time + 280  # 50s margin before 330s kill
    
    seed = 0
    while time.time() < deadline:
        # Alternate between different strategies
        if seed % 3 == 0:
            result = _greedy_minimal(points, conflict, dists, seed)
        elif seed % 3 == 1:
            result = _greedy_with_backtrack(points, conflict, dists, seed)
        else:
            # Another variant: greedy with different randomization
            random.seed(seed)
            order = list(range(V))
            # Sort by some heuristic? Let's try sorting by degree of conflict
            # Points that conflict with fewer others might be better to add first?
            # Or the opposite?
            random.shuffle(order)
            
            chosen = []
            chosen_set = set()
            blocked_mask = 0
            chosen_mask = 0
            
            for p in order:
                if (chosen_mask | blocked_mask) & (1 << p):
                    continue
                chosen.append(p)
                chosen_set.add(p)
                chosen_mask |= (1 << p)
                for q in chosen:
                    if q == p:
                        continue
                    blocked_mask |= conflict[p][q]
            
            # Try minimization
            def compute_blocked(chosen_list):
                bm = 0
                cl = list(chosen_list)
                for i in range(len(cl)):
                    for j in range(i + 1, len(cl)):
                        bm |= conflict[cl[i]][cl[j]]
                return bm
            
            for idx in range(len(chosen) - 1, -1, -1):
                p = chosen[idx]
                new_chosen = chosen[:idx] + chosen[idx+1:]
                if not new_chosen:
                    continue
                new_blocked = compute_blocked(new_chosen)
                new_chosen_mask = 0
                for q in new_chosen:
                    new_chosen_mask |= (1 << q)
                all_points_mask = (1 << V) - 1
                if (new_blocked | new_chosen_mask) == all_points_mask:
                    chosen = new_chosen
            
            result = [points[i] for i in chosen]
        
        if result:
            size = len(result)
            if size < best_size:
                best_size = size
                best_set = result
        
        seed += 1
    
    if best_set is None:
        best_set = [points[0]]
    
    return best_set
