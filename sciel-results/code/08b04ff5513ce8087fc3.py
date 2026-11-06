import time
import random

N = 64
TIME_LIMIT = 320.0

def _dist2(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def _can_add_fast(new_pt, pts, dist_sets):
    """
    pts: list of points
    dist_sets: list of sets, where dist_sets[i] is the set of squared distances from pts[i] to all other points in pts.
    """
    # 1. Check apex at new_pt
    dists_to_new = []
    for p in pts:
        d = _dist2(new_pt, p)
        dists_to_new.append(d)
        
    if len(dists_to_new) != len(set(dists_to_new)):
        return False
        
    # 2. Check apex at p in pts
    # For each p_i, if d(new, p_i) is in dist_sets[i], then there exists p_j such that d(p_i, p_j) == d(new, p_i)
    # This forms isosceles (new, p_i, p_j) with apex p_i.
    for i, d in enumerate(dists_to_new):
        if d in dist_sets[i]:
            return False
            
    return True

def _update_dist_sets(pts, dist_sets, new_pt_idx):
    """
    Update dist_sets after adding a point.
    new_pt_idx is the index of the new point in pts.
    """
    new_pt = pts[new_pt_idx]
    # Update existing sets
    for i in range(new_pt_idx):
        d = _dist2(pts[i], new_pt)
        dist_sets[i].add(d)
    # Create new set for the new point
    new_set = set()
    for i in range(new_pt_idx):
        d = _dist2(new_pt, pts[i])
        new_set.add(d)
    dist_sets.append(new_set)

def _greedy_construct(seed):
    rng = random.Random(seed)
    all_pts = [(x, y) for x in range(N) for y in range(N)]
    rng.shuffle(all_pts)
    
    pts = []
    dist_sets = []
    
    for pt in all_pts:
        if _can_add_fast(pt, pts, dist_sets):
            pts.append(pt)
            _update_dist_sets(pts, dist_sets, len(pts) - 1)
            
    return pts, dist_sets

def _local_search(pts, dist_sets, time_budget, seed):
    rng = random.Random(seed)
    all_pts = [(x, y) for x in range(N) for y in range(N)]
    start_time = time.time()
    current_pts = list(pts)
    current_dist_sets = [set(s) for s in dist_sets]
    best_pts = list(current_pts)
    best_size = len(current_pts)
    
    current_set = set(current_pts)
    
    while (time.time() - start_time) < time_budget:
        # Try to add a random point
        pt = rng.choice(all_pts)
        if pt not in current_set:
            if _can_add_fast(pt, current_pts, current_dist_sets):
                current_pts.append(pt)
                _update_dist_sets(current_pts, current_dist_sets, len(current_pts) - 1)
                current_set.add(pt)
                if len(current_pts) > best_size:
                    best_size = len(current_pts)
                    best_pts = list(current_pts)
                continue
                
        # Try a swap
        if len(current_pts) == 0:
            continue
            
        i = rng.randint(0, len(current_pts) - 1)
        pt_to_remove = current_pts[i]
        
        # Rebuild state without pt_to_remove
        # This is O(k^2)
        temp_pts = []
        for j, p in enumerate(current_pts):
            if j != i:
                temp_pts.append(p)
                
        temp_dist_sets = []
        for idx, p in enumerate(temp_pts):
            s = set()
            for other_idx, other_p in enumerate(temp_pts):
                if other_idx != idx:
                    s.add(_dist2(p, other_p))
            temp_dist_sets.append(s)
            
        # Try to add a random new point
        pt_to_add = rng.choice(all_pts)
        if pt_to_add in current_set:
            continue
            
        if _can_add_fast(pt_to_add, temp_pts, temp_dist_sets):
            # Accept swap
            current_pts = temp_pts + [pt_to_add]
            current_dist_sets = temp_dist_sets
            _update_dist_sets(current_pts, current_dist_sets, len(current_pts) - 1)
            
            current_set.discard(pt_to_remove)
            current_set.add(pt_to_add)
            
            if len(current_pts) > best_size:
                best_size = len(current_pts)
                best_pts = list(current_pts)
                
    return best_pts

def construct():
    start_time = time.time()
    best_pts = []
    best_size = 0
    
    seed = 0
    while (time.time() - start_time) < TIME_LIMIT - 10:
        pts, dist_sets = _greedy_construct(seed)
        if len(pts) > best_size:
            best_size = len(pts)
            best_pts = pts
            
        # Run local search
        if len(pts) >= best_size:
            remaining = TIME_LIMIT - 5 - (time.time() - start_time)
            if remaining > 5:
                ls_time = min(10.0, remaining / 2)
                improved_pts = _local_search(pts, dist_sets, ls_time, seed)
                if len(improved_pts) > best_size:
                    best_size = len(improved_pts)
                    best_pts = improved_pts
                    
        seed += 1
        
    return best_pts
