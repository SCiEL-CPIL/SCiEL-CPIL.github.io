import numpy as np
import random
import time
from collections import defaultdict

N = 16
GRID_POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(GRID_POINTS)

# Precompute squared distances
coords = np.array(GRID_POINTS, dtype=np.int32)
diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist_sq = np.sum(diffs * diffs, axis=-1)

# Precompute for each point p, the set of distances from p to all other points
# dist_to_others[p] = array of dist_sq[p, q] for q != p
dist_to_others = []
for p in range(NUM_POINTS):
    d = dist_sq[p].copy()
    d[p] = -1  # Mark self
    dist_to_others.append(d)

def _can_add(p, S_set, dist_used):
    """
    Check if point p can be added to set S_set.
    dist_used[a] = set of distances already used by point a to other points in S.
    Returns True if p can be added.
    """
    if not S_set:
        return True
    
    # Check 1: distances from p to all a in S must be unique
    seen_dists = set()
    for a in S_set:
        d = dist_sq[p, a]
        if d in seen_dists:
            return False
        seen_dists.add(d)
    
    # Check 2: for each a in S, d(p,a) must not be in dist_used[a]
    for a in S_set:
        if dist_sq[p, a] in dist_used[a]:
            return False
    
    return True

def _is_maximal(S_set, dist_used):
    """
    Check if S_set is maximal: no point outside can be added.
    """
    for q in range(NUM_POINTS):
        if q in S_set:
            continue
        if _can_add(q, S_set, dist_used):
            return False
    return True

def _build_dist_used(S_list):
    """Build dist_used dict from a list of point indices."""
    S_set = set(S_list)
    dist_used = {a: set() for a in S_list}
    for a in S_list:
        for b in S_list:
            if a != b:
                dist_used[a].add(dist_sq[a, b])
    return S_set, dist_used

def _greedy_maximal(order, deadline=None):
    """
    Greedily construct a maximal no-isosceles set.
    Returns (S_list, S_set, dist_used)
    """
    S_list = []
    S_set = set()
    dist_used = {}
    
    for p in order:
        if p in S_set:
            continue
        
        # Check if p can be added
        can_add = True
        if S_list:
            seen_dists = set()
            for a in S_list:
                d = dist_sq[p, a]
                if d in seen_dists:
                    can_add = False
                    break
                seen_dists.add(d)
                if d in dist_used.get(a, set()):
                    can_add = False
                    break
        
        if can_add:
            S_list.append(p)
            S_set.add(p)
            
            # Update dist_used
            for a in S_list[:-1]:
                d = dist_sq[p, a]
                if a not in dist_used:
                    dist_used[a] = set()
                dist_used[a].add(d)
            
            if S_list[:-1]:
                dist_used[p] = set(dist_sq[p, a] for a in S_list[:-1])
            else:
                dist_used[p] = set()
    
    return S_list, S_set, dist_used

def _local_search_minimize(S_list, S_set, dist_used, deadline):
    """
    Try to minimize by removing points.
    Returns (S_list, S_set, dist_used)
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        shuffled = S_list[:]
        random.shuffle(shuffled)
        
        for p in shuffled:
            if p not in S_set:
                continue
            
            # Try removing p
            S_new = [x for x in S_list if x != p]
            S_new_set, dist_used_new = _build_dist_used(S_new)
            
            if _is_maximal(S_new_set, dist_used_new):
                S_list = S_new
                S_set = S_new_set
                dist_used = dist_used_new
                improved = True
                break
    
    return S_list, S_set, dist_used

def _try_swap_improve(S_list, S_set, dist_used, deadline):
    """
    Try to improve by swapping one point for another.
    Returns (S_list, S_set, dist_used, improved)
    """
    for i in range(len(S_list)):
        if time.time() >= deadline:
            break
        
        p = S_list[i]
        S_without_p = [x for x in S_list if x != p]
        S_without_p_set, dist_used_trial = _build_dist_used(S_without_p)
        
        # Try adding random points q not in S
        candidates = [q for q in range(NUM_POINTS) if q not in S_set]
        random.shuffle(candidates)
        
        for q in candidates[:15]:
            if _can_add(q, S_without_p_set, dist_used_trial):
                S_new = S_without_p + [q]
                S_new_set, dist_used_new = _build_dist_used(S_new)
                if _is_maximal(S_new_set, dist_used_new):
                    return S_new, S_new_set, dist_used_new, True
    
    return S_list, S_set, dist_used, False

def _branch_and_bound_search(max_size, deadline, best_set_ref):
    """
    Branch and bound to find a maximal no-isosceles set of size <= max_size.
    Uses constraint propagation.
    """
    # We search for a maximal set with at most max_size points.
    # Strategy: pick points greedily but with backtracking.
    
    def _search(S_list, S_set, dist_used, remaining_candidates, depth):
        if time.time() >= deadline:
            return None
        
        # If we already have a valid maximal set, return it
        if len(S_list) >= max_size:
            if _is_maximal(S_set, dist_used):
                return S_list[:]
            return None
        
        # If no more candidates, check maximality
        if not remaining_candidates:
            if _is_maximal(S_set, dist_used):
                return S_list[:]
            return None
        
        # Pruning: if even adding all remaining can't reach maximality easily, 
        # but for now just continue
        
        # Try adding each candidate
        for idx, p in enumerate(remaining_candidates):
            if p in S_set:
                continue
            
            if not _can_add(p, S_set, dist_used):
                continue
            
            # Add p
            new_S_list = S_list + [p]
            new_S_set = S_set | {p}
            
            new_dist_used = {a: set(d) for a, d in dist_used.items()}
            for a in S_list:
                d = dist_sq[p, a]
                if a not in new_dist_used:
                    new_dist_used[a] = set()
                new_dist_used[a].add(d)
            if S_list:
                new_dist_used[p] = set(dist_sq[p, a] for a in S_list)
            else:
                new_dist_used[p] = set()
            
            new_remaining = remaining_candidates[:idx] + remaining_candidates[idx+1:]
            
            result = _search(new_S_list, new_S_set, new_dist_used, new_remaining, depth + 1)
            if result is not None:
                return result
        
        return None
    
    # Start search with empty set
    initial_candidates = list(range(NUM_POINTS))
    random.shuffle(initial_candidates)
    
    return _search([], set(), {}, initial_candidates, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_set = None
    best_size = NUM_POINTS + 1
    
    # Phase 1: Greedy construction with local search
    attempts = 0
    while time.time() < deadline - 60:  # Leave time for phase 2
        attempts += 1
        
        random_order = list(range(NUM_POINTS))
        random.shuffle(random_order)
        
        S_list, S_set, dist_used = _greedy_maximal(random_order, deadline)
        current_size = len(S_list)
        
        # Local search to minimize
        S_list, S_set, dist_used = _local_search_minimize(S_list, S_set, dist_used, deadline)
        current_size = len(S_list)
        
        # Try swaps
        improved = True
        while improved and time.time() < deadline - 60:
            S_list, S_set, dist_used, improved = _try_swap_improve(S_list, S_set, dist_used, deadline)
            current_size = len(S_list)
        
        if current_size < best_size:
            best_size = current_size
            best_set = S_list.copy()
        
        if best_size <= 10:
            break
    
    # Phase 2: Try branch and bound to find smaller sets
    if time.time() < deadline - 30:
        # Try to find a set of size best_size - 1
        target_size = best_size - 1
        if target_size >= 3:
            result = _branch_and_bound_search(target_size, deadline, best_set)
            if result is not None and len(result) < best_size:
                best_size = len(result)
                best_set = result
    
    if best_set is None:
        # Fallback: simple greedy
        random_order = list(range(NUM_POINTS))
        random.shuffle(random_order)
        best_set, _, _ = _greedy_maximal(random_order, deadline)
    
    return [GRID_POINTS[i] for i in best_set]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
