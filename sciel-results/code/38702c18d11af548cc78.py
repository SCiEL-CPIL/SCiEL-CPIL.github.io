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

def _check_valid(S_list):
    """Check if S_list is a valid no-isosceles set."""
    S_set = set(S_list)
    if not S_list:
        return True
    for a in S_list:
        seen = set()
        for b in S_list:
            if a == b:
                continue
            d = dist_sq[a, b]
            if d in seen:
                return False
            seen.add(d)
    return True

def _is_maximal(S_set):
    """Check if S_set is maximal: no point outside can be added."""
    for q in range(NUM_POINTS):
        if q in S_set:
            continue
        # Try adding q
        seen = set()
        ok = True
        for a in S_set:
            d = dist_sq[q, a]
            if d in seen:
                ok = False
                break
            seen.add(d)
            # Check if d is already in a's distances to other S points
            for b in S_set:
                if b == a or b == q:
                    continue
                if dist_sq[a, b] == d:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return False
    return True

def _can_add(p, S_set, dist_used):
    """Check if point p can be added to S_set with dist_used tracking."""
    if not S_set:
        return True
    seen_dists = set()
    for a in S_set:
        d = dist_sq[p, a]
        if d in seen_dists:
            return False
        seen_dists.add(d)
        if d in dist_used[a]:
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

def _greedy_maximal(order):
    """Greedily construct a maximal no-isosceles set."""
    S_list = []
    S_set = set()
    dist_used = {}
    
    for p in order:
        if p in S_set:
            continue
        
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
    """Try to minimize by removing points."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        shuffled = S_list[:]
        random.shuffle(shuffled)
        
        for p in shuffled:
            if p not in S_set:
                continue
            
            S_new = [x for x in S_list if x != p]
            S_new_set, dist_used_new = _build_dist_used(S_new)
            
            if _is_maximal(S_new_set):
                S_list = S_new
                S_set = S_new_set
                dist_used = dist_used_new
                improved = True
                break
    
    return S_list, S_set, dist_used

def _try_swap_improve(S_list, S_set, dist_used, deadline):
    """Try to improve by swapping one point for another."""
    for i in range(len(S_list)):
        if time.time() >= deadline:
            break
        
        p = S_list[i]
        S_without_p = [x for x in S_list if x != p]
        S_without_p_set, dist_used_trial = _build_dist_used(S_without_p)
        
        candidates = [q for q in range(NUM_POINTS) if q not in S_set]
        random.shuffle(candidates)
        
        for q in candidates[:20]:
            if _can_add(q, S_without_p_set, dist_used_trial):
                S_new = S_without_p + [q]
                S_new_set, dist_used_new = _build_dist_used(S_new)
                if _is_maximal(S_new_set):
                    return S_new, S_new_set, dist_used_new, True
    
    return S_list, S_set, dist_used, False

def _simulated_annealing(deadline, best_size):
    """
    Use simulated annealing on the set representation.
    State: a maximal no-isosceles set.
    Moves: remove a point (if still maximal), add a point (if valid and maximal),
           swap a point.
    Objective: minimize |S|.
    """
    # Start with a random greedy set
    random_order = list(range(NUM_POINTS))
    random.shuffle(random_order)
    S_list, S_set, dist_used = _greedy_maximal(random_order)
    
    current_size = len(S_list)
    best_size_sa = current_size
    best_set_sa = S_list[:]
    
    # Temperature schedule
    t0 = 50.0
    t_final = 0.1
    start_time = time.time()
    total_time = deadline - start_time
    if total_time <= 0:
        return best_set_sa, best_size_sa
    
    current_temp = t0
    
    while time.time() < deadline:
        elapsed = time.time() - start_time
        progress = min(elapsed / total_time, 1.0)
        current_temp = t0 * (t_final / t0) ** progress
        
        if current_temp < 0.05:
            break
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.4 and len(S_list) > 1:
            # Try removing a random point
            p = random.choice(S_list)
            S_new = [x for x in S_list if x != p]
            S_new_set, dist_used_new = _build_dist_used(S_new)
            
            if _is_maximal(S_new_set):
                new_size = len(S_new)
                delta = new_size - len(S_list)  # negative
                if delta < 0 or random.random() < np.exp(-delta / max(current_temp, 1e-10)):
                    S_list = S_new
                    S_set = S_new_set
                    dist_used = dist_used_new
                    if new_size < best_size_sa:
                        best_size_sa = new_size
                        best_set_sa = S_list[:]
                        
        elif move_type < 0.7:
            # Try adding a random point
            candidates = [q for q in range(NUM_POINTS) if q not in S_set]
            if candidates:
                q = random.choice(candidates)
                if _can_add(q, S_set, dist_used):
                    S_new = S_list + [q]
                    S_new_set, dist_used_new = _build_dist_used(S_new)
                    if _is_maximal(S_new_set):
                        new_size = len(S_new)
                        delta = new_size - len(S_list)  # positive
                        if random.random() < np.exp(-delta / max(current_temp, 1e-10)):
                            S_list = S_new
                            S_set = S_new_set
                            dist_used = dist_used_new
                            if new_size < best_size_sa:
                                best_size_sa = new_size
                                best_set_sa = S_list[:]
                            
        else:
            # Try swapping
            if len(S_list) > 0:
                p = random.choice(S_list)
                S_without_p = [x for x in S_list if x != p]
                S_without_p_set, dist_used_trial = _build_dist_used(S_without_p)
                
                candidates = [q for q in range(NUM_POINTS) if q not in S_set]
                if candidates:
                    q = random.choice(candidates)
                    if _can_add(q, S_without_p_set, dist_used_trial):
                        S_new = S_without_p + [q]
                        S_new_set, dist_used_new = _build_dist_used(S_new)
                        if _is_maximal(S_new_set):
                            new_size = len(S_new)
                            delta = new_size - len(S_list)  # 0
                            if random.random() < np.exp(-delta / max(current_temp, 1e-10)):
                                S_list = S_new
                                S_set = S_new_set
                                dist_used = dist_used_new
                                if new_size < best_size_sa:
                                    best_size_sa = new_size
                                    best_set_sa = S_list[:]
    
    return best_set_sa, best_size_sa

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_set = None
    best_size = NUM_POINTS + 1
    
    # Phase 1: Multiple greedy + local search attempts
    attempts = 0
    while time.time() < deadline - 120:
        attempts += 1
        
        random_order = list(range(NUM_POINTS))
        random.shuffle(random_order)
        
        S_list, S_set, dist_used = _greedy_maximal(random_order)
        
        # Local search to minimize
        S_list, S_set, dist_used = _local_search_minimize(S_list, S_set, dist_used, deadline - 60)
        
        # Try swaps
        improved = True
        while improved and time.time() < deadline - 60:
            S_list, S_set, dist_used, improved = _try_swap_improve(S_list, S_set, dist_used, deadline - 60)
        
        current_size = len(S_list)
        if current_size < best_size:
            best_size = current_size
            best_set = S_list.copy()
        
        if best_size <= 8:
            break
    
    # Phase 2: Simulated annealing
    if time.time() < deadline - 30:
        sa_set, sa_size = _simulated_annealing(deadline - 30, best_size)
        if sa_size < best_size:
            best_size = sa_size
            best_set = sa_set
    
    # Phase 3: More local search on best found
    if best_set is not None and time.time() < deadline - 10:
        S_list, S_set, dist_used = _build_dist_used(best_set)
        S_list, S_set, dist_used = _local_search_minimize(S_list, S_set, dist_used, deadline - 5)
        if len(S_list) < best_size:
            best_size = len(S_list)
            best_set = S_list
    
    if best_set is None:
        # Fallback: simple greedy
        random_order = list(range(NUM_POINTS))
        random.shuffle(random_order)
        best_set, _, _ = _greedy_maximal(random_order)
    
    return [GRID_POINTS[i] for i in best_set]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
