import itertools
import time
import random
import sys
from collections import defaultdict

N = 16
_DEADLINE = time.time() + 280

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _precompute(all_points):
    """
    Precompute:
    1. dist_matrix[i][j] = squared distance
    2. For each x, dist_to_points[x][d] = set of points y with d(x,y)=d
    """
    n = len(all_points)
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _d2(all_points[i], all_points[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    
    # For each x and each distance d, which points have that distance from x
    dist_to_points = [defaultdict(set) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                d = dist_matrix[i][j]
                dist_to_points[i][d].add(j)
    
    return dist_matrix, dist_to_points

def _can_add_fast(x, chosen_set, dist_matrix):
    """
    Check if x can be added to chosen_set.
    """
    if not chosen_set:
        return True
    
    # Check 1: All distances from x to points in chosen must be distinct
    seen_distances = set()
    for p in chosen_set:
        d = dist_matrix[x][p]
        if d in seen_distances:
            return False
        seen_distances.add(d)
    
    # Check 2: for each p in chosen, d(x,p) must not equal d(p,q) for any q in chosen, q≠p
    for p in chosen_set:
        d_xp = dist_matrix[x][p]
        for q in chosen_set:
            if q == p:
                continue
            if dist_matrix[p][q] == d_xp:
                return False
    
    return True

def _is_maximal_fast(chosen_set, all_indices, dist_matrix):
    """
    Check if chosen_set is maximal.
    """
    for x in all_indices:
        if x in chosen_set:
            continue
        if _can_add_fast(x, chosen_set, dist_matrix):
            return False
    return True

def _construct_greedy_smart(all_indices, dist_matrix, dist_to_points, rng, strategy='random'):
    """
    Greedy construction with various ordering strategies.
    """
    order = list(all_indices)
    
    if strategy == 'random':
        rng.shuffle(order)
    elif strategy == 'conflict_order':
        # Sort by number of potential conflicts (fewer conflicts first)
        def conflict_count(p):
            # Count how many other points have the same distance from p as some other point
            dists = defaultdict(int)
            for q in all_indices:
                if q != p:
                    d = dist_matrix[p][q]
                    dists[d] += 1
            return sum(1 for d, cnt in dists.items() if cnt > 1)
        order.sort(key=conflict_count)
    elif strategy == 'reverse_conflict':
        def conflict_count(p):
            dists = defaultdict(int)
            for q in all_indices:
                if q != p:
                    d = dist_matrix[p][q]
                    dists[d] += 1
            return sum(1 for d, cnt in dists.items() if cnt > 1)
        order.sort(key=conflict_count, reverse=True)
    elif strategy == 'center_first':
        # Prefer points near center
        center = (N/2, N/2)
        order.sort(key=lambda p: (p % N - center[0])**2 + (p // N - center[1])**2)
    elif strategy == 'corner_first':
        # Prefer points near corners
        center = (N/2, N/2)
        order.sort(key=lambda p: -((p % N - center[0])**2 + (p // N - center[1])**2))
    
    chosen = []
    chosen_set = set()
    for p in order:
        if _can_add_fast(p, chosen_set, dist_matrix):
            chosen.append(p)
            chosen_set.add(p)
    return chosen, chosen_set

def _local_search_reduce(chosen, chosen_set, all_indices, dist_matrix, rng, deadline):
    """
    Local search: try to remove 2 points and add 1 to reduce size.
    """
    s = list(chosen)
    s_set = set(s)
    n = len(s)
    improved = True
    max_iterations = 50
    
    while improved and time.time() < deadline and max_iterations > 0:
        improved = False
        max_iterations -= 1
        
        # Try all pairs to remove, but limit the number of attempts
        pairs_to_try = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs_to_try.append((i, j))
        
        rng.shuffle(pairs_to_try)
        
        # Try at most 20 pairs per iteration
        for i, j in pairs_to_try[:20]:
            if time.time() > deadline:
                return s, s_set
            
            p, q = s[i], s[j]
            base_set = s_set - {p, q}
            
            # Try to add a point r
            candidates = [r for r in all_indices if r not in base_set]
            rng.shuffle(candidates)
            
            for r in candidates[:30]:  # Limit candidates tried
                if not _can_add_fast(r, base_set, dist_matrix):
                    continue
                
                candidate_set = base_set | {r}
                if _is_maximal_fast(candidate_set, all_indices, dist_matrix):
                    s = list(candidate_set)
                    s_set = candidate_set
                    n = len(s)
                    improved = True
                    break
            if improved:
                break
                
    return s, s_set

def _local_search_swap(chosen, chosen_set, all_indices, dist_matrix, rng, deadline):
    """
    Local search: try to swap one point for another.
    """
    s = list(chosen)
    s_set = set(s)
    n = len(s)
    attempts = 0
    max_attempts = 30
    
    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        moved = False
        i = rng.randrange(n)
        p = s[i]
        base_set = s_set - {p}
        
        for _ in range(15):
            r = rng.choice(all_indices)
            if r in base_set:
                continue
            
            if not _can_add_fast(r, base_set, dist_matrix):
                continue
            
            candidate_set = base_set | {r}
            if _is_maximal_fast(candidate_set, all_indices, dist_matrix):
                s = list(candidate_set)
                s_set = candidate_set
                n = len(s)
                moved = True
                break
        
        if not moved:
            break
            
    return s, s_set

def _local_search_remove_one(chosen, chosen_set, all_indices, dist_matrix, rng, deadline):
    """
    Local search: try to remove one point and see if the set is still maximal.
    If not maximal, try to add a different point.
    """
    s = list(chosen)
    s_set = set(s)
    n = len(s)
    improved = True
    max_iterations = 20
    
    while improved and time.time() < deadline and max_iterations > 0:
        improved = False
        max_iterations -= 1
        
        # Try removing each point
        for i in range(n):
            if time.time() > deadline:
                return s, s_set
            
            p = s[i]
            base_set = s_set - {p}
            
            # Check if base_set is maximal
            if _is_maximal_fast(base_set, all_indices, dist_matrix):
                # It's maximal without p, so we can remove p
                s = list(base_set)
                s_set = base_set
                n = len(s)
                improved = True
                break
                
    return s, s_set

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    all_indices = list(range(len(all_points)))
    dist_matrix, dist_to_points = _precompute(all_points)
    
    best = None
    best_size = float('inf')
    rng = random.Random(42)
    
    strategies = ['random', 'conflict_order', 'reverse_conflict', 'center_first', 'corner_first']
    
    # First pass: try all strategies once
    for strategy in strategies:
        if time.time() >= _DEADLINE:
            break
        
        chosen, chosen_set = _construct_greedy_smart(all_indices, dist_matrix, dist_to_points, rng, strategy)
        
        # Local search reduce
        pruned, pruned_set = _local_search_reduce(chosen, chosen_set, all_indices, dist_matrix, rng, _DEADLINE)
        
        # Try remove one
        pruned, pruned_set = _local_search_remove_one(pruned, pruned_set, all_indices, dist_matrix, rng, _DEADLINE)
        
        # If no improvement, try swap
        if len(pruned) == len(chosen):
            pruned, pruned_set = _local_search_swap(pruned, pruned_set, all_indices, dist_matrix, rng, _DEADLINE)
        
        # Update best
        if len(pruned) < best_size:
            best_size = len(pruned)
            best = pruned
    
    # Second pass: more random restarts if time permits
    restart_count = 0
    while time.time() < _DEADLINE - 30 and restart_count < 20:
        restart_count += 1
        
        # Random strategy with different seed
        chosen, chosen_set = _construct_greedy_smart(all_indices, dist_matrix, dist_to_points, rng, 'random')
        
        # Local search reduce
        pruned, pruned_set = _local_search_reduce(chosen, chosen_set, all_indices, dist_matrix, rng, _DEADLINE)
        
        # Try remove one
        pruned, pruned_set = _local_search_remove_one(pruned, pruned_set, all_indices, dist_matrix, rng, _DEADLINE)
        
        # If no improvement, try swap
        if len(pruned) == len(chosen):
            pruned, pruned_set = _local_search_swap(pruned, pruned_set, all_indices, dist_matrix, rng, _DEADLINE)
        
        # Update best
        if len(pruned) < best_size:
            best_size = len(pruned)
            best = pruned
    
    if best is None:
        best, _ = _construct_greedy_smart(all_indices, dist_matrix, dist_to_points, rng, 'random')
        
    return [all_points[i] for i in best]
