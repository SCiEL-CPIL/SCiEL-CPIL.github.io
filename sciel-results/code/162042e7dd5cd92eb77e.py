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
    n = len(all_points)
    # dist_matrix[i][j] = squared distance
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _d2(all_points[i], all_points[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    
    # dist_lists[i] = {d: [j1, j2, ...]}
    dist_lists = [dict() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = dist_matrix[i][j]
            if d not in dist_lists[i]:
                dist_lists[i][d] = []
            dist_lists[i][d].append(j)
            
    # iso_pairs[x][a] = list of p such that {p, x, a} is isosceles
    # This helps in updating block_count
    # Actually, we can compute block_count updates on the fly using dist_lists
    # But precomputing "isosceles partners" might be useful.
    # For a pair (x, a), p is isosceles if:
    # 1. d(p,x) == d(p,a)
    # 2. d(p,x) == d(x,a)
    # 3. d(p,a) == d(x,a)
    
    return dist_matrix, dist_lists

def _can_add(x, chosen_set, chosen_list, dist_matrix, dist_lists):
    """
    Check if point x can be added to the current set.
    """
    if not chosen_list:
        return True
        
    # 1. Distances from x to all p in chosen must be distinct
    seen_dists = set()
    for p in chosen_list:
        d = dist_matrix[x][p]
        if d in seen_dists:
            return False
        seen_dists.add(d)
        
    # 2. For each p in chosen, d(x, p) must not be a distance already used by p to other points in chosen
    for p in chosen_list:
        d_xp = dist_matrix[x][p]
        # Check if any other q in chosen has dist_matrix[p][q] == d_xp
        if d_xp in dist_lists[p]:
            for q in dist_lists[p][d_xp]:
                if q in chosen_set:
                    return False
    return True

def _is_maximal(chosen_set, chosen_list, all_indices, dist_matrix, dist_lists):
    """
    Check if chosen is maximal.
    """
    for x in all_indices:
        if x in chosen_set:
            continue
        if _can_add(x, chosen_set, chosen_list, dist_matrix, dist_lists):
            return False
    return True

def _construct_greedy(all_indices, dist_matrix, dist_lists, rng):
    order = list(all_indices)
    rng.shuffle(order)
    chosen = []
    chosen_set = set()
    for p in order:
        if _can_add(p, chosen_set, chosen, dist_matrix, dist_lists):
            chosen.append(p)
            chosen_set.add(p)
    return chosen, chosen_set

def _local_search_reduce(chosen, chosen_set, all_indices, dist_matrix, dist_lists, rng, deadline):
    """
    Local search to reduce size.
    Moves:
    1. Swap (remove 1, add 1) - to escape local minima
    2. Reduce (remove 2, add 1) - to decrease size
    """
    s = list(chosen)
    s_set = set(s)
    n = len(s)
    improved = True
    no_improve_count = 0
    max_no_improve = 50
    
    while improved and time.time() < deadline:
        improved = False
        no_improve_count += 1
        if no_improve_count > max_no_improve:
            break
            
        # Try Swaps first to move in the search space
        # Remove i, Add j
        for i in range(n):
            if time.time() > deadline: break
            p_remove = s[i]
            base = [x for k, x in enumerate(s) if k != i]
            base_set = set(base)
            
            # Find a point j not in base that can be added
            # To be efficient, we can just iterate all j not in base
            for j in all_indices:
                if j in base_set:
                    continue
                if not _can_add(j, base_set, base, dist_matrix, dist_lists):
                    continue
                
                candidate_set = base_set | {j}
                candidate_list = base + [j]
                
                # Check maximality
                if _is_maximal(candidate_set, candidate_list, all_indices, dist_matrix, dist_lists):
                    # Accept swap (size unchanged, but structure changed)
                    s = candidate_list
                    s_set = candidate_set
                    n = len(s)
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # Try Reductions (remove 2, add 1)
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > deadline: break
                
                p1, p2 = s[i], s[j]
                base = [x for k, x in enumerate(s) if k != i and k != j]
                base_set = set(base)
                
                for r in all_indices:
                    if r in base_set:
                        continue
                    
                    if not _can_add(r, base_set, base, dist_matrix, dist_lists):
                        continue
                    
                    candidate_set = base_set | {r}
                    candidate_list = base + [r]
                    
                    if _is_maximal(candidate_set, candidate_list, all_indices, dist_matrix, dist_lists):
                        s = candidate_list
                        s_set = candidate_set
                        n = len(s)
                        improved = True
                        no_improve_count = 0
                        break
                if improved:
                    break
            if improved:
                break
                
    return s, s_set

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    all_indices = list(range(len(all_points)))
    dist_matrix, dist_lists = _precompute(all_points)
    
    best = None
    best_size = float('inf')
    rng = random.Random(42)
    
    # Run multiple restarts
    while time.time() < _DEADLINE:
        # 1. Greedy Construction
        chosen, chosen_set = _construct_greedy(all_indices, dist_matrix, dist_lists, rng)
        
        # 2. Local Search
        pruned, pruned_set = _local_search_reduce(chosen, chosen_set, all_indices, dist_matrix, dist_lists, rng, _DEADLINE)
        
        # 3. Update best
        if len(pruned) < best_size:
            best_size = len(pruned)
            best = pruned
            
    if best is None:
        best, _ = _construct_greedy(all_indices, dist_matrix, dist_lists, rng)
        
    return [all_points[i] for i in best]

# EVOLVE-BLOCK-END
