import itertools
import time
import random
import sys

N = 16
_DEADLINE = time.time() + 280

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _precompute_distances(all_points):
    n = len(all_points)
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _d2(all_points[i], all_points[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    return dist_matrix

def _can_add(x, chosen, dist_matrix):
    """
    Check if point x can be added to chosen.
    chosen is a list of indices.
    Condition:
    1. Distances from x to all p in chosen must be distinct.
    2. For each p in chosen, d(x, p) must not be a distance already used by p to other points in chosen.
    """
    if not chosen:
        return True
        
    # 1. Distances from x to all p in chosen must be distinct
    distances_from_x = set()
    for p in chosen:
        d = dist_matrix[x][p]
        if d in distances_from_x:
            return False
        distances_from_x.add(d)
        
    # 2. For each p in chosen, d(x, p) must not be a distance already used by p to other points in chosen
    for p in chosen:
        d_xp = dist_matrix[x][p]
        # Check if any other q in chosen has dist_matrix[p][q] == d_xp
        for q in chosen:
            if q == p:
                continue
            if dist_matrix[p][q] == d_xp:
                return False
    return True

def _is_maximal(chosen, all_indices, dist_matrix):
    """
    Check if chosen is maximal.
    i.e., no point in all_indices \ chosen can be added.
    """
    chosen_set = set(chosen)
    for x in all_indices:
        if x in chosen_set:
            continue
        if _can_add(x, chosen, dist_matrix):
            return False
    return True

def _construct_greedy(all_indices, dist_matrix, rng):
    """
    Construct a maximal set by adding points in random order.
    """
    order = list(all_indices)
    rng.shuffle(order)
    chosen = []
    for p in order:
        if _can_add(p, chosen, dist_matrix):
            chosen.append(p)
    return chosen

def _remove_and_remaximize(chosen, all_indices, dist_matrix, rng, deadline):
    """
    Remove a random point from chosen, then greedily re-add points to restore maximality.
    This can sometimes lead to a smaller maximal set.
    """
    s = list(chosen)
    if not s:
        return s
    
    n = len(s)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Try removing each point in the set
        for i in range(n):
            if time.time() > deadline:
                break
            
            p = s[i]
            base = [x for k, x in enumerate(s) if k != i]
            base_set = set(base)
            
            # Greedily add points to restore maximality
            # Try all points not in base in random order
            candidates = [x for x in all_indices if x not in base_set]
            rng.shuffle(candidates)
            
            new_s = list(base)
            for r in candidates:
                if _can_add(r, new_s, dist_matrix):
                    new_s.append(r)
            
            # Check if new_s is maximal (it should be by construction, but verify)
            # Actually, the greedy process above doesn't guarantee maximality.
            # We need to ensure that after adding, no more points can be added.
            # Let's do a final pass to ensure maximality
            new_s_set = set(new_s)
            final_candidates = [x for x in all_indices if x not in new_s_set]
            for r in final_candidates:
                if _can_add(r, new_s, dist_matrix):
                    new_s.append(r)
            
            if len(new_s) < n:
                s = new_s
                n = len(s)
                improved = True
                break
    
    return s

def _local_search_reduce(chosen, all_indices, dist_matrix, rng, deadline):
    """
    Try to reduce the size of the maximal set.
    Move: Remove p, q and add r.
    If (S \ {p, q}) U {r} is maximal, accept.
    """
    s = list(chosen)
    n = len(s)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Try all pairs (p, q) to remove
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                p, q = s[i], s[j]
                base = [x for k, x in enumerate(s) if k != i and k != j]
                base_set = set(base)
                
                # Try to add a point r not in base
                for r in all_indices:
                    if r in base_set:
                        continue
                    
                    candidate = base + [r]
                    if not _can_add(r, base, dist_matrix):
                        continue
                    
                    # Check maximality
                    if _is_maximal(candidate, all_indices, dist_matrix):
                        s = candidate
                        n = len(s)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
                
    return s

def _local_search_swap(chosen, all_indices, dist_matrix, rng, deadline):
    """
    Try to swap points to escape local minima.
    Move: Remove p, add q.
    If (S \ {p}) U {q} is maximal, accept.
    """
    s = list(chosen)
    n = len(s)
    attempts = 0
    max_attempts = 100
    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        moved = False
        i = rng.randrange(n)
        p = s[i]
        base = [x for k, x in enumerate(s) if k != i]
        base_set = set(base)
        
        for _ in range(10):
            r = rng.choice(all_indices)
            if r in base_set:
                continue
            
            if not _can_add(r, base, dist_matrix):
                continue
            
            candidate = base + [r]
            if _is_maximal(candidate, all_indices, dist_matrix):
                s = candidate
                n = len(s)
                moved = True
                break
        
        if not moved:
            break
            
    return s

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    all_indices = list(range(len(all_points)))
    dist_matrix = _precompute_distances(all_points)
    
    best = None
    best_size = float('inf')
    rng = random.Random(42)
    
    while time.time() < _DEADLINE:
        # 1. Greedy Construction
        chosen = _construct_greedy(all_indices, dist_matrix, rng)
        
        # 2. Remove-and-re-maximize (new step)
        remaximized = _remove_and_remaximize(chosen, all_indices, dist_matrix, rng, _DEADLINE)
        
        # 3. Local Search: Try to reduce size
        pruned = _local_search_reduce(remaximized, all_indices, dist_matrix, rng, _DEADLINE)
        
        # 4. If size didn't change, try swaps to escape
        if len(pruned) == len(remaximized):
            pruned = _local_search_swap(pruned, all_indices, dist_matrix, rng, _DEADLINE)
            
        # 5. Update best
        if len(pruned) < best_size:
            best_size = len(pruned)
            best = pruned
            
    if best is None:
        best = _construct_greedy(all_indices, dist_matrix, rng)
        
    return [all_points[i] for i in best]

# EVOLVE-BLOCK-END
