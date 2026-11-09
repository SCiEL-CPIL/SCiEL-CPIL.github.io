import itertools
import time
import random
import sys

N = 16
_DEADLINE = time.time() + 280

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _precompute_distances(all_points):
    """
    Precompute squared distances between all pairs.
    Also, for each point p, group other points by distance.
    dist_groups[p][d] = list of points q such that d2(p, q) == d
    """
    n = len(all_points)
    dist_matrix = [[0] * n for _ in range(n)]
    dist_groups = [dict() for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            d = _d2(all_points[i], all_points[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
            if d not in dist_groups[i]:
                dist_groups[i][d] = []
            dist_groups[i][d].append(j)
            if d not in dist_groups[j]:
                dist_groups[j][d] = []
            dist_groups[j][d].append(i)
            
    return dist_matrix, dist_groups

def _is_compatible(x, chosen, dist_matrix, dist_groups):
    """
    Check if point x can be added to chosen.
    chosen is a list of indices.
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
    # We need to know which distances are used by p in chosen.
    # Instead of recalculating, we can pass a structure, but for simplicity in this check:
    for p in chosen:
        d_xp = dist_matrix[x][p]
        # Check if any other q in chosen has dist_matrix[p][q] == d_xp
        for q in chosen:
            if q == p:
                continue
            if dist_matrix[p][q] == d_xp:
                return False
    return True

def _is_maximal(chosen, all_indices, dist_matrix, dist_groups):
    """
    Check if chosen is maximal.
    i.e., no point in all_indices \ chosen can be added.
    """
    chosen_set = set(chosen)
    for x in all_indices:
        if x in chosen_set:
            continue
        if _is_compatible(x, chosen, dist_matrix, dist_groups):
            return False
    return True

def _construct_greedy(all_indices, dist_matrix, dist_groups, rng):
    """
    Construct a maximal set by adding points in random order.
    """
    order = list(all_indices)
    rng.shuffle(order)
    chosen = []
    for p in order:
        if _is_compatible(p, chosen, dist_matrix, dist_groups):
            chosen.append(p)
    return chosen

def _local_search(initial_set, all_indices, dist_matrix, dist_groups, rng, deadline):
    """
    Try to improve the set by swapping points.
    Goal: Minimize size.
    Since removing a point makes it non-maximal (the removed point can be re-added),
    we must swap.
    However, if we swap p with x, size stays same.
    How to reduce size?
    We need to find a set S' such that S' is maximal and |S'| < |S|.
    This requires removing k points and adding < k points.
    
    Let's try a move: Remove p.
    If S \ {p} is not maximal, find a point x that can be added to S \ {p}.
    If we add x, we get (S \ {p}) U {x}. Size is same.
    But maybe (S \ {p}) U {x} allows removing another point?
    
    Actually, there is a known technique for "minimum maximal" problems:
    Start with a maximal set.
    Randomly remove a point p.
    If the set is no longer maximal, find a minimal set of points to add to restore maximality.
    If the number of points added < 1 (impossible) or we can add 0 points (impossible as p can be added),
    then we cannot reduce size by just removing.
    
    Wait, if we remove p, p can be added. So we MUST add at least 1 point to make it maximal?
    No, if we add x, and x blocks p, then p cannot be added.
    So if we remove p and add x such that x conflicts with p, then p cannot be added.
    Then we check if any other point can be added.
    If no point can be added, the new set is maximal.
    Size is same.
    
    To reduce size, we need to remove k points and add m < k points.
    Example: Remove p1, p2. Add x.
    If x blocks both p1 and p2, and no other point can be added, then size reduced by 1.
    
    This is a complex search.
    Let's use Simulated Annealing on the "Maximal Sets" space.
    State: A maximal set S.
    Move: 
      1. Pick random p in S.
      2. Remove p.
      3. Find all points x that can be added to S \ {p}.
      4. Pick one x at random.
      5. Add x.
      6. Check if new set is maximal.
         If not, find more points to add?
         This gets complicated.
         
    Alternative:
    Use the greedy construction with different random seeds and pruning.
    The current program does this.
    Let's try to improve the pruning.
    
    Pruning:
    After constructing a maximal set, try to remove points.
    If removing p makes it non-maximal, we don't remove it.
    But maybe removing p and q together allows it?
    Try pairs (p, q). Remove both. If maximal, keep removed.
    This is O(N^2) checks.
    
    Let's implement a "pair removal" heuristic.
    """
    best_set = list(initial_set)
    best_size = len(best_set)
    
    # Try to remove single points (should fail for maximal sets, but just in case)
    # Actually, for a maximal set, removing any point p allows p to be added back.
    # So single removal always fails.
    
    # Try to remove pairs (p, q)
    # If S \ {p, q} is maximal, we reduce size by 2.
    # If S \ {p, q} is not maximal, maybe we can add 1 point to make it maximal?
    # If we can add 1 point x to (S \ {p, q}) to make it maximal, then size reduces by 1.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        s = list(best_set)
        n = len(s)
        
        # Try removing pairs
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                p, q = s[i], s[j]
                candidate = [x for k, x in enumerate(s) if k != i and k != j]
                
                # Check if candidate is maximal
                if _is_maximal(candidate, all_indices, dist_matrix, dist_groups):
                    best_set = candidate
                    best_size = len(candidate)
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # If no pair removal worked, try "remove 2, add 1"
        # This is expensive.
        # Let's just do a few random restarts of greedy instead.
        break
        
    return best_set

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    all_indices = list(range(len(all_points)))
    dist_matrix, dist_groups = _precompute_distances(all_points)
    
    best = None
    rng = random.Random(42)
    
    # Run multiple greedy constructions with different seeds
    for attempt in range(50):
        if time.time() > _DEADLINE:
            break
            
        # Vary the random seed or order
        chosen = _construct_greedy(all_indices, dist_matrix, dist_groups, rng)
        
        # Try to prune
        pruned = _local_search(chosen, all_indices, dist_matrix, dist_groups, rng, _DEADLINE)
        
        if best is None or len(pruned) < len(best):
            best = pruned
            
    if best is None:
        best = _construct_greedy(all_indices, dist_matrix, dist_groups, rng)
        
    return [all_points[i] for i in best]

# EVOLVE-BLOCK-END
