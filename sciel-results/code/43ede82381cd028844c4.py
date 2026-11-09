import itertools
import time
import random
import sys

N = 16
_DEADLINE = time.time() + 280

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _build_conflict_graph(all_points):
    """
    Build conflict graph where edge (i,j) exists if there exists k such that
    {i,j,k} forms an isosceles triangle.
    Two points conflict if they cannot both be in a no-isosceles set.
    
    Actually, the constraint is: no three points form an isosceles triangle.
    So i,j conflict if adding any k would create an isosceles with i,j,k.
    But that's not quite right either.
    
    Let me think more carefully. A set S is "no-isosceles" if no three points in S
    form an isosceles triangle. This is a 3-uniform hypergraph constraint, not a 2-uniform.
    
    For MIS formulation: We want to find the smallest maximal independent set in a hypergraph.
    But MIS is typically for graphs. Let me use a different approach.
    
    Actually, let's use the original approach but with better data structures.
    """
    n = len(all_points)
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _d2(all_points[i], all_points[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    
    # For each point, track distances to other points
    # dist_sets[i] = set of distances from i to all other points
    dist_sets = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_sets[i].add(dist_matrix[i][j])
    
    return dist_matrix, dist_sets

def _can_add_fast(x, chosen_set, dist_matrix, dist_sets):
    """
    Check if x can be added to chosen_set (a set of indices).
    Uses precomputed distance sets for faster checking.
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
    
    # Check 2: For each p in chosen, d(x,p) must not be in dist_sets[p] \ {d(x,q) for q in chosen, q!=p}
    # Actually, the constraint is: for each p in chosen, the distance d(x,p) must not equal
    # any distance d(p,q) where q is in chosen and q != p.
    # This is because if d(x,p) == d(p,q), then triangle (x,p,q) is isosceles.
    for p in chosen_set:
        d_xp = dist_matrix[x][p]
        # Check if any other point q in chosen has d(p,q) == d_xp
        for q in chosen_set:
            if q == p:
                continue
            if dist_matrix[p][q] == d_xp:
                return False
    
    return True

def _is_maximal_fast(chosen_set, all_indices, dist_matrix, dist_sets):
    """
    Check if chosen_set is maximal (no point can be added).
    """
    for x in all_indices:
        if x in chosen_set:
            continue
        if _can_add_fast(x, chosen_set, dist_matrix, dist_sets):
            return False
    return True

def _greedy_construct(all_indices, dist_matrix, dist_sets, rng):
    """
    Greedy construction with random ordering.
    """
    order = list(all_indices)
    rng.shuffle(order)
    chosen = []
    chosen_set = set()
    for p in order:
        if _can_add_fast(p, chosen_set, dist_matrix, dist_sets):
            chosen.append(p)
            chosen_set.add(p)
    return chosen, chosen_set

def _local_search_reduce_fast(chosen, chosen_set, all_indices, dist_matrix, dist_sets, rng, deadline):
    """
    Local search: try to remove 2 points and add 1 to reduce size.
    """
    s = list(chosen)
    s_set = set(s)
    n = len(s)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Try all pairs to remove
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                p, q = s[i], s[j]
                # Create base set without p and q
                base = [x for k, x in enumerate(s) if k != i and k != j]
                base_set = s_set - {p, q}
                
                # Try to add a point r
                for r in all_indices:
                    if r in base_set:
                        continue
                    
                    if not _can_add_fast(r, base_set, dist_matrix, dist_sets):
                        continue
                    
                    candidate_set = base_set | {r}
                    if _is_maximal_fast(candidate_set, all_indices, dist_matrix, dist_sets):
                        s = list(candidate_set)
                        s_set = candidate_set
                        n = len(s)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
                
    return s, s_set

def _local_search_swap_fast(chosen, chosen_set, all_indices, dist_matrix, dist_sets, rng, deadline):
    """
    Local search: try to swap one point for another.
    """
    s = list(chosen)
    s_set = set(s)
    n = len(s)
    attempts = 0
    max_attempts = 50
    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        moved = False
        i = rng.randrange(n)
        p = s[i]
        base_set = s_set - {p}
        
        for _ in range(20):
            r = rng.choice(all_indices)
            if r in base_set:
                continue
            
            if not _can_add_fast(r, base_set, dist_matrix, dist_sets):
                continue
            
            candidate_set = base_set | {r}
            if _is_maximal_fast(candidate_set, all_indices, dist_matrix, dist_sets):
                s = list(candidate_set)
                s_set = candidate_set
                n = len(s)
                moved = True
                break
        
        if not moved:
            break
            
    return s, s_set

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    all_indices = list(range(len(all_points)))
    dist_matrix, dist_sets = _build_conflict_graph(all_points)
    
    best = None
    best_size = float('inf')
    rng = random.Random(42)
    
    while time.time() < _DEADLINE:
        # Greedy construction
        chosen, chosen_set = _greedy_construct(all_indices, dist_matrix, dist_sets, rng)
        
        # Local search reduce
        pruned, pruned_set = _local_search_reduce_fast(chosen, chosen_set, all_indices, dist_matrix, dist_sets, rng, _DEADLINE)
        
        # If no improvement, try swap
        if len(pruned) == len(chosen):
            pruned, pruned_set = _local_search_swap_fast(pruned, pruned_set, all_indices, dist_matrix, dist_sets, rng, _DEADLINE)
        
        # Update best
        if len(pruned) < best_size:
            best_size = len(pruned)
            best = pruned
    
    if best is None:
        best, _ = _greedy_construct(all_indices, dist_matrix, dist_sets, rng)
        
    return [all_points[i] for i in best]

# EVOLVE-BLOCK-END
