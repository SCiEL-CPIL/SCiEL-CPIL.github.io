"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def _precompute_dist2():
    """Precompute squared distances between all pairs of grid points."""
    points = list(itertools.product(range(N), repeat=2))
    V = len(points)
    dist2 = np.zeros((V, V), dtype=np.int64)
    for i in range(V):
        xi, yi = points[i]
        for j in range(i + 1, V):
            xj, yj = points[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dist2[i, j] = d
            dist2[j, i] = d
    return points, dist2

# Precompute at module load time to save time during construct()
POINTS, DIST2 = _precompute_dist2()
V = len(POINTS)

def _is_valid_no_isosceles(selected_indices):
    """Check if the set of indices forms a no-isosceles set."""
    S = list(selected_indices)
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                d1 = DIST2[S[i], S[j]]
                d2 = DIST2[S[i], S[k]]
                d3 = DIST2[S[j], S[k]]
                # Isosceles if any two sides are equal
                if d1 == d2 or d1 == d3 or d2 == d3:
                    return False
    return True

def _get_unblocked_points(selected_indices):
    """
    Return a list of points NOT in selected_indices that can be added 
    without creating an isosceles triangle.
    These are the points that would make the set non-maximal if added.
    """
    S = list(selected_indices)
    unblocked = []
    for p in range(V):
        if p in selected_indices:
            continue
        # Check if adding p creates an isosceles triangle with any pair in S
        # Or rather, check if S U {p} is a valid no-isosceles set.
        # We can do this quickly by checking distances from p to S.
        # If S is no-isosceles, then S U {p} is no-isosceles iff:
        # 1. All d(p, s) are distinct for s in S.
        # 2. d(p, s) != d(s, s') for any s, s' in S.
        
        dists_from_p = set()
        valid = True
        for s in S:
            d_ps = DIST2[p, s]
            if d_ps in dists_from_p:
                valid = False
                break
            dists_from_p.add(d_ps)
        
        if not valid:
            continue
            
        # Check condition 2: d(p, s) != d(s, s')
        # We can precompute or just check. Since |S| is small in our search, this is fine.
        # Actually, we can be smarter. 
        # For each s in S, d(p, s) must not be equal to any distance between s and another point in S.
        for s in S:
            d_ps = DIST2[p, s]
            for s2 in S:
                if s == s2:
                    continue
                if d_ps == DIST2[s, s2]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            unblocked.append(p)
            
    return unblocked

def _is_maximal(selected_indices):
    """Check if the set is maximal (no further points can be added)."""
    return len(_get_unblocked_points(selected_indices)) == 0

def _greedy_construct():
    """Construct a maximal no-isosceles set using greedy row-major order."""
    selected = set()
    for i in range(V):
        if _is_valid_no_isosceles(selected | {i}):
            selected.add(i)
    # Ensure it's maximal. Greedy row-major usually produces a maximal set 
    # if we iterate through all points, but let's verify.
    # Actually, the standard greedy "add if valid" produces a maximal independent set 
    # in the conflict graph, which corresponds to a maximal no-isosceles set.
    return list(selected)

def _try_remove_point(selected_set, idx):
    """
    Try to remove point idx from selected_set.
    Returns the new set if it remains maximal and no-isosceles, else None.
    """
    new_set = set(selected_set)
    new_set.remove(idx)
    
    # 1. No-isosceles is automatically preserved by removal.
    # 2. Check maximality.
    # A set is maximal if every point outside it forms an isosceles triangle with some pair inside.
    # If we remove idx, some points outside might become "unblocked".
    # Specifically, points p that were blocked ONLY by triangles involving idx might now be addable.
    
    # Efficient check:
    # For every p not in new_set, check if new_set U {p} is no-isosceles.
    # If any such p exists, new_set is not maximal.
    
    # Optimization: 
    # If p was blocked in selected_set, it means selected_set U {p} had an isosceles triangle.
    # If that triangle did not involve idx, it still exists in new_set U {p}.
    # So we only need to check p's that were blocked by triangles involving idx.
    
    # However, implementing that optimization is complex. 
    # Given V=256 and |S| ~ 20-30, checking all p in [0, V) is O(V * |S|^2).
    # 256 * 30^2 = 230k ops. This is fast in Python if optimized, or use the helper.
    
    # Let's use a helper that checks if a specific p can be added to new_set.
    def _can_add(p, S_set):
        S_list = list(S_set)
        dists_from_p = set()
        for s in S_list:
            d_ps = DIST2[p, s]
            if d_ps in dists_from_p:
                return False
            dists_from_p.add(d_ps)
        for s in S_list:
            d_ps = DIST2[p, s]
            for s2 in S_list:
                if s == s2:
                    continue
                if d_ps == DIST2[s, s2]:
                    return False
        return True

    for p in range(V):
        if p in new_set:
            continue
        if _can_add(p, new_set):
            return None # Not maximal
            
    return new_set

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_set = None
    best_size = V
    
    # Phase 1: Generate several diverse initial maximal sets
    # We can use different random seeds or orderings.
    # 1. Row-major greedy
    # 2. Random order greedy
    # 3. Column-major greedy
    
    initial_sets = []
    
    # Row-major
    s1 = _greedy_construct()
    initial_sets.append(s1)
    
    # Random order greedy (a few times)
    for _ in range(5):
        order = list(range(V))
        random.shuffle(order)
        selected = set()
        for idx in order:
            if _is_valid_no_isosceles(selected | {idx}):
                selected.add(idx)
        initial_sets.append(list(selected))
        
    # Phase 2: Local Search via Removal
    # For each initial set, try to remove points.
    # We can do this iteratively: 
    # While there is a removable point, remove it.
    # To escape local minima (where no single point is removable, but two are),
    # we can try "swaps" or random restarts within the local search.
    # But first, let's try pure greedy removal.
    
    for init_set in initial_sets:
        if time.time() > deadline:
            break
            
        current_set = set(init_set)
        
        # Greedy removal
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try to remove each point in random order to avoid bias
            candidates = list(current_set)
            random.shuffle(candidates)
            
            for idx in candidates:
                if time.time() > deadline:
                    break
                new_set = _try_remove_point(current_set, idx)
                if new_set is not None:
                    current_set = new_set
                    changed = True
                    break # Restart the loop with the new set
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
            # If we get very small, we might want to stop
            if best_size <= 10:
                break

    # Phase 3: Advanced Local Search (Swaps/Re-adds)
    # If greedy removal gets stuck, try to "shake" the set.
    # Pick a random point in the set, remove it, then try to re-add points 
    # in a random order to restore maximality, hoping for a smaller size.
    # This is essentially a randomized restart from a mutated state.
    
    if best_set is not None:
        for restart in range(100):
            if time.time() > deadline:
                break
                
            current_set = set(best_set)
            
            # Randomly remove a few points to break the local minimum structure
            # Then re-maximize by adding random points?
            # No, re-maximizing by adding points will likely grow it.
            # Instead, let's try "Swap":
            # Remove p from S. If S\{p} is not maximal, find a point q not in S\{p} 
            # that can be added. If S\{p} U {q} is valid and maximal, and size is same, 
            # we haven't improved. We need size to decrease.
            
            # Actually, the "Removal" strategy is the key. 
            # If we are stuck, it means for every p in S, removing p makes it non-maximal.
            # This implies that for every p, there is some q not in S such that 
            # S\{p} U {q} is valid.
            # If we could swap p for q and then remove another point, we might shrink.
            
            # Let's try a simple heuristic: 
            # Try all pairs of removals? Too expensive.
            
            # Let's stick to the greedy removal with random restarts on the initial sets.
            # The diversity of initial sets is the main driver.
            
            # Let's generate more diverse initial sets if we have time.
            order = list(range(V))
            random.shuffle(order)
            selected = set()
            for idx in order:
                if _is_valid_no_isosceles(selected | {idx}):
                    selected.add(idx)
                    
            # Apply greedy removal to this new set
            changed = True
            while changed and time.time() < deadline:
                changed = False
                candidates = list(selected)
                random.shuffle(candidates)
                for idx in candidates:
                    if time.time() > deadline:
                        break
                    new_set = _try_remove_point(selected, idx)
                    if new_set is not None:
                        selected = new_set
                        changed = True
                        break
                        
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = list(selected)
                if best_size <= 10:
                    break

    if best_set is None:
        # Fallback to the first greedy set
        return [POINTS[i] for i in _greedy_construct()]
        
    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-END
