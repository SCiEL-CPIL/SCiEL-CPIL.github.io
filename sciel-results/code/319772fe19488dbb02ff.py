import time
import random
import math
from collections import defaultdict

N = 64

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin before 330s kill
    
    num_points = N * N
    coords = [(i // N, i % N) for i in range(num_points)]
    
    # Precompute squared distances using numpy
    try:
        import numpy as np
        xs = np.array([c[0] for c in coords], dtype=np.int32)
        ys = np.array([c[1] for c in coords], dtype=np.int32)
        dist2 = (xs[:, None] - xs[None, :])**2 + (ys[:, None] - ys[None, :])**2
        dist2_flat = dist2.ravel()
    except ImportError:
        dist2 = [[0]*num_points for _ in range(num_points)]
        for i in range(num_points):
            xi, yi = coords[i]
            for j in range(num_points):
                xj, yj = coords[j]
                dx = xi - xj
                dy = yi - yj
                dist2[i][j] = dx*dx + dy*dy
        dist2_flat = None
    
    def get_dist(i, j):
        if dist2_flat is not None:
            return int(dist2_flat[i * num_points + j])
        return dist2[i][j]
    
    def validate_set(point_set):
        """Check if a set of points has no isosceles triangles."""
        if len(point_set) <= 2:
            return True
        for p in point_set:
            dists = set()
            for q in point_set:
                if p == q:
                    continue
                d = get_dist(p, q)
                if d in dists:
                    return False
                dists.add(d)
        return True
    
    def greedy_construction(order, time_budget=60.0):
        """
        Greedily add points in the given order if they don't create an isosceles triangle.
        """
        selected = set()
        dist_sets = {}
        
        for p in order:
            if time.time() > start_time + time_budget:
                break
            
            can_add = True
            new_dists = set()
            for q in selected:
                d = get_dist(p, q)
                if d in new_dists:
                    can_add = False
                    break
                new_dists.add(d)
            
            if not can_add:
                continue
            
            for q in selected:
                d = get_dist(p, q)
                if d in dist_sets[q]:
                    can_add = False
                    break
            
            if can_add:
                selected.add(p)
                dist_sets[p] = new_dists
                for q in selected:
                    if q != p:
                        dist_sets[q].add(get_dist(p, q))
        
        return selected, dist_sets
    
    def can_add_point(p, selected, dist_sets):
        """Check if point p can be added to the selected set."""
        new_dists = set()
        for q in selected:
            d = get_dist(p, q)
            if d in new_dists:
                return False
            new_dists.add(d)
            if d in dist_sets[q]:
                return False
        return True
    
    def add_point(p, selected, dist_sets):
        """Add point p to the selected set, updating dist_sets."""
        new_dists = set()
        for q in selected:
            new_dists.add(get_dist(p, q))
        selected.add(p)
        dist_sets[p] = new_dists
        for q in selected:
            if q != p:
                dist_sets[q].add(get_dist(p, q))
    
    def remove_point(p, selected, dist_sets):
        """Remove point p from the selected set, updating dist_sets."""
        for q in selected:
            if q != p:
                dist_sets[q].discard(get_dist(p, q))
        del dist_sets[p]
        selected.discard(p)
    
    def local_search(selected, dist_sets, order, max_time=30.0):
        """
        Perform local search with single-point swaps and additions.
        Returns improved set and dist_sets.
        """
        elapsed_limit = start_time + max_time
        improved = True
        while improved and time.time() < elapsed_limit:
            improved = False
            
            # Phase 1: Try to add any unselected point
            for p in order:
                if time.time() > elapsed_limit:
                    break
                if p in selected:
                    continue
                if can_add_point(p, selected, dist_sets):
                    add_point(p, selected, dist_sets)
                    improved = True
            
            # Phase 2: Try single-point swaps (remove one, add another)
            selected_list = list(selected)
            for p_remove in selected_list:
                if time.time() > elapsed_limit:
                    break
                if len(selected) <= 3:
                    break
                
                # Save state for rollback
                saved_dist_sets = {q: set(dist_sets[q]) for q in selected}
                
                remove_point(p_remove, selected, dist_sets)
                
                # Try to find a replacement
                found = False
                for p_add in order:
                    if time.time() > elapsed_limit:
                        break
                    if p_add in selected or p_add == p_remove:
                        continue
                    
                    if can_add_point(p_add, selected, dist_sets):
                        add_point(p_add, selected, dist_sets)
                        found = True
                        improved = True
                        break
                
                if not found:
                    # Restore
                    selected.add(p_remove)
                    dist_sets[p_remove] = saved_dist_sets[p_remove]
                    for q in selected:
                        if q != p_remove:
                            dist_sets[q] = saved_dist_sets[q]
        
        return selected, dist_sets
    
    def local_search_double_swap(selected, dist_sets, order, max_time=20.0):
        """
        Perform local search with double-point swaps (remove two, add two).
        This can escape local optima that single swaps cannot.
        """
        elapsed_limit = start_time + max_time
        improved = True
        while improved and time.time() < elapsed_limit:
            improved = False
            
            # Try double swaps: remove two points, add two different points
            selected_list = list(selected)
            n_sel = len(selected_list)
            
            for i in range(n_sel):
                if time.time() > elapsed_limit:
                    break
                for j in range(i+1, n_sel):
                    if time.time() > elapsed_limit:
                        break
                    if len(selected) <= 4:
                        break
                    
                    p_remove1 = selected_list[i]
                    p_remove2 = selected_list[j]
                    
                    # Save state
                    saved_selected = set(selected)
                    saved_dist_sets = {q: set(dist_sets[q]) for q in selected}
                    
                    # Remove points
                    remove_point(p_remove1, selected, dist_sets)
                    remove_point(p_remove2, selected, dist_sets)
                    
                    # Try to find two replacements
                    p_add1 = -1
                    p_add2 = -1
                    found1 = False
                    found2 = False
                    
                    for p_add in order:
                        if time.time() > elapsed_limit:
                            break
                        if p_add in selected or p_add == p_remove1 or p_add == p_remove2:
                            continue
                        
                        if can_add_point(p_add, selected, dist_sets):
                            # Tentatively add to temporary structures to check second point
                            temp_selected = set(selected)
                            temp_dist = {q: set(dist_sets[q]) for q in selected}
                            add_point(p_add, temp_selected, temp_dist)
                            p_add1 = p_add
                            found1 = True
                            
                            # Now try to find a second addition
                            for p_add2_candidate in order:
                                if time.time() > elapsed_limit:
                                    break
                                if p_add2_candidate in temp_selected or p_add2_candidate == p_remove1 or p_add2_candidate == p_remove2:
                                    continue
                                
                                if can_add_point(p_add2_candidate, temp_selected, temp_dist):
                                    p_add2 = p_add2_candidate
                                    found2 = True
                                    break
                            
                            break
                    
                    if found1 and found2:
                        # Commit the double swap to the main structures
                        # Note: selected and dist_sets are currently in the "removed" state
                        add_point(p_add1, selected, dist_sets)
                        add_point(p_add2, selected, dist_sets)
                        improved = True
                        break
                    else:
                        # Restore state
                        selected.clear()
                        selected.update(saved_selected)
                        dist_sets.clear()
                        for q in saved_dist_sets:
                            dist_sets[q] = saved_dist_sets[q]
            
            # Also try single additions after double swaps
            for p in order:
                if time.time() > elapsed_limit:
                    break
                if p in selected:
                    continue
                if can_add_point(p, selected, dist_sets):
                    add_point(p, selected, dist_sets)
                    improved = True
        
        return selected, dist_sets
    
    # Try multiple different orderings with time management
    best_set = set()
    best_dist_sets = {}
    
    remaining_time = deadline - start_time
    
    # Define orderings
    orderings = []
    
    # Ordering 1: Random
    rng = random.Random(42)
    order1 = list(range(num_points))
    rng.shuffle(order1)
    orderings.append(("random", order1))
    
    # Ordering 2: Sort by x-coordinate, then y
    order2 = sorted(range(num_points), key=lambda i: (coords[i][0], coords[i][1]))
    orderings.append(("xy_sort", order2))
    
    # Ordering 3: Sort by distance from center
    cx, cy = N/2.0, N/2.0
    order3 = sorted(range(num_points), key=lambda i: (coords[i][0]-cx)**2 + (coords[i][1]-cy)**2)
    orderings.append(("center_dist", order3))
    
    # Ordering 4: Corner bias
    corner_scores = []
    for i in range(num_points):
        x, y = coords[i]
        min_corner_dist = min(
            x*x + y*y,
            (N-1-x)**2 + y*y,
            x*x + (N-1-y)**2,
            (N-1-x)**2 + (N-1-y)**2
        )
        corner_scores.append((min_corner_dist, i))
    corner_scores.sort()
    order4 = [i for _, i in corner_scores]
    orderings.append(("corner", order4))
    
    # Ordering 5: Spiral from center
    order5 = []
    visited = set()
    cx, cy = N//2, N//2
    for r in range(max(N//2, 1) + 1):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                if max(abs(dx), abs(dy)) == r:
                    x, y = cx+dx, cy+dy
                    if 0 <= x < N and 0 <= y < N:
                        idx = y*N + x
                        if idx not in visited:
                            order5.append(idx)
                            visited.add(idx)
    for i in range(num_points):
        if i not in visited:
            order5.append(i)
    orderings.append(("spiral", order5))
    
    # Ordering 6: Diagonal pattern
    order6 = sorted(range(num_points), key=lambda i: (coords[i][0] + coords[i][1], coords[i][0]))
    orderings.append(("diagonal", order6))
    
    # Allocate time to each ordering
    num_orderings = len(orderings)
    time_per_ordering = (deadline - start_time - 10.0) / num_orderings  # Reserve 10s for final validation
    
    for name, order in orderings:
        if time.time() > deadline - 15.0:
            break
        
        # Phase 1: Greedy construction
        selected, dist_sets = greedy_construction(order, time_budget=time_per_ordering * 0.3)
        
        # Phase 2: Single-point local search
        time_remaining_for_ls = time_per_ordering * 0.4
        selected, dist_sets = local_search(selected, dist_sets, order, max_time=time_remaining_for_ls)
        
        # Phase 3: Double-swap local search (if time permits)
        if time.time() < deadline - 20.0:
            time_for_double = min(time_per_ordering * 0.3, deadline - time.time() - 15.0)
            if time_for_double > 5.0:
                selected, dist_sets = local_search_double_swap(selected, dist_sets, order, max_time=time_for_double)
        
        # Update best
        if len(selected) > len(best_set):
            best_set = set(selected)
            best_dist_sets = {q: set(dist_sets[q]) for q in selected}
    
    # Final: try to add any remaining points to the best set
    if time.time() < deadline - 5.0:
        for p in range(num_points):
            if time.time() > deadline - 5.0:
                break
            if p in best_set:
                continue
            if can_add_point(p, best_set, best_dist_sets):
                add_point(p, best_set, best_dist_sets)
    
    # Validate the best set
    if not validate_set(best_set):
        # Fallback to a small valid set
        best_set = {0, N+2, 4*N+1}
    
    return [(i // N, i % N) for i in best_set]

# EVOLVE-BLOCK-END
