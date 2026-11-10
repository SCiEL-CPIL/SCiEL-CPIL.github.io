"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

# Precompute all pairs of grid points and their squared distances
# Grid point index: i = y * N + x
def _grid_point(idx):
    return (idx % N, idx // N)

def _dist2(i, j):
    pi = _grid_point(i)
    pj = _grid_point(j)
    return (pi[0] - pj[0])**2 + (pi[1] - pj[1])**2

# Precompute distance matrix
def _build_dist_matrix():
    n_pts = N * N
    D = np.zeros((n_pts, n_pts), dtype=np.int64)
    for i in range(n_pts):
        xi, yi = _grid_point(i)
        for j in range(i + 1, n_pts):
            xj, yj = _grid_point(j)
            d = (xi - xj)**2 + (yi - yj)**2
            D[i, j] = d
            D[j, i] = d
    return D

D = _build_dist_matrix()
n_pts = N * N

# For each point p, we need to check if adding p to set S creates an isosceles triangle.
# p forms an isosceles triangle with a, b in S if:
#   dist(p,a) == dist(p,b)  OR  dist(p,a) == dist(a,b)  OR  dist(p,b) == dist(a,b)

def _can_add(p, chosen_set, chosen_list):
    """Check if point p can be added to chosen_set without creating an isosceles triangle."""
    # If p is already in chosen_set, we don't need to check
    if p in chosen_set:
        return False
    
    # Check against all pairs in chosen_set
    n = len(chosen_list)
    for i in range(n):
        a = chosen_list[i]
        d_pa = D[p, a]
        for j in range(i + 1, n):
            b = chosen_list[j]
            d_pb = D[p, b]
            d_ab = D[a, b]
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return False
    return True

def _greedy_maximal(order):
    """Build a maximal no-isosceles set by greedily adding points in the given order."""
    chosen = []
    chosen_set = set()
    for p in order:
        if _can_add(p, chosen_set, chosen):
            chosen.append(p)
            chosen_set.add(p)
    return chosen

def _is_maximal(chosen_list):
    """Check if the set is maximal (no point can be added)."""
    chosen_set = set(chosen_list)
    for p in range(n_pts):
        if p not in chosen_set and _can_add(p, chosen_set, chosen_list):
            return False
    return True

def _local_search(initial_set, max_iters=5000):
    """Local search to minimize the size of a maximal no-isosceles set."""
    chosen_list = list(initial_set)
    chosen_set = set(chosen_list)
    best_list = list(chosen_list)
    best_size = len(chosen_list)
    
    rng = random.Random(42)
    
    for it in range(max_iters):
        # Try to remove a point and see if the set remains maximal
        # If removing a point makes it non-maximal, we need to re-add points
        # Alternative: try swapping - remove one point, then greedily add points to restore maximality
        
        if len(chosen_list) <= 1:
            break
        
        # Try removing each point
        improved = False
        for idx in range(len(chosen_list)):
            p = chosen_list[idx]
            # Temporarily remove p
            new_list = [x for i, x in enumerate(chosen_list) if i != idx]
            new_set = set(new_list)
            
            # Check if new_set is still a no-isosceles set (it should be, since it's a subset)
            # Now check if it's maximal. If not, greedily add points to make it maximal.
            # The added points should be different from p to get a smaller set.
            
            # Check if p can be added back (if not, then removing p broke maximality in a way
            # that requires p or other points)
            # Actually, we want to see if we can make a maximal set from new_set that doesn't include p
            # and has smaller size than original.
            
            # Greedily add points to new_set to make it maximal, avoiding p
            added = []
            for q in range(n_pts):
                if q in new_set or q == p:
                    continue
                if _can_add(q, new_set, new_list + added):
                    added.append(q)
                    new_set.add(q)
                    new_list.append(q)
            
            # Now new_list is a maximal no-isosceles set that doesn't include p
            if len(new_list) < best_size:
                best_list = list(new_list)
                best_size = len(new_list)
                improved = True
                break
        
        if not improved:
            # Try a random swap: remove one point, add another
            idx = rng.randrange(len(chosen_list))
            p = chosen_list[idx]
            new_list = [x for i, x in enumerate(chosen_list) if i != idx]
            new_set = set(new_list)
            
            # Try to add a random point q != p
            candidates = [q for q in range(n_pts) if q not in new_set and q != p]
            if candidates:
                q = rng.choice(candidates)
                if _can_add(q, new_set, new_list):
                    new_list.append(q)
                    new_set.add(q)
                    # Now make it maximal
                    for r in range(n_pts):
                        if r in new_set:
                            continue
                        if _can_add(r, new_set, new_list):
                            new_list.append(r)
                            new_set.add(r)
                    if len(new_list) < best_size:
                        best_list = list(new_list)
                        best_size = len(new_list)
                        improved = True
        
        if not improved:
            # Try a different strategy: start over with a random subset of current set
            # and greedily complete it
            if len(chosen_list) > 5:
                subset_size = max(3, len(chosen_list) - rng.randrange(1, 4))
                subset = rng.sample(chosen_list, subset_size)
                # Check if subset is valid (no isosceles triangles)
                valid = True
                for i in range(len(subset)):
                    for j in range(i + 1, len(subset)):
                        for k in range(j + 1, len(subset)):
                            a, b, c = subset[i], subset[j], subset[k]
                            d1 = D[a, b]
                            d2 = D[a, c]
                            d3 = D[b, c]
                            if d1 == d2 or d1 == d3 or d2 == d3:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    new_list = list(subset)
                    new_set = set(subset)
                    for q in range(n_pts):
                        if q in new_set:
                            continue
                        if _can_add(q, new_set, new_list):
                            new_list.append(q)
                            new_set.add(q)
                    if len(new_list) < best_size:
                        best_list = list(new_list)
                        best_size = len(new_list)
                        improved = True
        
        if not improved:
            break
    
    return best_list

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    # Strategy 1: Greedy in different orders
    best_set = None
    best_size = float('inf')
    
    # Row-major order
    order1 = list(range(n_pts))
    s1 = _greedy_maximal(order1)
    best_set = s1
    best_size = len(s1)
    
    # Reverse order
    order2 = list(reversed(range(n_pts)))
    s2 = _greedy_maximal(order2)
    if len(s2) < best_size:
        best_set = s2
        best_size = len(s2)
    
    # Random orders
    rng = random.Random(123)
    for trial in range(20):
        if time.time() > deadline:
            break
        order = list(range(n_pts))
        rng.shuffle(order)
        s = _greedy_maximal(order)
        if len(s) < best_size:
            best_set = s
            best_size = len(s)
    
    # Local search on the best set found
    if time.time() < deadline:
        ls_result = _local_search(best_set, max_iters=2000)
        if len(ls_result) < best_size:
            best_set = ls_result
            best_size = len(ls_result)
    
    # More random restarts with local search
    for trial in range(30):
        if time.time() > deadline:
            break
        order = list(range(n_pts))
        rng.shuffle(order)
        s = _greedy_maximal(order)
        if len(s) < best_size:
            ls_result = _local_search(s, max_iters=1000)
            if len(ls_result) < best_size:
                best_set = ls_result
                best_size = len(ls_result)
    
    # Convert indices back to (x, y) tuples
    result = [_grid_point(p) for p in best_set]
    return result

# EVOLVE-BLOCK-END
