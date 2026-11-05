"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 290  # Safety margin
    
    # Precompute all squared distances between all pairs of grid points
    num_points = N * N
    xs = np.arange(N, dtype=np.int32)
    ys = np.arange(N, dtype=np.int32)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    xs_flat = X.flatten()
    ys_flat = Y.flatten()
    
    dx = xs_flat[:, None] - xs_flat[None, :]
    dy = ys_flat[:, None] - ys_flat[None, :]
    dist2 = (dx*dx + dy*dy)
    
    # Map from point index to (x, y)
    point_coords = [(int(xs_flat[i]), int(ys_flat[i])) for i in range(num_points)]
    
    def isosceles_check(S_list):
        """Check if a set of points has no isosceles triangle."""
        if len(S_list) < 3:
            return True
        S_set = set(S_list)
        # For each point, collect all squared distances to other points in S
        for u in S_list:
            d_set = {}
            for v in S_list:
                if u == v:
                    continue
                d = int(dist2[u, v])
                d_set[d] = d_set.get(d, 0) + 1
            for d, cnt in d_set.items():
                if cnt >= 2:
                    return False
        return True
    
    def greedy_construct(order, time_limit=None):
        """Greedy: add points in given order if they don't create an isosceles triangle."""
        S = []
        S_set = set()
        # dist_map[point] = {distance: count}
        dist_map = {}
        
        for idx_P in order:
            if idx_P in S_set:
                continue
            if time_limit and time.time() > time_limit:
                break
            
            # Check if P can be added
            valid = True
            new_dists = {}
            
            for idx_Q in S:
                d = int(dist2[idx_P, idx_Q])
                # Check if P already has another point at distance d
                if d in new_dists:
                    valid = False
                    break
                new_dists[d] = new_dists.get(d, 0) + 1
                # Check if Q already has another point at distance d
                q_dists = dist_map.get(idx_Q, {})
                if d in q_dists:
                    valid = False
                    break
            
            if valid:
                S.append(idx_P)
                S_set.add(idx_P)
                dist_map[idx_P] = new_dists
                for idx_Q in S:
                    if idx_Q != idx_P:
                        d = int(dist2[idx_Q, idx_P])
                        if idx_Q in dist_map:
                            dist_map[idx_Q][d] = dist_map[idx_Q].get(d, 0) + 1
        
        return S, S_set, dist_map
    
    def can_add(point, S_set, dist_map):
        """Check if a point can be added without creating an isosceles triangle."""
        if point in S_set:
            return False
        
        new_dists = {}
        for idx_Q in S_set:
            d = int(dist2[point, idx_Q])
            if d in new_dists:
                return False
            new_dists[d] = new_dists.get(d, 0) + 1
            q_dists = dist_map.get(idx_Q, {})
            if d in q_dists:
                return False
        
        return True
    
    def add_point(point, S, S_set, dist_map):
        """Add a point to the set, updating the dist_map."""
        S.append(point)
        S_set.add(point)
        new_dists = {}
        for idx_Q in S_set:
            if idx_Q == point:
                continue
            d = int(dist2[point, idx_Q])
            new_dists[d] = new_dists.get(d, 0) + 1
            q_dists = dist_map.get(idx_Q, {})
            if d in q_dists:
                q_dists[d] = q_dists[d] + 1
            else:
                q_dists[d] = 1
                dist_map[idx_Q] = q_dists
        dist_map[point] = new_dists
    
    def remove_point(point, S, S_set, dist_map):
        """Remove a point from the set, updating the dist_map."""
        S.remove(point)
        S_set.discard(point)
        point_dists = dist_map.get(point, {})
        for idx_Q in S_set:
            d = int(dist2[point, idx_Q])
            q_dists = dist_map.get(idx_Q, {})
            if d in q_dists:
                q_dists[d] = q_dists[d] - 1
                if q_dists[d] == 0:
                    del q_dists[d]
        del dist_map[point]
    
    # Phase 1: Multiple greedy constructions with different random orders
    best_S = []
    best_size = 0
    
    for trial in range(20):
        if time.time() > deadline - 120:
            break
        rng = random.Random(42 + trial)
        order = list(range(num_points))
        rng.shuffle(order)
        S, S_set, dist_map = greedy_construct(order, time_limit=deadline - 120)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
            best_S_set = S_set.copy()
            best_dist_map = {k: dict(v) for k, v in dist_map.items()}
    
    # Phase 2: Local search to improve the best set found
    if len(best_S) > 0:
        S = best_S[:]
        S_set = best_S_set.copy()
        dist_map = {k: dict(v) for k, v in best_dist_map.items()}
        rng = random.Random(12345)
        
        stuck_count = 0
        while time.time() < deadline - 5:
            if time.time() > deadline - 5:
                break
            
            # Try to add random points
            improved = False
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                p = rng.randint(0, num_points - 1)
                if p in S_set:
                    continue
                if can_add(p, S_set, dist_map):
                    add_point(p, S, S_set, dist_map)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S[:]
                        best_S_set = S_set.copy()
                        best_dist_map = {k: dict(v) for k, v in dist_map.items()}
            
            if not improved:
                stuck_count += 1
                # If stuck, try removing a random point to explore different configurations
                if len(S) > 0 and stuck_count > 10:
                    p = rng.choice(S)
                    remove_point(p, S, S_set, dist_map)
                    stuck_count = 0
                    # After removal, try greedy fill again
                    remaining = list(range(num_points))
                    rng.shuffle(remaining)
                    for idx_P in remaining:
                        if time.time() > deadline - 5:
                            break
                        if can_add(idx_P, S_set, dist_map):
                            add_point(idx_P, S, S_set, dist_map)
                            if len(S) > best_size:
                                best_size = len(S)
                                best_S = S[:]
                                best_S_set = S_set.copy()
                                best_dist_map = {k: dict(v) for k, v in dist_map.items()}
            else:
                stuck_count = 0
    
    return [point_coords[i] for i in best_S]
# EVOLVE-BLOCK-END
