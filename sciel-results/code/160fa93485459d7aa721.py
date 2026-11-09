"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

N = 64
M = N * N
MAX_DIST = 2 * (N - 1) ** 2

# Precompute distance matrix once at module level for speed
_x = np.arange(N)
_y = np.arange(N)
_x_grid, _y_grid = np.meshgrid(_x, _y, indexing='ij')
_dx = _x_grid.ravel()[:, None] - _x_grid.ravel()[None, :]
_dy = _y_grid.ravel()[:, None] - _y_grid.ravel()[None, :]
DIST_MATRIX = (_dx ** 2 + _dy ** 2).astype(np.int64)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290

    def get_dist(i, j):
        return int(DIST_MATRIX[i, j])

    def can_add(S_set, dist_bitsets, p_idx):
        """
        Check if p_idx can be added to S_set without creating isosceles triangles.
        S_set: set of point indices
        dist_bitsets: dict mapping point index to int bitmask of distances to other points in S
        """
        if p_idx in S_set:
            return False

        # Get all distances from p_idx to points in S at once
        s_indices = list(S_set)
        if not s_indices:
            return True
        
        dists = DIST_MATRIX[p_idx, s_indices]

        # Case 1: p is apex. Check for duplicate distances from p to points in S.
        seen_dists = set()
        for d in dists:
            d_int = int(d)
            if d_int in seen_dists:
                return False
            seen_dists.add(d_int)

        # Cases 2 & 3: For each s in S, check if d(p,s) is in dist_bitsets[s]
        # If d(p,s) is in dist_bitsets[s], then there exists t in S such that d(s,t) == d(s,p),
        # which means triangle (s, p, t) is isosceles with apex s.
        for i, s_idx in enumerate(s_indices):
            d_ps = int(dists[i])
            if (dist_bitsets[s_idx] >> d_ps) & 1:
                return False

        return True

    def add_point(S_list, S_set, dist_bitsets, p_idx):
        """Add p_idx to S_list and update dist_bitsets."""
        S_list.append(p_idx)
        S_set.add(p_idx)
        new_bitset = 0
        for s_idx in S_list:
            if s_idx != p_idx:
                d = int(DIST_MATRIX[p_idx, s_idx])
                new_bitset |= (1 << d)
                dist_bitsets[s_idx] |= (1 << d)
        dist_bitsets[p_idx] = new_bitset

    def remove_point(S_list, S_set, dist_bitsets, p_idx):
        """Remove p_idx from S_list and update dist_bitsets."""
        for s_idx in S_list:
            if s_idx != p_idx:
                d = int(DIST_MATRIX[p_idx, s_idx])
                dist_bitsets[s_idx] &= ~(1 << d)
        del dist_bitsets[p_idx]
        S_set.discard(p_idx)
        S_list.remove(p_idx)

    def greedy_construct(order_indices, time_limit):
        """Greedy construction given an order of point indices to try."""
        S_list = []
        dist_bitsets = {}
        S_set = set()
        for p_idx in order_indices:
            if time.time() > time_limit:
                break
            if p_idx in S_set:
                continue
            if can_add(S_set, dist_bitsets, p_idx):
                add_point(S_list, S_set, dist_bitsets, p_idx)
        return S_list, dist_bitsets, S_set

    def generate_diagonal_order():
        """Generate points in diagonal sweep order."""
        order = []
        for s in range(2 * N - 1):
            for x in range(max(0, s - N + 1), min(N, s + 1)):
                y = s - x
                order.append(x * N + y)
        return order

    def generate_spiral_order():
        """Generate points in spiral order from center."""
        cx, cy = N // 2, N // 2
        order = [cx * N + cy]
        visited = set([cx * N + cy])
        dx, dy = 1, 0
        steps = 1
        count = 1
        while count < M:
            for _ in range(2):
                for _ in range(steps):
                    cx += dx
                    cy += dy
                    if 0 <= cx < N and 0 <= cy < N:
                        idx = cx * N + cy
                        if idx not in visited:
                            visited.add(idx)
                            order.append(idx)
                            count += 1
                dx, dy = -dy, dx
            steps += 1
        for i in range(M):
            if i not in visited:
                order.append(i)
        return order

    def generate_random_order(seed):
        """Generate a random order with given seed."""
        rng = random.Random(seed)
        order = list(range(M))
        rng.shuffle(order)
        return order

    best_S_list = []
    best_dist_bitsets = {}
    best_S_set = set()
    best_size = 0

    # Phase 1: Greedy restarts with structured and random orders
    strategies = [
        generate_diagonal_order(),
        generate_spiral_order(),
    ]
    
    # Add several random orders - increase count since operations are faster
    num_random_restarts = 40
    for i in range(num_random_restarts):
        strategies.append(generate_random_order(i))

    for order_indices in strategies:
        if time.time() > deadline - 60:
            break
        
        S_list, dist_bitsets, S_set = greedy_construct(order_indices, deadline - 60)
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_S_list = S_list.copy()
            best_dist_bitsets = dict(dist_bitsets)
            best_S_set = S_set.copy()

    # Phase 2: Local Search with Perturbation
    # We use the best solution from Phase 1 as the starting point.
    S_list = best_S_list
    dist_bitsets = best_dist_bitsets
    S_set = best_S_set
    
    rng = random.Random(42)
    no_improve_count = 0
    max_no_improve_for_perturb = 10
    max_no_improve_for_restart = 100
    
    # Precompute list of all indices for candidate generation
    all_indices = list(range(M))

    while time.time() < deadline - 5:
        improved = False
        
        # Generate a shuffled list of candidates (points not in S)
        # This is O(M) which is 4096, very fast.
        candidates = [i for i in all_indices if i not in S_set]
        rng.shuffle(candidates)
        
        # Try to add any point from the shuffled candidates
        for p_idx in candidates:
            if time.time() > deadline - 5:
                break
            if can_add(S_set, dist_bitsets, p_idx):
                add_point(S_list, S_set, dist_bitsets, p_idx)
                improved = True
                no_improve_count = 0
                if len(S_list) > best_size:
                    best_size = len(S_list)
                    best_S_list = S_list.copy()
                    best_dist_bitsets = dict(dist_bitsets)
                    best_S_set = S_set.copy()
                break # Restart candidate generation to find the next best one
        
        if not improved:
            no_improve_count += 1
            
            if no_improve_count > max_no_improve_for_restart:
                # Full restart with a new random order
                if time.time() > deadline - 30:
                    break
                
                order = generate_random_order(rng.randint(0, 100000))
                S_list, dist_bitsets, S_set = greedy_construct(order, deadline - 30)
                
                if len(S_list) > best_size:
                    best_size = len(S_list)
                    best_S_list = S_list.copy()
                    best_dist_bitsets = dict(dist_bitsets)
                    best_S_set = S_set.copy()
                
                no_improve_count = 0
            elif no_improve_count > max_no_improve_for_perturb:
                # Perturbation: remove a few points to allow different additions
                if len(S_list) > 5:
                    num_remove = min(len(S_list), rng.randint(1, 3))
                    to_remove = rng.sample(S_list, num_remove)
                    for p in to_remove:
                        remove_point(S_list, S_set, dist_bitsets, p)
                    no_improve_count = 0
                else:
                    # If set is small, just restart
                    order = generate_random_order(rng.randint(0, 100000))
                    S_list, dist_bitsets, S_set = greedy_construct(order, deadline - 30)
                    if len(S_list) > best_size:
                        best_size = len(S_list)
                        best_S_list = S_list.copy()
                        best_dist_bitsets = dict(dist_bitsets)
                        best_S_set = S_set.copy()
                    no_improve_count = 0

    # Phase 3: Final polish - try adding points multiple times with different orderings
    if time.time() < deadline - 2:
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            candidates = [i for i in all_indices if i not in best_S_set]
            rng.shuffle(candidates)
            for p_idx in candidates:
                if time.time() > deadline - 2:
                    break
                if can_add(best_S_set, best_dist_bitsets, p_idx):
                    add_point(best_S_list, best_S_set, best_dist_bitsets, p_idx)

    result = [(idx // N, idx % N) for idx in best_S_list]
    return result
# EVOLVE-BLOCK-END
