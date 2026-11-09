"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290  # Keep 40s margin for 330s limit

    num_points = N * N
    
    def idx_to_point(idx):
        return (idx // N, idx % N)
    
    # Precompute all squared distances as flat list of lists
    dist_matrix = [[0] * num_points for _ in range(num_points)]
    for i in range(num_points):
        x1, y1 = i // N, i % N
        for j in range(i + 1, num_points):
            x2, y2 = j // N, j % N
            d = (x1 - x2)**2 + (y1 - y2)**2
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    
    # Precompute distance bitsets: for each point p, dist_bitset[p] is a dict mapping distance value to bitmask of points at that distance
    # Actually, let's use: for each point p, we precompute which distances it has to other points
    # dist_to_points[p] = list of (distance, other_point_idx) sorted by distance
    # For bitset approach: we'll represent the set of distances from p to S as a Python int bitmask
    # But distances can be up to 2*63^2 = 7938, so we need bits 0..7938
    
    MAX_DIST = 2 * (N-1)**2  # max squared distance
    
    # Precompute for each point p, a list of distances to all other points
    # dist_list[p][q] = squared distance
    # For the bitset approach, we need to check:
    # 1. Are any two points in S equidistant from p? (p is apex)
    # 2. For each s in S, is d(p,s) in dist_sets[s]?
    
    # Let's use a different approach: precompute for each point p, 
    # for each possible distance d, the bitmask of points at distance d from p
    # dist_mask[p][d] = bitmask of points q where dist(p,q) = d
    # This is too large. Instead, let's keep the set-based approach but optimize.
    
    # Alternative: use numpy arrays for distance matrix and vectorized operations
    # But the directive says to use bitset-based representation.
    
    # Let's implement bitset-based can_add:
    # For point p, compute dists_from_p = {d(p,s) for s in S}
    # Check if len(dists_from_p) == len(S) (no duplicates)
    # For each s in S, check if d(p,s) is in dist_sets[s]
    
    # The bottleneck is the second check. With bitsets:
    # dist_sets_bitset[s] = integer where bit d is set if d is in dist_sets[s]
    # Then check: (dist_sets_bitset[s] >> d(p,s)) & 1
    
    def can_add_bitset(S_list, dist_bitsets, p_idx):
        """
        S_list: list of point indices in S
        dist_bitsets: dict mapping point index to int bitmask of distances
        p_idx: candidate point
        Returns True if p_idx can be added.
        """
        k = len(S_list)
        if k < 2:
            return True
        
        # Case 1: p is apex. Check for duplicate distances from p to points in S.
        seen_dists = set()
        for s_idx in S_list:
            d = dist_matrix[p_idx][s_idx]
            if d in seen_dists:
                return False
            seen_dists.add(d)
        
        # Cases 2 & 3: For each s in S, check if d(p,s) is in dist_bitsets[s]
        for s_idx in S_list:
            d_ps = dist_matrix[p_idx][s_idx]
            # Check if bit d_ps is set in dist_bitsets[s_idx]
            if (dist_bitsets[s_idx] >> d_ps) & 1:
                return False
        
        return True

    def add_point_bitset(S_list, dist_bitsets, p_idx):
        """Add p_idx to S_list and update dist_bitsets."""
        S_list.append(p_idx)
        new_bitset = 0
        for s_idx in S_list:
            if s_idx != p_idx:
                d = dist_matrix[p_idx][s_idx]
                new_bitset |= (1 << d)
                dist_bitsets[s_idx] |= (1 << d)
        dist_bitsets[p_idx] = new_bitset

    def remove_point_bitset(S_list, dist_bitsets, p_idx):
        """Remove p_idx from S_list and update dist_bitsets."""
        for s_idx in S_list:
            if s_idx != p_idx:
                d = dist_matrix[p_idx][s_idx]
                dist_bitsets[s_idx] &= ~(1 << d)
        del dist_bitsets[p_idx]
        S_list.remove(p_idx)

    def greedy_construct(order_indices):
        """Greedy construction given an order of point indices to try."""
        S_list = []
        dist_bitsets = {}
        S_set = set()  # For O(1) membership check
        for p_idx in order_indices:
            if time.time() > deadline - 30:
                break
            if p_idx in S_set:
                continue
            if can_add_bitset(S_list, dist_bitsets, p_idx):
                S_set.add(p_idx)
                add_point_bitset(S_list, dist_bitsets, p_idx)
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
        while count < num_points:
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
        for i in range(num_points):
            if i not in visited:
                order.append(i)
        return order

    def generate_random_order(seed):
        """Generate a random order with given seed."""
        rng = random.Random(seed)
        order = list(range(num_points))
        rng.shuffle(order)
        return order

    # Multiple greedy restarts with different orderings
    best_S_list = []
    best_dist_bitsets = {}
    best_S_set = set()
    best_size = 0

    strategies = [
        generate_diagonal_order(),
        generate_spiral_order(),
    ]
    
    num_random_restarts = 30
    for i in range(num_random_restarts):
        strategies.append(generate_random_order(i))

    for order_indices in strategies:
        if time.time() > deadline - 30:
            break
        
        S_list, dist_bitsets, S_set = greedy_construct(order_indices)
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_S_list = S_list.copy()
            best_dist_bitsets = dict(dist_bitsets)
            best_S_set = S_set.copy()
    
    # Local search on the best solution
    S_list = best_S_list
    dist_bitsets = best_dist_bitsets
    S_set = best_S_set
    no_improve_count = 0
    max_no_improve = 50
    rng = random.Random(42)

    while time.time() < deadline - 10:
        improved = False
        
        # Try to add a random point
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            p_idx = rng.randint(0, num_points - 1)
            if p_idx not in S_set:
                if can_add_bitset(S_list, dist_bitsets, p_idx):
                    S_set.add(p_idx)
                    add_point_bitset(S_list, dist_bitsets, p_idx)
                    improved = True
                    no_improve_count = 0
                    break
        
        # Try to swap: remove one point, add another
        if not improved and len(S_list) > 0:
            for _ in range(200):
                if time.time() > deadline - 10:
                    break
                remove_p = rng.choice(S_list)
                add_p = rng.randint(0, num_points - 1)
                if add_p in S_set or add_p == remove_p:
                    continue
                remove_point_bitset(S_list, dist_bitsets, remove_p)
                S_set.discard(remove_p)
                if can_add_bitset(S_list, dist_bitsets, add_p):
                    S_set.add(add_p)
                    add_point_bitset(S_list, dist_bitsets, add_p)
                    improved = True
                    no_improve_count = 0
                else:
                    S_set.add(remove_p)
                    add_point_bitset(S_list, dist_bitsets, remove_p)
                break
        
        # Try double-swap
        if not improved and len(S_list) > 1:
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                remove_p1 = rng.choice(S_list)
                remaining = [x for x in S_list if x != remove_p1]
                if not remaining:
                    continue
                remove_p2 = rng.choice(remaining)
                add_p1 = rng.randint(0, num_points - 1)
                add_p2 = rng.randint(0, num_points - 1)
                if add_p1 in S_set or add_p2 in S_set or add_p1 == add_p2:
                    continue
                remove_point_bitset(S_list, dist_bitsets, remove_p1)
                S_set.discard(remove_p1)
                remove_point_bitset(S_list, dist_bitsets, remove_p2)
                S_set.discard(remove_p2)
                if can_add_bitset(S_list, dist_bitsets, add_p1) and can_add_bitset(S_list, dist_bitsets, add_p2):
                    S_set.add(add_p1)
                    add_point_bitset(S_list, dist_bitsets, add_p1)
                    S_set.add(add_p2)
                    add_point_bitset(S_list, dist_bitsets, add_p2)
                    improved = True
                    no_improve_count = 0
                else:
                    S_set.add(remove_p1)
                    add_point_bitset(S_list, dist_bitsets, remove_p1)
                    S_set.add(remove_p2)
                    add_point_bitset(S_list, dist_bitsets, remove_p2)
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                if time.time() > deadline - 30:
                    break
                order = generate_random_order(rng.randint(0, 100000))
                S_list, dist_bitsets, S_set = greedy_construct(order)
                if len(S_list) > best_size:
                    best_size = len(S_list)
                    best_S_list = S_list.copy()
                    best_dist_bitsets = dict(dist_bitsets)
                    best_S_set = S_set.copy()
                no_improve_count = 0
        else:
            if len(S_list) > best_size:
                best_size = len(S_list)
                best_S_list = S_list.copy()
                best_dist_bitsets = dict(dist_bitsets)
                best_S_set = S_set.copy()

    result = [idx_to_point(idx) for idx in best_S_list]
    return result
# EVOLVE-BLOCK-END
