"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290  # Keep 40s margin for 330s limit

    all_points = [(x, y) for x in range(N) for y in range(N)]
    num_points = N * N
    
    # Pre-compute squared distances between all pairs
    # dist_cache[p1_index][p2_index] = sq_dist
    # Use flat indexing: index = x * N + y
    def point_to_idx(p):
        return p[0] * N + p[1]
    
    def idx_to_point(idx):
        return (idx // N, idx % N)
    
    # Precompute all squared distances
    dist_matrix = [[0] * num_points for _ in range(num_points)]
    for i in range(num_points):
        x1, y1 = idx_to_point(i)
        for j in range(i + 1, num_points):
            x2, y2 = idx_to_point(j)
            d = (x1 - x2)**2 + (y1 - y2)**2
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d

    def sq_dist(p1, p2):
        return dist_matrix[point_to_idx(p1)][point_to_idx(p2)]

    def can_add_fast(S_idx_set, dist_sets, p_idx):
        """
        Check if point p_idx can be added to set S_idx_set without creating an isosceles triangle.
        dist_sets: dict mapping each point index in S to a set of squared distances to other points in S.
        Returns True if p_idx can be added.
        """
        if len(S_idx_set) < 2:
            return True
        
        # Case 1: p is the apex. Check if any two points in S are equidistant from p.
        dists_from_p = set()
        for s_idx in S_idx_set:
            d = dist_matrix[p_idx][s_idx]
            if d in dists_from_p:
                return False
            dists_from_p.add(d)
        
        # Cases 2 & 3: For each s in S, check if d(p,s) equals d(s,t) for some t in S, t != s.
        for s_idx in S_idx_set:
            d_ps = dist_matrix[p_idx][s_idx]
            if d_ps in dist_sets[s_idx]:
                return False
        
        return True

    def add_point(S_idx_set, dist_sets, p_idx):
        """Add point p_idx to S_idx_set and update dist_sets."""
        S_idx_set.add(p_idx)
        dist_sets[p_idx] = set()
        for s_idx in S_idx_set:
            if s_idx != p_idx:
                d = dist_matrix[p_idx][s_idx]
                dist_sets[p_idx].add(d)
                dist_sets[s_idx].add(d)

    def remove_point(S_idx_set, dist_sets, p_idx):
        """Remove point p_idx from S_idx_set and update dist_sets."""
        for s_idx in S_idx_set:
            if s_idx != p_idx:
                d = dist_matrix[p_idx][s_idx]
                dist_sets[s_idx].discard(d)
        del dist_sets[p_idx]
        S_idx_set.discard(p_idx)

    def greedy_construct(order_indices):
        """Greedy construction given an order of point indices to try."""
        S = set()
        dist_sets = {}
        for p_idx in order_indices:
            if time.time() > deadline - 30:
                break
            if can_add_fast(S, dist_sets, p_idx):
                add_point(S, dist_sets, p_idx)
        return S, dist_sets

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
        # Add any remaining points
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
    best_set = set()
    best_dist_sets = {}
    best_size = 0

    # Try several different orderings: diagonal, spiral, and random
    strategies = [
        generate_diagonal_order(),
        generate_spiral_order(),
    ]
    
    # Add random orders
    num_random_restarts = 30
    for i in range(num_random_restarts):
        strategies.append(generate_random_order(i))

    for order_indices in strategies:
        if time.time() > deadline - 30:
            break
        
        S, dist_sets = greedy_construct(order_indices)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            best_dist_sets = {k: v.copy() for k, v in dist_sets.items()}
    
    # Local search on the best solution
    S = best_set
    dist_sets = best_dist_sets
    no_improve_count = 0
    max_no_improve = 50

    while time.time() < deadline - 10:
        improved = False
        
        # Try to add a random point
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            p_idx = random.randint(0, num_points - 1)
            if p_idx not in S:
                if can_add_fast(S, dist_sets, p_idx):
                    add_point(S, dist_sets, p_idx)
                    improved = True
                    no_improve_count = 0
                    break
        
        # Try to swap: remove one point, add another
        if not improved and len(S) > 0:
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                remove_p = random.choice(list(S))
                add_p = random.randint(0, num_points - 1)
                if add_p in S or add_p == remove_p:
                    continue
                # Temporarily remove remove_p
                remove_point(S, dist_sets, remove_p)
                if can_add_fast(S, dist_sets, add_p):
                    add_point(S, dist_sets, add_p)
                    improved = True
                    no_improve_count = 0
                else:
                    # Restore remove_p
                    add_point(S, dist_sets, remove_p)
                break  # Only try one swap per iteration
        
        # Try double-swap: remove two points, add two different points
        if not improved and len(S) > 1:
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                remove_p1 = random.choice(list(S))
                remove_p2 = random.choice(list(S - {remove_p1}))
                add_p1 = random.randint(0, num_points - 1)
                add_p2 = random.randint(0, num_points - 1)
                if add_p1 in S or add_p2 in S or add_p1 == add_p2:
                    continue
                # Temporarily remove both
                remove_point(S, dist_sets, remove_p1)
                remove_point(S, dist_sets, remove_p2)
                if can_add_fast(S, dist_sets, add_p1) and can_add_fast(S, dist_sets, add_p2):
                    add_point(S, dist_sets, add_p1)
                    add_point(S, dist_sets, add_p2)
                    improved = True
                    no_improve_count = 0
                else:
                    # Restore both
                    add_point(S, dist_sets, remove_p1)
                    add_point(S, dist_sets, remove_p2)
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random restart: rebuild with different randomization
                if time.time() > deadline - 30:
                    break
                order = generate_random_order(random.randint(0, 10000))
                S, dist_sets = greedy_construct(order)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
                    best_dist_sets = {k: v.copy() for k, v in dist_sets.items()}
                no_improve_count = 0
        else:
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
                best_dist_sets = {k: v.copy() for k, v in dist_sets.items()}

    # Convert indices back to (x, y) tuples
    result = [idx_to_point(idx) for idx in best_set]
    return result
# EVOLVE-BLOCK-END
