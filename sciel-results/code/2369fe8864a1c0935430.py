"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 64
M = N * N
MAX_DIST = 2 * (N - 1) ** 2

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290

    # Precompute squared distances in a flat list of arrays for speed
    # dist_flat[i * M + j] = squared distance between point i and point j
    dist_flat = [0] * (M * M)
    for i in range(M):
        x1, y1 = i // N, i % N
        for j in range(i + 1, M):
            x2, y2 = j // N, j % N
            d = (x1 - x2) ** 2 + (y1 - y2) ** 2
            dist_flat[i * M + j] = d
            dist_flat[j * M + i] = d

    def get_dist(i, j):
        return dist_flat[i * M + j]

    def can_add(S_list, dist_bitsets, p_idx):
        """
        Check if p_idx can be added to S_list without creating isosceles triangles.
        S_list: list of point indices
        dist_bitsets: dict mapping point index to int bitmask of distances to other points in S
        """
        k = len(S_list)
        if k < 2:
            return True

        # Case 1: p is apex. Check for duplicate distances from p to points in S.
        seen_dists = set()
        for s_idx in S_list:
            d = get_dist(p_idx, s_idx)
            if d in seen_dists:
                return False
            seen_dists.add(d)

        # Cases 2 & 3: For each s in S, check if d(p,s) is in dist_bitsets[s]
        # If d(p,s) is in dist_bitsets[s], then there exists t in S such that d(s,t) == d(s,p),
        # which means triangle (s, p, t) is isosceles with apex s.
        for s_idx in S_list:
            d_ps = get_dist(p_idx, s_idx)
            if (dist_bitsets[s_idx] >> d_ps) & 1:
                return False

        return True

    def add_point(S_list, dist_bitsets, p_idx):
        """Add p_idx to S_list and update dist_bitsets."""
        S_list.append(p_idx)
        new_bitset = 0
        for s_idx in S_list:
            if s_idx != p_idx:
                d = get_dist(p_idx, s_idx)
                new_bitset |= (1 << d)
                dist_bitsets[s_idx] |= (1 << d)
        dist_bitsets[p_idx] = new_bitset

    def remove_point(S_list, dist_bitsets, p_idx):
        """Remove p_idx from S_list and update dist_bitsets."""
        for s_idx in S_list:
            if s_idx != p_idx:
                d = get_dist(p_idx, s_idx)
                dist_bitsets[s_idx] &= ~(1 << d)
        del dist_bitsets[p_idx]
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
            if can_add(S_list, dist_bitsets, p_idx):
                S_set.add(p_idx)
                add_point(S_list, dist_bitsets, p_idx)
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
    
    # Add several random orders
    num_random_restarts = 20
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

    # Phase 2: Local Search / Simulated Annealing hybrid
    # We use the best solution from Phase 1 as the starting point.
    S_list = best_S_list
    dist_bitsets = best_dist_bitsets
    S_set = best_S_set
    
    rng = random.Random(42)
    no_improve_count = 0
    max_no_improve = 50
    
    # We will try to improve the set by:
    # 1. Adding points if possible
    # 2. Swapping points (remove one, add another)
    # 3. Double swapping (remove two, add two)
    # If stuck for a while, we perturb by removing a few points and re-greedy-ing.
    
    while time.time() < deadline - 10:
        improved = False
        
        # 1. Try to add a random point
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            p_idx = rng.randint(0, M - 1)
            if p_idx not in S_set:
                if can_add(S_list, dist_bitsets, p_idx):
                    S_set.add(p_idx)
                    add_point(S_list, dist_bitsets, p_idx)
                    improved = True
                    no_improve_count = 0
                    if len(S_list) > best_size:
                        best_size = len(S_list)
                        best_S_list = S_list.copy()
                        best_dist_bitsets = dict(dist_bitsets)
                        best_S_set = S_set.copy()
                    break
        
        # 2. Try to swap: remove one point, add another
        if not improved and len(S_list) > 0:
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                remove_p = rng.choice(S_list)
                add_p = rng.randint(0, M - 1)
                if add_p in S_set or add_p == remove_p:
                    continue
                remove_point(S_list, dist_bitsets, remove_p)
                S_set.discard(remove_p)
                if can_add(S_list, dist_bitsets, add_p):
                    S_set.add(add_p)
                    add_point(S_list, dist_bitsets, add_p)
                    improved = True
                    no_improve_count = 0
                    if len(S_list) > best_size:
                        best_size = len(S_list)
                        best_S_list = S_list.copy()
                        best_dist_bitsets = dict(dist_bitsets)
                        best_S_set = S_set.copy()
                else:
                    # Revert
                    S_set.add(remove_p)
                    add_point(S_list, dist_bitsets, remove_p)
                break
        
        # 3. Try double-swap
        if not improved and len(S_list) > 1:
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                remove_p1 = rng.choice(S_list)
                remaining = [x for x in S_list if x != remove_p1]
                if not remaining:
                    continue
                remove_p2 = rng.choice(remaining)
                add_p1 = rng.randint(0, M - 1)
                add_p2 = rng.randint(0, M - 1)
                if add_p1 in S_set or add_p2 in S_set or add_p1 == add_p2:
                    continue
                
                # Backup state
                S_list_backup = S_list.copy()
                dist_bitsets_backup = dict(dist_bitsets)
                S_set_backup = S_set.copy()
                
                remove_point(S_list, dist_bitsets, remove_p1)
                S_set.discard(remove_p1)
                remove_point(S_list, dist_bitsets, remove_p2)
                S_set.discard(remove_p2)
                
                if can_add(S_list, dist_bitsets, add_p1) and can_add(S_list, dist_bitsets, add_p2):
                    S_set.add(add_p1)
                    add_point(S_list, dist_bitsets, add_p1)
                    S_set.add(add_p2)
                    add_point(S_list, dist_bitsets, add_p2)
                    improved = True
                    no_improve_count = 0
                    if len(S_list) > best_size:
                        best_size = len(S_list)
                        best_S_list = S_list.copy()
                        best_dist_bitsets = dict(dist_bitsets)
                        best_S_set = S_set.copy()
                else:
                    # Revert
                    S_list = S_list_backup
                    dist_bitsets = dist_bitsets_backup
                    S_set = S_set_backup
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                if time.time() > deadline - 30:
                    break
                
                # Perturbation: remove a few points and re-greedy
                # Or just restart with a new random order
                # Let's try a hybrid: remove 5-10 random points, then greedily add
                if len(S_list) > 10:
                    num_remove = min(len(S_list), rng.randint(5, 10))
                    to_remove = rng.sample(S_list, num_remove)
                    for p in to_remove:
                        remove_point(S_list, dist_bitsets, p)
                        S_set.discard(p)
                    
                    # Re-greedy with a random order
                    order = generate_random_order(rng.randint(0, 100000))
                    S_list, dist_bitsets, S_set = greedy_construct(order, deadline - 30)
                    # Note: greedy_construct starts from empty, so we lose the previous points.
                    # To keep them, we'd need a "fill" mode. For now, let's just restart fully if stuck.
                    # Actually, let's just do a full restart with a new random order if stuck for too long.
                    # The above code replaced S_list with a new greedy result.
                    
                    if len(S_list) > best_size:
                        best_size = len(S_list)
                        best_S_list = S_list.copy()
                        best_dist_bitsets = dict(dist_bitsets)
                        best_S_set = S_set.copy()
                else:
                    # Small set, just restart
                    order = generate_random_order(rng.randint(0, 100000))
                    S_list, dist_bitsets, S_set = greedy_construct(order, deadline - 30)
                    if len(S_list) > best_size:
                        best_size = len(S_list)
                        best_S_list = S_list.copy()
                        best_dist_bitsets = dict(dist_bitsets)
                        best_S_set = S_set.copy()
                
                no_improve_count = 0

    result = [(idx // N, idx % N) for idx in best_S_list]
    return result
# EVOLVE-BLOCK-END
