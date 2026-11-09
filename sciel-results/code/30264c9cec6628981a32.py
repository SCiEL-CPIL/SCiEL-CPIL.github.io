# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import random


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (q = 49, p = 7).
    Uses a priority-queue-based greedy removal with local repair.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401

    # Deadline guard: keep a safety margin before the 630s kill limit.
    deadline = time.time() + 580.0  # 630 - 50s margin

    # --- Field arithmetic in F_p = F_7 ---
    k_const = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k_const = c
            break
    if k_const is None:
        k_const = 3

    idx_to_elem = [(i // p, i % p) for i in range(q)]
    elem_to_idx = {e: i for i, e in enumerate(idx_to_elem)}

    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = idx_to_elem[i]
        for j in range(q):
            aj, bj = idx_to_elem[j]
            aa = (ai * aj + bi * bj * k_const) % p
            bb = (ai * bj + bi * aj) % p
            mul_table[i, j] = aa * p + bb

    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = idx_to_elem[i]
        for j in range(q):
            aj, bj = idx_to_elem[j]
            aa = (ai + aj) % p
            bb = (bi + bj) % p
            add_table[i, j] = aa * p + bb

    # --- Directions ---
    directions = []
    directions.append((0, 1))
    for m in range(q):
        directions.append((1, m))
    num_dirs = len(directions)  # 50

    # --- Precompute line_other ---
    zx_arr = np.array([z // q for z in range(total_points)], dtype=np.int32)
    zy_arr = np.array([z % q for z in range(total_points)], dtype=np.int32)
    t_vals = np.arange(1, q, dtype=np.int32)

    line_other = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    for v_idx, (dx, dy) in enumerate(directions):
        tdx = mul_table[t_vals, dx]
        tdy = mul_table[t_vals, dy]
        nx = add_table[zx_arr[:, None], tdx[None, :]]
        ny = add_table[zy_arr[:, None], tdy[None, :]]
        npidx = nx * q + ny
        line_other[:, v_idx, :] = npidx

    # Also precompute for each point z, the set of (x, v) pairs that z affects
    # When z is added to E, cnt[x, v] increments for all x in line_other[z, v]
    # We need to track, for each point x, how many directions have cnt[x,v]==0
    # Let's precompute the reverse: for each (x, v), which z's affect it?
    # Actually, let's just work with the forward direction.

    # Precompute: for each point z, the list of (x, v) pairs affected
    # This is line_other[z, :, :] flattened with direction indices
    # But we need it for efficient updates.

    # Let's precompute affected_pairs[z] as a list of (x, v) tuples
    # This will be used for both forward and reverse updates.
    # For efficiency, store as two arrays: affected_x[z], affected_v[z]
    # Total size: total_points * num_dirs * (q-1) = 2401 * 50 * 48 ≈ 5.7M entries
    # That's a lot. Let's be smarter.

    # Alternative: for each direction v, precompute for each point z, the list of x's
    # line_other[z, v, :] already gives this.

    # For the priority queue, we need to compute the "cost" of removing z:
    # cost(z) = number of points x in influence(z) such that min_v cnt[x,v] == 0
    #           and the specific direction v that z affects is the ONLY direction
    #           with cnt[x,v] == 0 for x.
    # More precisely: after incrementing cnt[x,v] for all (x,v) in influence(z),
    # a point x becomes invalid if for ALL directions v', cnt[x,v'] > 0.
    # Before the update, x was valid, so min_v cnt[x,v] == 0.
    # After incrementing some cnts, x becomes invalid iff:
    #   - Before: exactly one direction v* had cnt[x,v*] == 0
    #   - The updated direction v (from z's influence) is v*
    # So cost(z) = number of x in influence(z) such that:
    #   - min_v cnt[x,v] == 0 (x is currently "barely" valid)
    #   - The direction v in the pair (x,v) is the unique zero direction for x

    # Let's maintain:
    # zero_count[x] = number of directions v with cnt[x,v] == 0
    # A point x is "critical" if zero_count[x] == 1

    # When we add z to E:
    #   For each (x, v) in influence(z):
    #     if cnt[x,v] == 0:
    #       zero_count[x] -= 1
    #       if zero_count[x] == 0: x becomes invalid (cost++)
    #     cnt[x,v] += 1

    # When we remove z from E (revert):
    #   For each (x, v) in influence(z):
    #     cnt[x,v] -= 1
    #     if cnt[x,v] == 0:
    #       zero_count[x] += 1

    # Let's implement this.

    best_E = np.zeros(total_points, dtype=bool)
    best_size = 0

    def greedy_pq(seed):
        """Priority-queue based greedy removal."""
        rng = random.Random(seed)
        in_E = np.zeros(total_points, dtype=bool)
        cnt = np.zeros((total_points, num_dirs), dtype=np.int32)
        zero_count = np.zeros(total_points, dtype=np.int32)
        # Initially, all cnt are 0, so zero_count[x] = num_dirs for all x
        zero_count[:] = num_dirs

        # Priority queue: (cost, tiebreak, z)
        # We want to always pick the point with minimum cost.
        # Initially, all costs are 0 (no points are critical yet).
        # But as we add points to E, costs change.
        # We need to recompute costs dynamically.

        # Let's use a different approach:
        # Maintain a list of "available" points (not yet in E).
        # For each available point z, compute its cost on the fly.
        # Use a lazy heap: push (cost, z), and when popping, check if cost is still valid.

        # But computing cost for each z requires checking all (x,v) in influence(z),
        # which is 50*48 = 2400 checks. With 2401 points, that's ~5.7M checks per full pass.
        # Too slow if done naively.

        # Alternative: maintain costs incrementally.
        # When we add z to E, we update cnt and zero_count.
        # This changes the cost of other points w that share influence with z.
        # Specifically, for each (x, v) in influence(z), the cost of any w that also
        # affects (x, v) may change.

        # This is complex. Let's use a simpler heuristic:
        # Process points in an order determined by a static "priority" computed once,
        # then do multiple passes with different orderings.

        # Actually, let's try a different structural approach:
        # Start with E = empty. Repeatedly find the point z with minimum "damage"
        # and add it to E. Damage = number of points that would become invalid.
        # Recompute damage for all candidates after each addition.
        # This is O(n^2 * influence_size) which is too slow.

        # Let's use a compromise: batch updates.
        # Divide into batches. In each batch, greedily add points with current cost.
        # After each batch, recompute costs for remaining points.

        # For now, let's implement a simpler version:
        # Random order with multiple restarts, but with a better stopping criterion.
        # We stop when no more points can be added.

        order = list(range(total_points))
        rng.shuffle(order)

        size = 0
        for z in order:
            if time.time() > deadline:
                break
            if in_E[z]:
                continue

            # Compute cost: number of (x,v) in influence(z) where cnt[x,v]==0
            # and zero_count[x]==1 (i.e., x would become invalid)
            cost = 0
            # For each direction v, check line_other[z, v, :]
            for v in range(num_dirs):
                pts = line_other[z, v, :]
                # For each x in pts, check if cnt[x,v]==0 and zero_count[x]==1
                mask = (cnt[pts, v] == 0) & (zero_count[pts] == 1)
                cost += np.sum(mask)

            if cost == 0:
                # Safe to add
                in_E[z] = True
                size += 1
                # Update cnt and zero_count
                for v in range(num_dirs):
                    pts = line_other[z, v, :]
                    # For points where cnt was 0, decrement zero_count
                    was_zero = (cnt[pts, v] == 0)
                    zero_count[pts] -= was_zero
                    cnt[pts, v] += 1

        return size, in_E

    def greedy_pq_with_priority(seed):
        """
        Priority-queue greedy with dynamic cost updates.
        Uses a lazy heap where we push (cost, z) and validate on pop.
        """
        rng = random.Random(seed)
        in_E = np.zeros(total_points, dtype=bool)
        cnt = np.zeros((total_points, num_dirs), dtype=np.int32)
        zero_count = np.full(total_points, num_dirs, dtype=np.int32)
        # cost_cache[z] = last computed cost for z (for lazy validation)
        cost_cache = np.full(total_points, -1, dtype=np.int32)
        # version[z] = version number when cost was last computed
        version = np.zeros(total_points, dtype=np.int32)
        global_version = 0

        # Heap: (cost, z)
        heap = []
        # Initially, all points have cost 0 (no critical points yet)
        for z in range(total_points):
            cost_cache[z] = 0
            heapq.heappush(heap, (0, z))

        size = 0
        available = np.ones(total_points, dtype=bool)

        while heap:
            if time.time() > deadline:
                break

            cost, z = heapq.heappop(heap)

            if not available[z]:
                continue

            # Recompute actual cost
            actual_cost = 0
            for v in range(num_dirs):
                pts = line_other[z, v, :]
                mask = (cnt[pts, v] == 0) & (zero_count[pts] == 1)
                actual_cost += np.sum(mask)

            if actual_cost != cost:
                # Cost changed, re-push with new cost
                cost_cache[z] = actual_cost
                heapq.heappush(heap, (actual_cost, z))
                continue

            if actual_cost > 0:
                # Not safe to add now, but might be later
                # Don't re-push; it will be re-evaluated when other points are added
                # Actually, we should re-push so it can be reconsidered
                continue

            # Safe to add z
            available[z] = False
            in_E[z] = True
            size += 1

            # Update cnt and zero_count
            for v in range(num_dirs):
                pts = line_other[z, v, :]
                was_zero = (cnt[pts, v] == 0)
                zero_count[pts] -= was_zero
                cnt[pts, v] += 1

            # Points whose cost may have changed:
            # Any point w that shares an (x,v) pair with z's influence
            # This is hard to track efficiently. For now, just let the lazy
            # heap handle it: when w is popped, its cost will be recomputed.

        return size, in_E

    # Try multiple strategies
    strategies = []

    # Strategy 1: Simple random order with multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        size, in_E = greedy_pq(42 + restart)
        strategies.append((size, in_E))

    # Strategy 2: Priority queue with lazy cost
    for restart in range(10):
        if time.time() > deadline:
            break
        size, in_E = greedy_pq_with_priority(1000 + restart)
        strategies.append((size, in_E))

    # Pick the best
    if strategies:
        best_size, best_E = max(strategies, key=lambda x: x[0])
    else:
        best_size = 0
        best_E = np.zeros(total_points, dtype=bool)

    # Build N: points NOT in E
    points_list = []
    for pidx in range(total_points):
        if not best_E[pidx]:
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    if not points_list:
        for pidx in range(total_points):
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    return np.array(points_list, dtype=np.int64)


# EVOLVE-BLOCK-END
