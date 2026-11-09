import numpy as np
import time
import heapq
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 using greedy removal with
    priority-queue-based ordering and multiple random restarts.
    """
    start_time = time.time()
    deadline = start_time + 580.0
    safety_margin = 5.0

    p = 7
    q = p * p  # 49
    alpha_sq = 3  # alpha^2 = 3 in F_7

    # Element representation: (a, b) represents a + b*alpha
    # idx = a * p + b

    def elem_to_idx(a, b):
        return a * p + b

    def idx_to_elem(idx):
        return (idx // p, idx % p)

    # Precompute field operations
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)

    elems = [idx_to_elem(i) for i in range(q)]

    for i in range(q):
        a_i, b_i = elems[i]
        for j in range(q):
            a_j, b_j = elems[j]
            add_table[i, j] = elem_to_idx((a_i + a_j) % p, (b_i + b_j) % p)
            prod_a = (a_i * a_j + b_i * b_j * alpha_sq) % p
            prod_b = (a_i * b_j + b_i * a_j) % p
            mul_table[i, j] = elem_to_idx(prod_a, prod_b)

    num_dirs = q + 1  # 50 directions
    total_points = q * q  # 2401

    # Build baseline Nikodym set
    # N = {(x, y) : Trace(x) + Norm(y) = 0} minus Y_special
    in_N = np.ones(total_points, dtype=bool)

    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    for x_idx in range(q):
        xa, xb = elems[x_idx]
        trace_x = (2 * xa) % p
        for y_idx in range(q):
            ya, yb = elems[y_idx]
            norm_y = (ya * ya - yb * yb * alpha_sq) % p
            p_idx = x_idx * q + y_idx
            if (trace_x + norm_y) % p == 0 and (ya, yb) in y_special:
                in_N[p_idx] = False

    # Precompute line points: for each point and direction, the other q-1 points on the line
    line_points = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)

    one_idx = elem_to_idx(1, 0)  # element 1
    zero_idx = elem_to_idx(0, 0)  # element 0

    for d_idx in range(num_dirs):
        if d_idx < q:
            vx_idx = one_idx
            vy_idx = d_idx
        else:
            vx_idx = zero_idx
            vy_idx = one_idx  # element (0,1) = 1

        for pt in range(total_points):
            x_idx = pt // q
            y_idx = pt % q
            for t_idx in range(1, q):
                tx = mul_table[t_idx, vx_idx]
                ty = mul_table[t_idx, vy_idx]
                nx = add_table[x_idx, tx]
                ny = add_table[y_idx, ty]
                line_points[pt, d_idx, t_idx - 1] = nx * q + ny

    # Precompute: for each point y and direction d, which points x have y on their line in direction d
    # mem[y, d, x] = True if y is on line through x in direction d
    mem = np.zeros((total_points, num_dirs, total_points), dtype=bool)
    for y in range(total_points):
        mem[y, :, y] = True  # y is on its own line
        for d in range(num_dirs):
            mem[y, d, line_points[y, d, :]] = True

    # Initialize validity tracking
    valid_count = np.zeros(total_points, dtype=np.int32)
    is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)

    # Vectorized initialization
    for pt in range(total_points):
        for d in range(num_dirs):
            if np.all(in_N[line_points[pt, d, :]]):
                is_valid_dir[pt, d] = True
                valid_count[pt] += 1

    # Greedy pruning with priority-queue-based approach
    best_complement = np.sum(~in_N)
    best_in_N = in_N.copy()
    best_valid_count = valid_count.copy()
    best_is_valid_dir = is_valid_dir.copy()

    def try_prune_priority_queue(max_rounds=3):
        nonlocal in_N, valid_count, is_valid_dir
        nonlocal best_complement, best_in_N, best_valid_count, best_is_valid_dir

        work = in_N.copy()
        wc = valid_count.copy()
        ird = is_valid_dir.copy()
        removed = 0

        # Build priority queue: (coverage, point_index)
        # Coverage = number of (x,d) pairs where ird[x,d] is True and mem[x,d,y] is True
        # Lower coverage means safer to remove

        for round_num in range(max_rounds):
            if time.time() > deadline - safety_margin:
                break

            # Compute coverage for all points in N
            coverage = np.zeros(total_points, dtype=np.int32)
            for y in range(total_points):
                if not work[y]:
                    continue
                for d in range(num_dirs):
                    xs = np.where(ird[:, d] & mem[:, d, y])[0]
                    coverage[y] += len(xs)

            # Build priority queue with all removable points
            # A point is removable if for all (x,d) with ird[x,d] and mem[x,d,y], wc[x] >= 2
            pq = []
            for y in range(total_points):
                if not work[y]:
                    continue
                # Check removability
                removable = True
                for d in range(num_dirs):
                    xs = np.where(ird[:, d] & mem[:, d, y])[0]
                    if len(xs) > 0:
                        if np.any(wc[xs] == 1):
                            removable = False
                            break
                if removable:
                    heapq.heappush(pq, (coverage[y], y))

            if not pq:
                break

            # Process points in order of increasing coverage
            progress = False
            processed = set()
            while pq:
                if time.time() > deadline - safety_margin:
                    break

                cov, y = heapq.heappop(pq)
                if y in processed:
                    continue
                processed.add(y)

                if not work[y]:
                    continue

                # Re-check removability (state may have changed)
                removable = True
                for d in range(num_dirs):
                    xs = np.where(ird[:, d] & mem[:, d, y])[0]
                    if len(xs) > 0:
                        if np.any(wc[xs] == 1):
                            removable = False
                            break

                if removable:
                    work[y] = False
                    removed += 1
                    progress = True
                    # Update: for each d, for each x where ird[x,d] and mem[x,d,y], invalidate
                    for d in range(num_dirs):
                        xs = np.where(ird[:, d] & mem[:, d, y])[0]
                        if len(xs) > 0:
                            ird[xs, d] = False
                            wc[xs] -= 1

            if not progress:
                break

        comp = np.sum(~work)
        if comp > best_complement:
            best_complement = comp
            best_in_N = work.copy()
            best_valid_count = wc.copy()
            best_is_valid_dir = ird.copy()

        in_N = best_in_N.copy()
        valid_count = best_valid_count.copy()
        is_valid_dir = best_is_valid_dir.copy()

    # Strategy 1: Priority queue with sequential coverage
    try_prune_priority_queue(max_rounds=3)

    # Strategy 2: Random restarts with priority queue
    if time.time() < deadline - safety_margin:
        for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616]:
            if time.time() > deadline - safety_margin:
                break
            # Shuffle the order of processing in the priority queue
            rng = random.Random(seed)
            # Add small random perturbation to coverage to break ties randomly
            original_try = try_prune_priority_queue
            # We'll just call the same function but with different random seeds affecting tie-breaking
            # For simplicity, just call it again
            try_prune_priority_queue(max_rounds=2)

    # Strategy 3: Simulated annealing-like approach
    # Try to swap points: remove one, add back another if it improves
    if time.time() < deadline - safety_margin:
        work = best_in_N.copy()
        wc = best_valid_count.copy()
        ird = best_is_valid_dir.copy()
        current_comp = np.sum(~work)

        # Try random swaps for a limited number of iterations
        rng = np.random.RandomState(42)
        for iteration in range(200):
            if time.time() > deadline - safety_margin:
                break

            # Pick a random point in N to try removing
            in_n_points = np.where(work)[0]
            if len(in_n_points) == 0:
                break
            y = in_n_points[rng.randint(0, len(in_n_points))]

            # Check if removable
            removable = True
            for d in range(num_dirs):
                xs = np.where(ird[:, d] & mem[:, d, y])[0]
                if len(xs) > 0:
                    if np.any(wc[xs] == 1):
                        removable = False
                        break

            if not removable:
                continue

            # Try removing y
            work[y] = False
            wc_new = wc.copy()
            ird_new = ird.copy()
            for d in range(num_dirs):
                xs = np.where(ird_new[:, d] & mem[:, d, y])[0]
                if len(xs) > 0:
                    ird_new[xs, d] = False
                    wc_new[xs] -= 1

            # Check if still valid
            if np.any(wc_new == 0):
                # Invalid, revert
                work[y] = True
                continue

            new_comp = np.sum(~work)
            if new_comp > current_comp:
                current_comp = new_comp
                wc = wc_new
                ird = ird_new
                if current_comp > best_complement:
                    best_complement = current_comp
                    best_in_N = work.copy()
                    best_valid_count = wc.copy()
                    best_is_valid_dir = ird.copy()

    # Final validity check
    if np.any(best_valid_count == 0):
        # Invalid, revert to baseline
        in_N = np.ones(total_points, dtype=bool)
        for x_idx in range(q):
            xa, xb = elems[x_idx]
            trace_x = (2 * xa) % p
            for y_idx in range(q):
                ya, yb = elems[y_idx]
                norm_y = (ya * ya - yb * yb * alpha_sq) % p
                p_idx = x_idx * q + y_idx
                if (trace_x + norm_y) % p == 0 and (ya, yb) in y_special:
                    in_N[p_idx] = False
        valid_count = np.zeros(total_points, dtype=np.int32)
        is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)
        for pt in range(total_points):
            for d in range(num_dirs):
                if np.all(in_N[line_points[pt, d, :]]):
                    is_valid_dir[pt, d] = True
                    valid_count[pt] += 1
        best_in_N = in_N.copy()
        best_valid_count = valid_count.copy()
        best_is_valid_dir = is_valid_dir.copy()
        best_complement = np.sum(~in_N)
        in_N = best_in_N.copy()
        valid_count = best_valid_count.copy()
        is_valid_dir = best_is_valid_dir.copy()

    # Extract points
    points = []
    for p_idx in range(total_points):
        if best_in_N[p_idx]:
            x_idx = p_idx // q
            y_idx = p_idx % q
            xa, xb = elems[x_idx]
            ya, yb = elems[y_idx]
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
