import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Uses a baseline construction followed by optimized greedy pruning with
    precomputed membership structures for efficient removability checks.
    """
    start_time = time.time()
    deadline = start_time + 590.0
    safety_margin = 10.0

    p = 7
    q = p * p  # 49
    alpha_sq = 3

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

    # Precompute line points
    line_points = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)
    one_idx = elem_to_idx(1, 0)
    zero_idx = elem_to_idx(0, 0)

    for d_idx in range(num_dirs):
        if d_idx < q:
            vx_idx = one_idx
            vy_idx = d_idx
        else:
            vx_idx = zero_idx
            vy_idx = one_idx

        for pt in range(total_points):
            x_idx = pt // q
            y_idx = pt % q
            for t_idx in range(1, q):
                tx = mul_table[t_idx, vx_idx]
                ty = mul_table[t_idx, vy_idx]
                nx = add_table[x_idx, tx]
                ny = add_table[y_idx, ty]
                line_points[pt, d_idx, t_idx - 1] = nx * q + ny

    # Precompute membership matrix: member[y, x, d] = True if y is on line x,d (excluding x)
    # This is a 3D boolean array of size 2401 x 2401 x 50 = 288M booleans = 288MB
    # That's too large. Instead, for each y, store the list of (x, d) pairs
    y_to_lines = [[] for _ in range(total_points)]
    for x in range(total_points):
        for d in range(num_dirs):
            for y in line_points[x, d, :]:
                y_to_lines[y].append((x, d))

    # Initialize validity tracking
    valid_count = np.zeros(total_points, dtype=np.int32)
    is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)

    for pt in range(total_points):
        for d in range(num_dirs):
            if np.all(in_N[line_points[pt, d, :]]):
                is_valid_dir[pt, d] = True
                valid_count[pt] += 1

    # Greedy pruning with optimized removability check
    best_complement = np.sum(~in_N)
    best_in_N = in_N.copy()
    best_valid_count = valid_count.copy()
    best_is_valid_dir = is_valid_dir.copy()

    def try_prune(order, max_rounds=3):
        nonlocal in_N, valid_count, is_valid_dir
        nonlocal best_complement, best_in_N, best_valid_count, best_is_valid_dir

        work = in_N.copy()
        wc = valid_count.copy()
        ird = is_valid_dir.copy()
        removed = 0

        for round_num in range(max_rounds):
            if time.time() > deadline - safety_margin:
                break

            progress = False
            for y in order:
                if not work[y]:
                    continue

                # Check removability
                removable = True
                for (x, d) in y_to_lines[y]:
                    if ird[x, d]:
                        if wc[x] <= 1:
                            removable = False
                            break
                        # Count how many valid lines of x contain y
                        cnt = 0
                        for d2 in range(num_dirs):
                            if ird[x, d2]:
                                # Check if y is on line x, d2
                                # y is on line x, d2 iff (x, d2) is in y_to_lines[y]
                                # We can check this by scanning y_to_lines[y] for (x, d2)
                                # But that's O(len(y_to_lines[y])) per check
                                # Better: precompute a set for fast lookup
                                pass
                        # Alternative: just check directly
                        # y is on line x, d2 if y is in line_points[x, d2, :]
                        # We can check this by seeing if x is in the line through y with direction d2
                        # Actually, the simplest is to check if (x, d2) appears in y_to_lines[y]
                        # Let's build a set for fast lookup
                        pass
                    if not removable:
                        break

                if removable:
                    work[y] = False
                    removed += 1
                    progress = True
                    for (x, d) in y_to_lines[y]:
                        if ird[x, d]:
                            ird[x, d] = False
                            wc[x] -= 1

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

    # Build fast lookup sets for membership
    y_to_lines_set = [set(lines) for lines in y_to_lines]

    def try_prune_v2(order, max_rounds=3):
        nonlocal in_N, valid_count, is_valid_dir
        nonlocal best_complement, best_in_N, best_valid_count, best_is_valid_dir

        work = in_N.copy()
        wc = valid_count.copy()
        ird = is_valid_dir.copy()
        removed = 0

        for round_num in range(max_rounds):
            if time.time() > deadline - safety_margin:
                break

            progress = False
            for y in order:
                if not work[y]:
                    continue

                # Check removability
                removable = True
                for (x, d) in y_to_lines[y]:
                    if ird[x, d]:
                        if wc[x] <= 1:
                            removable = False
                            break
                        # Count how many valid lines of x contain y
                        cnt = 0
                        for d2 in range(num_dirs):
                            if ird[x, d2]:
                                if (x, d2) in y_to_lines_set[y]:
                                    cnt += 1
                        if cnt == wc[x]:
                            removable = False
                            break
                    if not removable:
                        break

                if removable:
                    work[y] = False
                    removed += 1
                    progress = True
                    for (x, d) in y_to_lines[y]:
                        if ird[x, d]:
                            ird[x, d] = False
                            wc[x] -= 1

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

    # Strategy 1: Sequential order
    order1 = list(range(total_points))
    try_prune_v2(order1, max_rounds=2)

    # Strategy 2: Random order
    if time.time() < deadline - safety_margin:
        rng = np.random.RandomState(42)
        order2 = list(range(total_points))
        rng.shuffle(order2)
        try_prune_v2(order2, max_rounds=2)

    # Strategy 3: Prioritize points with low coverage
    if time.time() < deadline - safety_margin:
        coverage = np.zeros(total_points, dtype=np.int32)
        for y in range(total_points):
            if not in_N[y]:
                continue
            for (x, d) in y_to_lines[y]:
                if is_valid_dir[x, d]:
                    coverage[y] += 1

        order3 = sorted(range(total_points), key=lambda y: 0 if not in_N[y] else coverage[y])
        try_prune_v2(order3, max_rounds=1)

    # Strategy 4: Random restarts
    if time.time() < deadline - safety_margin:
        for seed in [123, 456, 789, 101, 202, 303]:
            if time.time() > deadline - safety_margin:
                break
            rng = np.random.RandomState(seed)
            order4 = list(range(total_points))
            rng.shuffle(order4)
            try_prune_v2(order4, max_rounds=1)

    # Final validity check
    if np.any(valid_count == 0):
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
