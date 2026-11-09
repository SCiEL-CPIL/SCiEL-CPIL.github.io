import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Uses a hypergraph vertex-cover formulation solved via a greedy algorithm
    optimized with vectorized membership checks to allow for extensive random
    restarts and deeper pruning within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 600s
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

    # Optimization: Precompute a boolean matrix `on_line`
    # on_line[pt, d, other_pt] is True if other_pt is on the line through pt with direction d
    # Shape: (total_points, num_dirs, total_points)
    # Memory: 2401 * 50 * 2401 * 1 byte ~ 288 MB. This is acceptable.
    on_line = np.zeros((total_points, num_dirs, total_points), dtype=bool)
    for pt in range(total_points):
        for d in range(num_dirs):
            pts_on_line = line_points[pt, d, :]
            on_line[pt, d, pts_on_line] = True

    # Initialize validity tracking
    valid_count = np.zeros(total_points, dtype=np.int32)
    is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)

    for pt in range(total_points):
        for d in range(num_dirs):
            if np.all(in_N[line_points[pt, d, :]]):
                is_valid_dir[pt, d] = True
                valid_count[pt] += 1

    # Greedy pruning with multiple strategies
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

                # Check removability: for all x connected to y,
                # y must not be on ALL valid lines of x
                removable = True
                for d in range(num_dirs):
                    # Get all points x such that y is on the line through x with direction d
                    # This is equivalent to: for all x in line_points[y, d, :], check if y is on line through x with direction d
                    # Note: The line through x with direction d contains y iff the line through y with direction d contains x.
                    # So we can just check: is x in line_points[y, d, :]?
                    # Wait, the condition is: 
                    # "y must not be on ALL valid lines of x"
                    # This means: It is NOT the case that (for all d2 where ird[x, d2] is True, y is on line x-d2).
                    # Equivalently: There exists a d2 such that ird[x, d2] is True AND y is NOT on line x-d2.
                    # OR: The count of d2 where (ird[x, d2] AND y is on line x-d2) is less than wc[x].
                    
                    # Let's iterate over x in the neighborhood of y.
                    # For a fixed y and d, the points x that are connected to y via direction d are exactly line_points[y, d, :].
                    # For each such x, we need to check if y is on ALL valid lines of x.
                    # y is on line x-d2 iff x is in line_points[y, d2, :]? No.
                    # y is on line x-d2 iff on_line[x, d2, y] is True.
                    
                    # To make this fast, we can precompute for each y, a bitmask of which (x, d2) pairs are "blocking".
                    # But let's just use the on_line matrix.
                    
                    # For each x in line_points[y, d, :]:
                    #   if ird[x, d]: 
                    #      # Check if y is on all valid lines of x
                    #      # This is true if: sum(on_line[x, :, y] & ird[x, :]) == wc[x]
                    #      # If wc[x] == 0, it's trivially true (but wc[x]>0 if ird has any True).
                    #      # If wc[x] > 0 and the sum is wc[x], then y is on all valid lines of x -> NOT removable.
                    
                    # We can vectorize this check for all x in line_points[y, d, :] at once?
                    # Let X = line_points[y, d, :] (size 48)
                    # We need to check for each x in X:
                    #   if ird[x, d]:
                    #       if np.sum(on_line[x, :, y] & ird[x, :]) == wc[x]:
                    #           removable = False
                    #
                    # This inner loop over X is small (48). The operation on_line[x, :, y] is a slice of size 50.
                    # This is much faster than `y in line_points[x, d2, :]`.
                    
                    for x in line_points[y, d, :]:
                        if ird[x, d]:
                            # Check if y is on all valid lines of x
                            if wc[x] <= 0:
                                continue # Should not happen if ird is consistent
                            # Count valid lines of x that contain y
                            # on_line[x, :, y] is a boolean array of size num_dirs
                            # ird[x, :] is a boolean array of size num_dirs
                            cnt = np.sum(on_line[x, :, y] & ird[x, :])
                            if cnt == wc[x]:
                                removable = False
                                break
                    if not removable:
                        break

                if removable:
                    work[y] = False
                    removed += 1
                    progress = True
                    # Update validity for all x that are on lines through y
                    # For each d, for each x in line_points[y, d, :]:
                    #   if ird[x, d]:
                    #      ird[x, d] = False
                    #      wc[x] -= 1
                    for d in range(num_dirs):
                        xs = line_points[y, d, :]
                        valid_xs = xs[ird[xs, d]]
                        if len(valid_xs) > 0:
                            ird[valid_xs, d] = False
                            wc[valid_xs] -= 1

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
    try_prune(order1, max_rounds=2)

    # Strategy 2: Random restarts with more iterations
    # Since the inner loop is faster, we can afford more restarts and rounds
    if time.time() < deadline - safety_margin:
        rng = np.random.RandomState(42)
        order2 = list(range(total_points))
        rng.shuffle(order2)
        try_prune(order2, max_rounds=3)

    # Strategy 3: Prioritize points with low "coverage"
    if time.time() < deadline - safety_margin:
        coverage = np.zeros(total_points, dtype=np.int32)
        for y in range(total_points):
            if not in_N[y]:
                continue
            for d in range(num_dirs):
                for x in line_points[y, d, :]:
                    if is_valid_dir[x, d]:
                        coverage[y] += 1

        order3 = sorted(range(total_points), key=lambda y: 0 if not in_N[y] else coverage[y])
        try_prune(order3, max_rounds=2)

    # Strategy 4: Many random restarts
    # Use the remaining time to run as many random restarts as possible
    seeds = [123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333]
    for seed in seeds:
        if time.time() > deadline - safety_margin:
            break
        rng = np.random.RandomState(seed)
        order4 = list(range(total_points))
        rng.shuffle(order4)
        # Use 2 rounds for each restart, as it's fast now
        try_prune(order4, max_rounds=2)

    # Final validity check
    if np.any(valid_count == 0):
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
