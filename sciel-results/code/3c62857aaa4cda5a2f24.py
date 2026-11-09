import numpy as np
import time
import sys

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0
    safety_margin = 10.0

    p = 7
    q = p * p
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

    num_dirs = q + 1
    total_points = q * q

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

    # Initialize validity tracking
    valid_count = np.zeros(total_points, dtype=np.int32)
    is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)

    for d in range(num_dirs):
        all_in = np.all(in_N[line_points[:, d, :]], axis=1)
        is_valid_dir[:, d] = all_in
        valid_count += all_in.astype(np.int32)

    # Helper to check if a point can be removed
    def can_remove(y, work, wc, ird, line_pts):
        """Check if point y can be removed from N while maintaining validity."""
        for d in range(num_dirs):
            xs = line_pts[y, d, :]
            for x in xs:
                if not work[x]:
                    continue
                if ird[x, d]:
                    # x uses line d through y as witness
                    # Check if x has another witness
                    if wc[x] <= 1:
                        return False
                    # Check if y is on a valid line through x
                    # If y is the ONLY valid direction for x, we can't remove y
                    # Actually we need to check: after removing y, does x still have witness d?
                    # x loses witness d if y is the only point on L_d(x) \ {x} that's not in N
                    # Wait, the condition is: L_d(x) \ {x} must be subset of N
                    # If we remove y, and y is on L_d(x) \ {x}, then L_d(x) \ {x} contains y which is not in N
                    # So x loses witness d
                    # x remains valid if it has another witness
                    if wc[x] == 1 and ird[x, d]:
                        return False
                    # More precisely: x loses witness d. x is still valid if it has another valid direction
                    # wc[x] counts valid directions. If wc[x] > 1, x still has another
                    # But we need to be careful: wc[x] might count d, and after removal d is gone
                    # So if wc[x] == 1, x becomes invalid
                    # If wc[x] > 1, x still has at least one other valid direction
                    pass
        
        return True

    def try_remove(y, work, wc, ird, line_pts):
        """Try to remove point y. Return True if successful."""
        if not work[y]:
            return False
        
        # Check removability
        for d in range(num_dirs):
            xs = line_pts[y, d, :]
            for x in xs:
                if not work[x]:
                    continue
                if ird[x, d]:
                    if wc[x] <= 1:
                        return False
        
        # Perform removal
        work[y] = False
        for d in range(num_dirs):
            xs = line_pts[y, d, :]
            for x in xs:
                if work[x] and ird[x, d]:
                    ird[x, d] = False
                    wc[x] -= 1
        return True

    def greedy_prune(order, max_rounds, work, wc, ird, line_pts):
        """Greedy pruning in given order for max_rounds."""
        removed = 0
        for round_num in range(max_rounds):
            if time.time() > deadline - safety_margin:
                break
            progress = False
            for y in order:
                if not work[y]:
                    continue
                if try_remove(y, work, wc, ird, line_pts):
                    removed += 1
                    progress = True
            if not progress:
                break
        return removed

    best_complement = np.sum(~in_N)
    best_in_N = in_N.copy()
    best_wc = valid_count.copy()
    best_ird = is_valid_dir.copy()

    # Strategy 1: Sequential order
    work = in_N.copy()
    wc = valid_count.copy()
    ird = is_valid_dir.copy()
    greedy_prune(list(range(total_points)), 3, work, wc, ird, line_points)
    comp = np.sum(~work)
    if comp > best_complement:
        best_complement = comp
        best_in_N = work.copy()
        best_wc = wc.copy()
        best_ird = ird.copy()

    # Strategy 2: Random orders
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        if time.time() > deadline - safety_margin:
            break
        rng = np.random.RandomState(seed)
        work = best_in_N.copy()
        wc = best_wc.copy()
        ird = best_ird.copy()
        order = list(range(total_points))
        rng.shuffle(order)
        greedy_prune(order, 2, work, wc, ird, line_points)
        comp = np.sum(~work)
        if comp > best_complement:
            best_complement = comp
            best_in_N = work.copy()
            best_wc = wc.copy()
            best_ird = ird.copy()

    # Strategy 3: Coverage-based ordering
    if time.time() < deadline - safety_margin:
        for pass_num in range(5):
            if time.time() > deadline - safety_margin:
                break
            work = best_in_N.copy()
            wc = best_wc.copy()
            ird = best_ird.copy()
            
            # Compute coverage: for each point, how many valid (x,d) pairs are affected
            coverage = np.zeros(total_points, dtype=np.int32)
            for y in range(total_points):
                if not work[y]:
                    continue
                for d in range(num_dirs):
                    xs = line_points[y, d, :]
                    valid_xs = ird[xs, d]
                    coverage[y] += np.sum(valid_xs)
            
            active = [y for y in range(total_points) if work[y]]
            active.sort(key=lambda y: coverage[y])
            greedy_prune(active, 2, work, wc, ird, line_points)
            comp = np.sum(~work)
            if comp > best_complement:
                best_complement = comp
                best_in_N = work.copy()
                best_wc = wc.copy()
                best_ird = ird.copy()

    # Final validity check
    if np.any(best_wc == 0):
        # Fallback to baseline
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
        for d in range(num_dirs):
            all_in = np.all(in_N[line_points[:, d, :]], axis=1)
            is_valid_dir[:, d] = all_in
            valid_count += all_in.astype(np.int32)
        best_in_N = in_N.copy()
        best_wc = valid_count.copy()
        best_ird = is_valid_dir.copy()
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
