import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Returns an array of points in N, where N is the Nikodym set.
    Uses baseline construction followed by aggressive greedy pruning.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 10.0  # Keep a larger safety margin to ensure we return

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
    # We want to maximize complement, so start with all True (in N) and remove some.
    # Actually, the definition of Nikodym set N requires that for every x, there is a line L_x containing x such that L_x \ {x} subset N.
    # A standard construction for F_q^2 is to take a set of lines and include all points on them? No, that's large.
    # The provided code uses: in_N[p_idx] = False if (trace_x + norm_y) == 0 and (ya, yb) in y_special.
    # This means in_N is True everywhere EXCEPT for those specific points.
    # So N is almost the entire space.
    
    in_N = np.ones(total_points, dtype=bool)
    
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
            
    # Vectorize baseline construction
    # x_idx ranges 0..48, y_idx ranges 0..48
    # p_idx = x_idx * q + y_idx
    # Condition: (Trace(x) + Norm(y)) % p == 0 AND (ya, yb) in y_special
    
    # Precompute trace and norm for all elements
    trace_arr = np.zeros(q, dtype=np.int64)
    norm_arr = np.zeros(q, dtype=np.int64)
    for i in range(q):
        a, b = elems[i]
        trace_arr[i] = (2 * a) % p
        norm_arr[i] = (a * a - b * b * alpha_sq) % p
    
    # Create boolean mask for y_special
    y_special_mask = np.zeros(q, dtype=bool)
    for ys in y_special:
        idx = elem_to_idx(ys[0], ys[1])
        y_special_mask[idx] = True
        
    # Broadcast to check condition for all pairs
    # trace_arr[:, None] + norm_arr[None, :]
    sum_arr = (trace_arr[:, None] + norm_arr[None, :]) % p
    condition = (sum_arr == 0) & y_special_mask[None, :]
    
    # Flatten to 1D
    condition_flat = condition.flatten()
    in_N[condition_flat] = False

    # Precompute line points
    # line_points[pt, d, t_idx] = index of point at distance t_idx along direction d from pt
    # We only need t_idx 1..48 for the "other than x" part
    line_points = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)
    
    one_idx = elem_to_idx(1, 0)  # element 1
    zero_idx = elem_to_idx(0, 0)  # element 0
    
    # Directions
    # d < q: v = (1, d)
    # d == q: v = (0, 1)
    
    # We can compute this faster by iterating directions and using vectorized additions
    # For a fixed direction d, the line through pt is pt + t*v_d.
    # Since we need line_points for ALL pt, we can compute offsets.
    
    # Let's stick to the loop but optimize the inner part if possible.
    # Actually, 2401 * 50 * 48 is ~5.7 million operations. This is fine in Python if not heavily nested with logic.
    # But we can precompute the "offset" indices for each direction.
    
    # offset[d, t] = index of element t * v_d? No, v_d is a pair of field elements.
    # Let's just use the loop, it's fast enough.
    
    for d_idx in range(num_dirs):
        if d_idx < q:
            vx_idx = one_idx
            vy_idx = d_idx
        else:
            vx_idx = zero_idx
            vy_idx = one_idx  # element (0,1) = 1
            
        # Precompute t*v for all t in 1..48
        txs = np.array([mul_table[t_idx, vx_idx] for t_idx in range(1, q)])
        tys = np.array([mul_table[t_idx, vy_idx] for t_idx in range(1, q)])
        
        # For each point pt, line_points[pt, d_idx, :] = pt + t*v
        # pt is (x, y). x_idx = pt // q, y_idx = pt % q
        # We can vectorize over pt?
        # x_indices = np.arange(total_points) // q
        # y_indices = np.arange(total_points) % q
        # new_x = add_table[x_indices[:, None], txs[None, :]] -> shape (2401, 48)
        # new_y = add_table[y_indices[:, None], tys[None, :]] -> shape (2401, 48)
        # line_points[:, d_idx, :] = new_x * q + new_y
        
        x_indices = np.arange(total_points) // q
        y_indices = np.arange(total_points) % q
        
        # add_table is (49, 49). x_indices is (2401,), txs is (48,)
        # We need add_table[x_indices[:, None], txs[None, :]]
        # This creates a (2401, 48) array.
        
        # To save memory/time, we can do this in chunks or just let numpy handle it.
        # 2401 * 48 * 8 bytes * 2 (x and y) ~ 18 MB. Acceptable.
        
        nx = add_table[x_indices[:, None], txs[None, :]]
        ny = add_table[y_indices[:, None], tys[None, :]]
        line_points[:, d_idx, :] = nx * q + ny

    # Initialize validity tracking
    # is_valid_dir[pt, d] = True if all points on line (excluding pt) are in N
    # valid_count[pt] = number of valid directions for pt
    
    # We can compute this efficiently.
    # For each direction d, we have line_points[:, d, :] which is (2401, 48).
    # We need to check if in_N[line_points[:, d, :]] is all True.
    
    is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)
    valid_count = np.zeros(total_points, dtype=np.int32)
    
    for d in range(num_dirs):
        # line_points[:, d, :] is (2401, 48)
        # in_N is (2401,)
        # We need in_N[line_points[:, d, :]] -> (2401, 48) boolean
        # Check if all are True along axis 1
        
        # This might be memory heavy: 2401 * 48 booleans ~ 115 KB per direction. Fine.
        mask = in_N[line_points[:, d, :]]
        valid_pts = np.all(mask, axis=1)
        is_valid_dir[:, d] = valid_pts
        valid_count += valid_pts.astype(np.int32)

    # Precompute is_on_line[pt, d, other_pt] = True if other_pt is on the line through pt with direction d (and other_pt != pt)
    # This allows O(1) check for "is y on line x,d"
    # Size: 2401 * 50 * 2401 booleans ~ 288 MB. This is large but manageable if we use bool (1 byte).
    # 2401 * 50 * 2401 = 288,240,050 bytes ~ 288 MB.
    # This might be too much for some environments or slow to allocate.
    # Alternative: Store line_points and use searchsorted? No, line_points is unsorted.
    # Alternative: For each point y, determine which (x, d) pairs have y on their line.
    # Since lines are defined by x and d, y is on the line if y = x + t*v_d for some t.
    # x = y - t*v_d.
    # So for a fixed y and d, the points x such that y is on the line through x with direction d are:
    # x = y - t*v_d for t in 1..48. (Since t=0 gives x=y, and we exclude x from the line check for x's validity? No, the line for x excludes x. So y must be one of the 48 other points.)
    
    # So, for each y, and each d, we can compute the 48 points x that are "covered" by y for direction d.
    # Let's precompute `covers[y, d]` = array of 48 x indices.
    # Then, if we remove y, we need to update is_valid_dir[x, d] for x in covers[y, d].
    # Specifically, if y was in N, and we remove it, any line that contained y is no longer fully in N.
    # So is_valid_dir[x, d] becomes False for all x in covers[y, d] where the line x-d contained y.
    # Wait, the line x-d contains y if y = x + t*v_d.
    # This is exactly x = y - t*v_d.
    
    # So `covers[y, d]` contains the x's.
    # If we remove y, then for each x in covers[y, d], the line x-d is no longer valid (because it contained y which is no longer in N).
    # So we set is_valid_dir[x, d] = False.
    # And decrement valid_count[x].
    
    # This is much faster than checking "is y on line x-d" on the fly.
    
    covers = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)
    
    for d in range(num_dirs):
        if d < q:
            vx_idx = one_idx
            vy_idx = d
        else:
            vx_idx = zero_idx
            vy_idx = one_idx
            
        # We need x = y - t*v_d
        # x_idx = y_idx - t*v_x_idx? No, field subtraction.
        # x = y + (-t)*v
        # Let's compute (-t)*v for t in 1..48.
        # Or just use the fact that x is on line y-d if y is on line x-d.
        # The set of x such that y is on line x-d is the same as the set of points on the line through y with direction d, excluding y.
        # So covers[y, d] should be the points on the line through y with direction d, excluding y.
        # This is exactly line_points[y, d, :]!
        
        # Let's verify:
        # line_points[pt, d, :] are the points on the line through pt with direction d, excluding pt.
        # If y is one of these points for pt=x, then y is on the line through x.
        # So covers[y, d] is NOT line_points[y, d, :].
        # covers[y, d] is the set of x such that y in line_points[x, d, :].
        
        # Since lines are symmetric in the sense that if y is on line x-d, then x is on line y-d (just with t' = -t),
        # the set of x such that y is on line x-d is exactly the set of points on the line through y with direction d, excluding y.
        # So covers[y, d] == line_points[y, d, :].
        
        # Let's double check.
        # Line L_x = {x + t v | t in F}.
        # y in L_x \ {x} iff y = x + t v for some t != 0.
        # x = y - t v = y + (-t) v.
        # Since t != 0, -t != 0.
        # So x is in {y + t' v | t' != 0} = L_y \ {y}.
        # Yes, covers[y, d] is exactly line_points[y, d, :].
        
        covers[:, d, :] = line_points[:, d, :]

    # Now we can do efficient pruning.
    # State:
    # in_N (bool, 2401)
    # is_valid_dir (bool, 2401, 50)
    # valid_count (int, 2401)
    
    # Best solution tracking
    best_complement = np.sum(~in_N)
    best_in_N = in_N.copy()
    best_is_valid_dir = is_valid_dir.copy()
    best_valid_count = valid_count.copy()

    def try_prune(order, max_rounds=5):
        nonlocal in_N, is_valid_dir, valid_count
        nonlocal best_complement, best_in_N, best_is_valid_dir, best_valid_count
        
        work_in_N = in_N.copy()
        work_is_valid_dir = is_valid_dir.copy()
        work_valid_count = valid_count.copy()
        
        for round_num in range(max_rounds):
            if time.time() > deadline - safety_margin:
                break
            
            progress = False
            for y in order:
                if not work_in_N[y]:
                    continue
                
                # Check if y is removable.
                # y is removable if for all x such that y is on line x-d, x has another valid line.
                # i.e., for all x in covers[y, d] for all d, if work_is_valid_dir[x, d] is True,
                # then work_valid_count[x] > 1 (so x still has another valid line).
                
                removable = True
                
                # Iterate over directions d
                for d in range(num_dirs):
                    # Get all x such that y is on line x-d
                    xs = covers[y, d, :]
                    
                    # For each x, check if work_is_valid_dir[x, d] is True
                    # If it is, check if work_valid_count[x] > 1
                    # If work_valid_count[x] == 1, then y is critical for x.
                    
                    # Vectorize:
                    # valid_mask = work_is_valid_dir[xs, d]
                    # critical = valid_mask & (work_valid_count[xs] == 1)
                    # if np.any(critical): removable = False; break
                    
                    valid_mask = work_is_valid_dir[xs, d]
                    if np.any(valid_mask):
                        counts = work_valid_count[xs]
                        if np.any(counts[valid_mask] == 1):
                            removable = False
                            break
                            
                if removable:
                    # Remove y
                    work_in_N[y] = False
                    progress = True
                    
                    # Update is_valid_dir and valid_count for all x affected by removing y
                    # For each d, for each x in covers[y, d]:
                    #   if work_is_valid_dir[x, d] was True, it becomes False
                    #   work_valid_count[x] -= 1
                    
                    for d in range(num_dirs):
                        xs = covers[y, d, :]
                        valid_mask = work_is_valid_dir[xs, d]
                        if np.any(valid_mask):
                            # Update valid_count
                            work_valid_count[xs] -= valid_mask.astype(np.int32)
                            # Update is_valid_dir
                            work_is_valid_dir[xs, d] = False & work_is_valid_dir[xs, d] # Just set to False where it was True?
                            # Actually, if y is removed, the line is no longer valid.
                            # So work_is_valid_dir[xs, d] = False for all xs where it was True.
                            work_is_valid_dir[xs, d][valid_mask] = False

            if not progress:
                break
        
        comp = np.sum(~work_in_N)
        if comp > best_complement:
            best_complement = comp
            best_in_N = work_in_N.copy()
            best_is_valid_dir = work_is_valid_dir.copy()
            best_valid_count = work_valid_count.copy()
            
        # Restore state for next strategy?
        # The original code restored to best. I will keep the current best as the starting point for next strategy.
        in_N = best_in_N.copy()
        is_valid_dir = best_is_valid_dir.copy()
        valid_count = best_valid_count.copy()

    # Strategy 1: Sequential order
    order1 = list(range(total_points))
    try_prune(order1, max_rounds=10)

    # Strategy 2: Random order
    if time.time() < deadline - safety_margin:
        rng = random.Random(42)
        order2 = list(range(total_points))
        rng.shuffle(order2)
        try_prune(order2, max_rounds=10)

    # Strategy 3: Prioritize points with low "criticality"
    # Criticality of y: number of pairs (x, d) such that y is on line x-d and x has valid_count 1 and is_valid_dir[x, d] is True.
    # If criticality is 0, y is immediately removable.
    if time.time() < deadline - safety_margin:
        criticality = np.zeros(total_points, dtype=np.int32)
        # For each y, for each d, for each x in covers[y, d]:
        # if is_valid_dir[x, d] and valid_count[x] == 1: criticality[y] += 1
        
        # This is O(2401 * 50 * 48) ~ 5.7M ops. Fast.
        for y in range(total_points):
            if not in_N[y]:
                continue
            for d in range(num_dirs):
                xs = covers[y, d, :]
                valid_mask = is_valid_dir[xs, d]
                if np.any(valid_mask):
                    counts = valid_count[xs]
                    criticality[y] += np.sum(counts[valid_mask] == 1)
                    
        # Sort by criticality, then by index
        order3 = sorted(range(total_points), key=lambda y: (criticality[y], y))
        try_prune(order3, max_rounds=20)

    # Strategy 4: Random restarts with best so far
    if time.time() < deadline - safety_margin:
        for seed in [123, 456, 789, 1000]:
            if time.time() > deadline - safety_margin:
                break
            rng = random.Random(seed)
            order4 = list(range(total_points))
            rng.shuffle(order4)
            try_prune(order4, max_rounds=20)

    # Final validity check
    # Ensure valid_count > 0 for all points
    if np.any(valid_count == 0):
        # Invalid, revert to baseline
        in_N = np.ones(total_points, dtype=bool)
        in_N[condition_flat] = False
        # Recalculate validity? Or just return baseline.
        # The baseline is known to be valid.
        
        # Re-extract baseline points
        points = []
        for p_idx in range(total_points):
            if in_N[p_idx]:
                x_idx = p_idx // q
                y_idx = p_idx % q
                xa, xb = elems[x_idx]
                ya, yb = elems[y_idx]
                points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)

    # Extract points from best_in_N
    points = []
    for p_idx in range(total_points):
        if best_in_N[p_idx]:
            x_idx = p_idx // q
            y_idx = p_idx % q
            xa, xb = elems[x_idx]
            ya, yb = elems[y_idx]
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
