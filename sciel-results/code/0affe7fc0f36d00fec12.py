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
    safety_margin = 10.0

    p = 7
    q = p * p
    alpha_sq = 3

    def elem_to_idx(a, b):
        return a * p + b

    def idx_to_elem(idx):
        return (idx // p, idx % p)

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

    in_N = np.ones(total_points, dtype=bool)
    
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
            
    trace_arr = np.zeros(q, dtype=np.int64)
    norm_arr = np.zeros(q, dtype=np.int64)
    for i in range(q):
        a, b = elems[i]
        trace_arr[i] = (2 * a) % p
        norm_arr[i] = (a * a - b * b * alpha_sq) % p
    
    y_special_mask = np.zeros(q, dtype=bool)
    for ys in y_special:
        idx = elem_to_idx(ys[0], ys[1])
        y_special_mask[idx] = True
        
    sum_arr = (trace_arr[:, None] + norm_arr[None, :]) % p
    condition = (sum_arr == 0) & y_special_mask[None, :]
    condition_flat = condition.flatten()
    in_N[condition_flat] = False

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
            
        txs = np.array([mul_table[t_idx, vx_idx] for t_idx in range(1, q)])
        tys = np.array([mul_table[t_idx, vy_idx] for t_idx in range(1, q)])
        
        x_indices = np.arange(total_points) // q
        y_indices = np.arange(total_points) % q
        
        nx = add_table[x_indices[:, None], txs[None, :]]
        ny = add_table[y_indices[:, None], tys[None, :]]
        line_points[:, d_idx, :] = nx * q + ny

    is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)
    valid_count = np.zeros(total_points, dtype=np.int32)
    
    for d in range(num_dirs):
        mask = in_N[line_points[:, d, :]]
        valid_pts = np.all(mask, axis=1)
        is_valid_dir[:, d] = valid_pts
        valid_count += valid_pts.astype(np.int32)

    covers = line_points # covers[y, d] == line_points[y, d]

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
                
                removable = True
                for d in range(num_dirs):
                    xs = covers[y, d, :]
                    valid_mask = work_is_valid_dir[xs, d]
                    if np.any(valid_mask):
                        counts = work_valid_count[xs]
                        if np.any(counts[valid_mask] == 1):
                            removable = False
                            break
                            
                if removable:
                    work_in_N[y] = False
                    progress = True
                    for d in range(num_dirs):
                        xs = covers[y, d, :]
                        valid_mask = work_is_valid_dir[xs, d]
                        if np.any(valid_mask):
                            work_valid_count[xs] -= valid_mask.astype(np.int32)
                            work_is_valid_dir[xs, d][valid_mask] = False

            if not progress:
                break
        
        comp = np.sum(~work_in_N)
        if comp > best_complement:
            best_complement = comp
            best_in_N = work_in_N.copy()
            best_is_valid_dir = work_is_valid_dir.copy()
            best_valid_count = work_valid_count.copy()
            
        in_N = best_in_N.copy()
        is_valid_dir = best_is_valid_dir.copy()
        valid_count = best_valid_count.copy()

    # Strategy 1: Sequential
    order1 = list(range(total_points))
    try_prune(order1, max_rounds=10)

    # Strategy 2: Random
    if time.time() < deadline - safety_margin:
        rng = random.Random(42)
        order2 = list(range(total_points))
        rng.shuffle(order2)
        try_prune(order2, max_rounds=10)

    # Strategy 3: Criticality
    if time.time() < deadline - safety_margin:
        criticality = np.zeros(total_points, dtype=np.int32)
        for y in range(total_points):
            if not in_N[y]:
                continue
            for d in range(num_dirs):
                xs = covers[y, d, :]
                valid_mask = is_valid_dir[xs, d]
                if np.any(valid_mask):
                    counts = valid_count[xs]
                    criticality[y] += np.sum(counts[valid_mask] == 1)
        order3 = sorted(range(total_points), key=lambda y: (criticality[y], y))
        try_prune(order3, max_rounds=20)

    # Strategy 4: More random restarts
    if time.time() < deadline - safety_margin:
        for seed in [123, 456, 789, 1000, 2000, 3000]:
            if time.time() > deadline - safety_margin:
                break
            rng = random.Random(seed)
            order4 = list(range(total_points))
            rng.shuffle(order4)
            try_prune(order4, max_rounds=20)

    if np.any(valid_count == 0):
        in_N = np.ones(total_points, dtype=bool)
        in_N[condition_flat] = False
        points = []
        for p_idx in range(total_points):
            if in_N[p_idx]:
                x_idx = p_idx // q
                y_idx = p_idx % q
                xa, xb = elems[x_idx]
                ya, yb = elems[y_idx]
                points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)

    points = []
    for p_idx in range(total_points):
        if best_in_N[p_idx]:
            x_idx = p_idx // q
            y_idx = p_idx % q
            xa, xb = elems[x_idx]
            ya, yb = elems[y_idx]
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
