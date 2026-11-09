import numpy as np
import time
import random
import heapq

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Returns an array of points in N, where N is the Nikodym set.
    Uses baseline construction followed by greedy pruning with incremental cost updates.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 15.0

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
    in_N = np.ones(total_points, dtype=bool)
    
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
            
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
    sum_arr = (trace_arr[:, None] + norm_arr[None, :]) % p
    condition = (sum_arr == 0) & y_special_mask[None, :]
    
    # Flatten to 1D
    condition_flat = condition.flatten()
    in_N[condition_flat] = False

    # Precompute line points
    # line_points[pt, d, t_idx] = index of point at distance t_idx along direction d from pt
    line_points = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)
    
    one_idx = elem_to_idx(1, 0)
    zero_idx = elem_to_idx(0, 0)
    
    # Directions
    # d < q: v = (1, d)
    # d == q: v = (0, 1)
    
    x_indices_all = np.arange(total_points) // q
    y_indices_all = np.arange(total_points) % q
    
    for d_idx in range(num_dirs):
        if d_idx < q:
            vx_idx = one_idx
            vy_idx = d_idx
        else:
            vx_idx = zero_idx
            vy_idx = one_idx
        
        txs = np.array([mul_table[t_idx, vx_idx] for t_idx in range(1, q)])
        tys = np.array([mul_table[t_idx, vy_idx] for t_idx in range(1, q)])
        
        nx = add_table[x_indices_all[:, None], txs[None, :]]
        ny = add_table[y_indices_all[:, None], tys[None, :]]
        line_points[:, d_idx, :] = nx * q + ny

    # Initialize validity tracking
    is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)
    valid_count = np.zeros(total_points, dtype=np.int32)
    
    for d in range(num_dirs):
        mask = in_N[line_points[:, d, :]]
        valid_pts = np.all(mask, axis=1)
        is_valid_dir[:, d] = valid_pts
        valid_count += valid_pts.astype(np.int32)

    # covers[y, d] = line_points[y, d, :] (the points on the line through y in direction d, excluding y)
    covers = line_points

    # State for greedy pruning
    in_N = in_N.copy()
    is_valid_dir = is_valid_dir.copy()
    valid_count = valid_count.copy()
    
    # cost[y] = number of x such that y is on a valid line through x and valid_count[x] == 1
    cost = np.zeros(total_points, dtype=np.int32)
    
    # Compute initial costs
    for y in range(total_points):
        if in_N[y]:
            c = 0
            for d in range(num_dirs):
                xs = covers[y, d, :]
                valid_mask = is_valid_dir[xs, d]
                if np.any(valid_mask):
                    counts = valid_count[xs]
                    c += np.sum(counts[valid_mask] == 1)
            cost[y] = c

    # Heap of (cost, y) for points with cost 0
    heap = []
    for y in range(total_points):
        if in_N[y] and cost[y] == 0:
            heapq.heappush(heap, (0, y))

    # Greedy loop with incremental updates
    while heap:
        if time.time() > deadline - safety_margin:
            break
            
        c, y = heapq.heappop(heap)
        
        if not in_N[y] or cost[y] != 0:
            continue
        
        # Remove y from N
        in_N[y] = False
        
        # Update valid_count and is_valid_dir for affected x
        # For each direction d, find x such that y is on the line through x in direction d
        # and is_valid_dir[x, d] is True
        for d in range(num_dirs):
            xs = covers[y, d, :]
            valid_mask = is_valid_dir[xs, d]
            if not np.any(valid_mask):
                continue
            
            x_indices_valid = np.where(valid_mask)[0]
            
            for x_idx in x_indices_valid:
                x = xs[x_idx]
                
                # x loses direction d
                is_valid_dir[x, d] = False
                old_count = valid_count[x]
                valid_count[x] = old_count - 1
                
                if old_count == 2:
                    # x becomes critical (valid_count drops to 1)
                    # For each direction d' where is_valid_dir[x, d'] is True,
                    # all points z on the line through x in direction d' have their cost increased by 1
                    for d_prime in range(num_dirs):
                        if is_valid_dir[x, d_prime]:
                            zs = covers[x, d_prime, :]
                            mask_z = in_N[zs]
                            if np.any(mask_z):
                                cost[zs[mask_z]] += 1

    # Extract points from in_N
    points = []
    for p_idx in range(total_points):
        if in_N[p_idx]:
            x_idx = p_idx // q
            y_idx = p_idx % q
            xa, xb = elems[x_idx]
            ya, yb = elems[y_idx]
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
