import numpy as np
import time
import random
from heapq import heappush, heappop

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Returns an array of points in N, where N is the Nikodym set.
    Uses optimized baseline construction followed by aggressive multi-strategy pruning
    and local search to minimize |N| while maintaining Nikodym validity.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s kill
    safety_margin = 10.0

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
    # Start with a known valid construction, then optimize
    in_N = np.ones(total_points, dtype=bool)
    
    # Standard construction: N = {(x,y) : Tr(x) + N(y) != 0}
    # This gives a valid Nikodym set where each point has a line with all other points in N
    trace_arr = np.zeros(q, dtype=np.int64)
    norm_arr = np.zeros(q, dtype=np.int64)
    for i in range(q):
        a, b = elems[i]
        trace_arr[i] = (2 * a) % p
        norm_arr[i] = (a * a - b * b * alpha_sq) % p
    
    # Create the baseline: in_N[x_idx, y_idx] = True if (trace[x] + norm[y]) % p != 0
    sum_arr = (trace_arr[:, None] + norm_arr[None, :]) % p
    baseline_mask = (sum_arr != 0).flatten()
    in_N[baseline_mask] = True
    in_N[~baseline_mask] = False

    # Precompute line points for all points and directions
    line_points = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)
    
    one_idx = elem_to_idx(1, 0)  # element 1
    zero_idx = elem_to_idx(0, 0)  # element 0
    
    x_indices_all = np.arange(total_points) // q
    y_indices_all = np.arange(total_points) % q
    
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
        
        # Vectorized computation
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

    # Check baseline validity
    if np.any(valid_count == 0):
        # Fallback: use the "almost full" set
        in_N = np.ones(total_points, dtype=bool)
        y_special = set()
        for u in range(p):
            for v in range(p):
                y_special.add(((u * u) % p, v))
        
        y_special_mask = np.zeros(q, dtype=bool)
        for ys in y_special:
            idx = elem_to_idx(ys[0], ys[1])
            y_special_mask[idx] = True
            
        sum_arr2 = (trace_arr[:, None] + norm_arr[None, :]) % p
        condition = (sum_arr2 == 0) & y_special_mask[None, :]
        in_N[condition.flatten()] = False
        
        # Recompute validity
        is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)
        valid_count = np.zeros(total_points, dtype=np.int32)
        for d in range(num_dirs):
            mask = in_N[line_points[:, d, :]]
            valid_pts = np.all(mask, axis=1)
            is_valid_dir[:, d] = valid_pts
            valid_count += valid_pts.astype(np.int32)

    # Best solution tracking
    best_complement = np.sum(~in_N)
    best_in_N = in_N.copy()
    best_is_valid_dir = is_valid_dir.copy()
    best_valid_count = valid_count.copy()

    # Compute covers: covers[y, d, :] = points x such that y is on line x-d
    # This is the same as line_points[y, d, :] due to line symmetry
    covers = line_points  # They're the same

    def try_prune(order, max_rounds=5, use_heap=False):
        """Try to prune points in the given order."""
        nonlocal best_complement, best_in_N, best_is_valid_dir, best_valid_count
        
        work_in_N = best_in_N.copy()
        work_is_valid_dir = best_is_valid_dir.copy()
        work_valid_count = best_valid_count.copy()
        
        if use_heap:
            # Priority queue based on criticality
            import heapq
            heap = []
            for y in order:
                if not work_in_N[y]:
                    continue
                # Criticality: number of (x,d) where y is critical for x's validity
                crit = 0
                for d in range(num_dirs):
                    xs = covers[y, d, :]
                    valid_mask = work_is_valid_dir[xs, d]
                    if np.any(valid_mask):
                        counts = work_valid_count[xs]
                        crit += np.sum(counts[valid_mask] == 1)
                heappush(heap, (crit, y))
            
            for round_num in range(max_rounds):
                if time.time() > deadline - safety_margin:
                    break
                
                progress = False
                while heap:
                    crit, y = heappop(heap)
                    if not work_in_N[y]:
                        continue
                    
                    # Check if y is removable
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
                        # Remove y
                        work_in_N[y] = False
                        progress = True
                        
                        # Update validity for affected points
                        for d in range(num_dirs):
                            xs = covers[y, d, :]
                            valid_mask = work_is_valid_dir[xs, d]
                            if np.any(valid_mask):
                                work_valid_count[xs] -= valid_mask.astype(np.int32)
                                work_is_valid_dir[xs, d] = False
                        
                        # Recompute criticality for neighbors and push them
                        for d in range(num_dirs):
                            xs = covers[y, d, :]
                            for x in xs:
                                if work_in_N[x]:
                                    new_crit = 0
                                    for d2 in range(num_dirs):
                                        xs2 = covers[x, d2, :]
                                        vm2 = work_is_valid_dir[xs2, d2]
                                        if np.any(vm2):
                                            c2 = work_valid_count[xs2]
                                            new_crit += np.sum(c2[vm2] == 1)
                                    heappush(heap, (new_crit, x))
                    else:
                        # Re-push y with current criticality
                        heappush(heap, (crit, y))
                
                if not progress:
                    break
        else:
            # Simple sequential pruning
            for round_num in range(max_rounds):
                if time.time() > deadline - safety_margin:
                    break
                
                progress = False
                for y in order:
                    if not work_in_N[y]:
                        continue
                    
                    # Check if y is removable
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
                        # Remove y
                        work_in_N[y] = False
                        progress = True
                        
                        # Update validity for affected points
                        for d in range(num_dirs):
                            xs = covers[y, d, :]
                            valid_mask = work_is_valid_dir[xs, d]
                            if np.any(valid_mask):
                                work_valid_count[xs] -= valid_mask.astype(np.int32)
                                work_is_valid_dir[xs, d] = False
                
                if not progress:
                    break

        comp = np.sum(~work_in_N)
        if comp > best_complement:
            best_complement = comp
            best_in_N = work_in_N.copy()
            best_is_valid_dir = work_is_valid_dir.copy()
            best_valid_count = work_valid_count.copy()

    # Strategy 1: Sequential order
    order1 = list(range(total_points))
    try_prune(order1, max_rounds=10)

    # Strategy 2: Random order
    if time.time() < deadline - safety_margin:
        rng = random.Random(42)
        order2 = list(range(total_points))
        rng.shuffle(order2)
        try_prune(order2, max_rounds=10)

    # Strategy 3: Criticality-based priority queue
    if time.time() < deadline - safety_margin:
        order3 = list(range(total_points))
        try_prune(order3, max_rounds=15, use_heap=True)

    # Strategy 4: Multiple random restarts
    if time.time() < deadline - safety_margin:
        for seed in [123, 456, 789, 1000, 2000]:
            if time.time() > deadline - safety_margin:
                break
            rng = random.Random(seed)
            order4 = list(range(total_points))
            rng.shuffle(order4)
            try_prune(order4, max_rounds=15)

    # Local search: try swapping points
    if time.time() < deadline - safety_margin:
        # Try to add a point that was removed and remove a different point
        removed_points = np.where(~best_in_N)[0]
        kept_points = np.where(best_in_N)[0]
        
        if len(removed_points) > 0 and len(kept_points) > 0:
            rng = random.Random(42)
            for _ in range(min(100, len(removed_points) * len(kept_points) // 10)):
                if time.time() > deadline - safety_margin:
                    break
                
                add_pt = rng.choice(removed_points)
                rem_pt = rng.choice(kept_points)
                
                # Try the swap
                trial_in_N = best_in_N.copy()
                trial_in_N[add_pt] = True
                trial_in_N[rem_pt] = False
                
                # Check validity
                trial_valid = np.zeros(total_points, dtype=np.int32)
                for d in range(num_dirs):
                    mask = trial_in_N[line_points[:, d, :]]
                    valid_pts = np.all(mask, axis=1)
                    trial_valid += valid_pts.astype(np.int32)
                
                if np.all(trial_valid > 0):
                    comp = np.sum(~trial_in_N)
                    if comp > best_complement:
                        best_complement = comp
                        best_in_N = trial_in_N.copy()
                        # Recompute full validity
                        best_is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)
                        best_valid_count = np.zeros(total_points, dtype=np.int32)
                        for d in range(num_dirs):
                            mask = best_in_N[line_points[:, d, :]]
                            valid_pts = np.all(mask, axis=1)
                            best_is_valid_dir[:, d] = valid_pts
                            best_valid_count += valid_pts.astype(np.int32)

    # Final validity check
    if np.any(best_valid_count == 0):
        # Invalid, revert to a known valid set
        in_N = np.ones(total_points, dtype=bool)
        y_special = set()
        for u in range(p):
            for v in range(p):
                y_special.add(((u * u) % p, v))
        
        y_special_mask = np.zeros(q, dtype=bool)
        for ys in y_special:
            idx = elem_to_idx(ys[0], ys[1])
            y_special_mask[idx] = True
            
        sum_arr2 = (trace_arr[:, None] + norm_arr[None, :]) % p
        condition = (sum_arr2 == 0) & y_special_mask[None, :]
        in_N[condition.flatten()] = False
        
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
