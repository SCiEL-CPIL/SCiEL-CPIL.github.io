"""Model-and-optimize TSP constructive heuristic.

Strategy:
  1. For each new instance, fit parameters (alpha, gamma, delta, temp, eta) via
     a focused grid search using a fast approximate tour builder.
  2. Use a vectorized 2-step lookahead with soft-min (log-sum-exp) for
     smoothness.
  3. Add a smooth connectivity/isolation penalty term.
  4. Add a geometric turn-angle penalty: reward locally tight turns (small
     circumradius of triangle cur-cand-dest) to preserve convexity and avoid
     long inefficient chords.
  5. Critically: reset per-instance state when a new instance is detected,
     so parameters are not carried over between different problem instances.
  6. Optimize the fitting loop to be faster by reducing the grid and
     short-circuiting when time is tight.

Score per candidate c:
  score(c) = d(cur,c) + alpha*d(c,dest) + gamma*softmin_{u in rem\{c}} d(c,u)
             + delta * (d(c,dest) - softmin_u d(u,dest))
             + eta * circumradius_penalty(cur, c, dest)
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

# Per-instance state - MUST be reset between instances
_current_instance_id = -1
_FIT_PARAMS = None  # (alpha, gamma, delta, temp, eta) best found by fitting
_instance_counter = 0
_last_dm_shape = None  # Track distance matrix shape to detect new instances
_last_dm_checksum = None  # Quick checksum to detect if same matrix


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _softmin_vectorized(arr_2d, temperature=1.0):
    """Vectorized soft-min along axis=1 for a 2D array.
    
    Returns an array where each element is the soft-min of the corresponding row.
    softmin(x) = -T * log(sum(exp(-x_i / T)))
    """
    if arr_2d.ndim == 1:
        arr_2d = arr_2d.reshape(1, -1)
    if arr_2d.size == 0:
        return np.zeros(arr_2d.shape[0])
    if temperature <= 0:
        return np.min(arr_2d, axis=1)
    # Numerically stable: subtract max before exp
    scaled = -arr_2d / temperature
    max_scaled = np.max(scaled, axis=1, keepdims=True)
    # Handle cases where max is -inf (all -inf) or +inf
    max_scaled_safe = np.where(np.isfinite(max_scaled), max_scaled, 0)
    exp_vals = np.exp(scaled - max_scaled_safe)
    # Where max was not finite, the row was all -inf or had issues; fall back to min
    log_sum = np.log(np.sum(exp_vals, axis=1, keepdims=True))
    result = -temperature * (max_scaled_safe.squeeze() + log_sum.squeeze())
    # For rows where max_scaled was not finite, use hard min
    bad_mask = ~np.isfinite(max_scaled.squeeze())
    if np.any(bad_mask):
        result[bad_mask] = np.min(arr_2d[bad_mask], axis=1)
    return result


def _softmin_1d(arr, temperature=1.0):
    """Soft-min for a 1D array."""
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        return float(arr[0])
    if temperature <= 0:
        return float(np.min(arr))
    scaled = -arr / temperature
    max_scaled = np.max(scaled)
    if not np.isfinite(max_scaled):
        return float(np.min(arr))
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return float(-temperature * log_sum)


def _build_tour_fast(distance_matrix, start_node, alpha=0.0, gamma=0.0, delta=0.0,
                     soft_temp=0.5, use_soft=False, eta=0.0):
    """Build a full TSP tour using the given parameters. Returns tour length.
    
    Optimized for speed: uses vectorized operations and avoids unnecessary work.
    """
    n = distance_matrix.shape[0]
    current = start_node
    total_dist = 0.0
    remaining = list(range(1, n))  # 0 is start
    
    while len(remaining) > 0:
        rem_arr = np.array(remaining)
        # Vectorized scoring
        d_cur = distance_matrix[current, rem_arr]
        
        scores = d_cur.copy()
        
        if alpha > 0:
            d_dest = distance_matrix[rem_arr, start_node]
            scores += alpha * d_dest
        
        if gamma > 0 and len(remaining) > 2:
            # For each candidate c, compute min distance from c to other remaining nodes
            sub_dist = distance_matrix[np.ix_(rem_arr, rem_arr)].copy()
            np.fill_diagonal(sub_dist, np.inf)
            if use_soft:
                min_next = _softmin_vectorized(sub_dist, temperature=soft_temp)
            else:
                min_next = np.min(sub_dist, axis=1)
            scores += gamma * min_next
        
        if delta > 0 and len(remaining) > 2:
            # Isolation penalty: penalize candidates whose distance to dest
            # is much larger than the minimum among remaining
            d_dest_all = distance_matrix[rem_arr, start_node]
            if use_soft:
                min_d_dest = _softmin_1d(d_dest_all, temperature=soft_temp)
            else:
                min_d_dest = np.min(d_dest_all)
            scores += delta * (d_dest_all - min_d_dest)
        
        if eta > 0 and len(remaining) > 1:
            # Geometric turn-angle penalty: circumradius of triangle (current, cand, dest)
            # Circumradius R = (a*b*c) / (4*Area)
            # For degenerate triangles (collinear), R is large -> penalty
            # We use a normalized penalty: R / max_d where max_d is max distance in matrix
            d_cand_dest = distance_matrix[rem_arr, start_node]
            # Compute area using Heron's formula: a=d_cur, b=d_cand_dest, c=d(current,dest)
            d_cur_dest = distance_matrix[current, start_node]
            a = d_cur
            b = d_cand_dest
            c = d_cur_dest
            # Semi-perimeter
            s = (a + b + c) / 2.0
            # Area^2 = s*(s-a)*(s-b)*(s-c)
            area_sq = s * (s - a) * (s - b) * (s - c)
            # Clamp to avoid negative due to floating point
            area_sq = np.maximum(area_sq, 1e-12)
            area = np.sqrt(area_sq)
            # Circumradius R = a*b*c / (4*area)
            R = (a * b * c) / (4.0 * area)
            # Normalize by the scale of the problem
            max_d = np.max(distance_matrix)
            if max_d > 0:
                R_norm = R / max_d
            else:
                R_norm = R
            scores += eta * R_norm
        
        best_idx = int(np.argmin(scores))
        next_node = remaining[best_idx]
        total_dist += distance_matrix[current, next_node]
        remaining.pop(best_idx)
        current = next_node
    
    # Close the tour
    total_dist += distance_matrix[current, start_node]
    return total_dist


def _fit_params(distance_matrix, start_node, time_budget=3.0):
    """Fit alpha, gamma, delta, temp, eta by grid search on the full instance.
    
    Uses fast tour builder for speed. Tries hard-min first, then refines
    with soft-min if time permits.
    """
    global _FIT_PARAMS
    
    if _is_deadline_passed():
        return (0.0, 0.0, 0.0, 0.5, 0.0)
    
    # Focused grid for speed
    alpha_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    gamma_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    delta_values = [0.0, 0.05, 0.1, 0.15]
    eta_values = [0.0, 0.02, 0.05, 0.1]
    temp_values = [0.5, 1.0]
    
    best_alpha = 0.0
    best_gamma = 0.0
    best_delta = 0.0
    best_temp = 0.5
    best_eta = 0.0
    best_len = float('inf')
    
    elapsed_start = time.time()
    remaining_time = _LIMIT - (time.time() - _start_time) - 5.0
    deadline = elapsed_start + min(time_budget, max(0.5, remaining_time))
    
    # Phase 1: Coarse grid with hard min (fast) - skip temp variation
    for alpha in alpha_values:
        if time.time() > deadline:
            break
        for gamma in gamma_values:
            if time.time() > deadline:
                break
            for delta in delta_values:
                if time.time() > deadline:
                    break
                for eta in eta_values:
                    if time.time() > deadline:
                        break
                    tour_len = _build_tour_fast(distance_matrix, start_node, alpha, gamma, delta, 
                                                use_soft=False, eta=eta)
                    if tour_len < best_len:
                        best_len = tour_len
                        best_alpha = alpha
                        best_gamma = gamma
                        best_delta = delta
                        best_eta = eta
    
    # Phase 2: If time remains, try soft-min around best point with varying temperature
    if time.time() < deadline - 0.5:
        for temp in temp_values:
            for sa in [max(0, best_alpha - 0.05), best_alpha, min(0.4, best_alpha + 0.05)]:
                if time.time() > deadline:
                    break
                for sg in [max(0, best_gamma - 0.05), best_gamma, min(0.3, best_gamma + 0.05)]:
                    if time.time() > deadline:
                        break
                    for sd in [max(0, best_delta - 0.05), best_delta, min(0.2, best_delta + 0.05)]:
                        if time.time() > deadline:
                            break
                        for se in [max(0, best_eta - 0.02), best_eta, min(0.2, best_eta + 0.02)]:
                            if time.time() > deadline:
                                break
                            tour_len = _build_tour_fast(distance_matrix, start_node, sa, sg, sd,
                                                        soft_temp=temp, use_soft=True, eta=se)
                            if tour_len < best_len:
                                best_len = tour_len
                                best_alpha = sa
                                best_gamma = sg
                                best_delta = sd
                                best_temp = temp
                                best_eta = se
    
    _FIT_PARAMS = (best_alpha, best_gamma, best_delta, best_temp, best_eta)
    return (best_alpha, best_gamma, best_delta, best_temp, best_eta)


def _detect_new_instance(distance_matrix, unvisited_nodes):
    """Detect if we're starting a new instance by checking unvisited count 
    against matrix size and using a checksum."""
    global _last_dm_shape, _last_dm_checksum
    
    n_total = distance_matrix.shape[0]
    n_unvisited = len(unvisited_nodes)
    
    # A new instance starts when all (or nearly all) nodes are unvisited
    # and the distance matrix is different from the previous one
    if n_unvisited >= n_total - 1:
        # Quick check: is this the same matrix as before?
        # Use a small checksum for speed
        checksum = np.sum(distance_matrix[::max(1, n_total//10), ::max(1, n_total//10)])
        if _last_dm_checksum is None or abs(checksum - _last_dm_checksum) > 1e-6:
            _last_dm_shape = distance_matrix.shape
            _last_dm_checksum = checksum
            return True
        elif _last_dm_shape != distance_matrix.shape:
            _last_dm_shape = distance_matrix.shape
            _last_dm_checksum = checksum
            return True
    
    return False


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Uses model-and-optimize: fits parameters per instance via grid search,
    then uses them for all subsequent steps of that instance.
    Critically resets parameters when a new instance is detected.
    """
    global _current_instance_id, _FIT_PARAMS, _instance_counter, _last_dm_shape, _last_dm_checksum
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Detect new instance
    is_new_instance = _detect_new_instance(distance_matrix, unvisited_nodes)
    
    if is_new_instance:
        _instance_counter += 1
        # Reset parameters for the new instance
        _FIT_PARAMS = None
        # Use more time for first instance, less for subsequent
        tb = 2.5 if _instance_counter == 1 else 1.2
        _FIT_PARAMS = _fit_params(distance_matrix, current_node, time_budget=tb)
    
    if _FIT_PARAMS is None:
        # Fallback default
        alpha, gamma, delta, temp, eta = 0.1, 0.1, 0.05, 0.5, 0.05
    else:
        alpha, gamma, delta, temp, eta = _FIT_PARAMS
    
    # If deadline is close, use pure nearest neighbor
    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    # Compute scores for candidates (vectorized)
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores = dists_from_current.copy()
    
    if alpha > 0:
        dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += alpha * dists_to_dest
    
    if gamma > 0 and n_unvisited > 2:
        # Vectorized 2-step lookahead with soft-min for smoothness
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].copy()
        np.fill_diagonal(sub_dist, np.inf)
        min_next = _softmin_vectorized(sub_dist, temperature=temp)
        scores += gamma * min_next
    
    if delta > 0 and n_unvisited > 2:
        # Isolation penalty with soft-min
        d_dest_all = distance_matrix[unvisited_nodes, destination_node]
        min_d_dest = _softmin_1d(d_dest_all, temperature=temp)
        scores += delta * (d_dest_all - min_d_dest)
    
    if eta > 0 and n_unvisited > 1:
        # Geometric turn-angle penalty: circumradius of triangle (current, cand, dest)
        d_cand_dest = distance_matrix[unvisited_nodes, destination_node]
        d_cur_dest = distance_matrix[current_node, destination_node]
        a = dists_from_current
        b = d_cand_dest
        c = d_cur_dest
        # Semi-perimeter
        s = (a + b + c) / 2.0
        # Area^2 = s*(s-a)*(s-b)*(s-c)
        area_sq = s * (s - a) * (s - b) * (s - c)
        # Clamp to avoid negative due to floating point
        area_sq = np.maximum(area_sq, 1e-12)
        area = np.sqrt(area_sq)
        # Circumradius R = a*b*c / (4*area)
        R = (a * b * c) / (4.0 * area)
        # Normalize by the scale of the problem
        max_d = np.max(distance_matrix)
        if max_d > 0:
            R_norm = R / max_d
        else:
            R_norm = R
        scores += eta * R_norm
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
