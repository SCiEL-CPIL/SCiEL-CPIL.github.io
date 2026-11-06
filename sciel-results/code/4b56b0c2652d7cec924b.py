"""Rewritten program: spectral + local-structure + simulated-greedy value priority for Maximum Independent Set."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_is_size(order, A, n):
    """Run greedy independent set selection given a priority order (array of vertex indices sorted by score desc).
    Returns the size of the resulting independent set."""
    selected = np.zeros(n, dtype=bool)
    in_set = np.zeros(n, dtype=bool)
    size = 0
    for v in order:
        v = int(v)
        if selected[v]:
            continue
        in_set[v] = True
        size += 1
        # Mark neighbors as selected (blocked)
        nbrs = np.where(A[v] > 0.5)[0]
        selected[nbrs] = True
    return size


def priority(adjacency):
    """
    Assign a score to each vertex. The greedy decoder picks vertices in descending
    score order, skipping neighbours of already-picked vertices.
    
    Strategy: Combine structural features (degree, mean neighbor degree, triangle count)
    with a simulated-greedy value signal. We run several perturbed greedy simulations
    and credit vertices that are selected early/decisively, directly targeting the
    realized independent-set size metric.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]

    if n == 0:
        return np.array([], dtype=float)

    # Compute degree
    degree = A.sum(axis=1)

    # Compute mean neighbor degree
    neighbor_degree_sum = A @ degree
    mean_neighbor_degree = np.where(degree > 0, neighbor_degree_sum / np.maximum(degree, 1), 0.0)

    # Triangle count per vertex: number of triangles containing vertex i
    # tri[i] = 0.5 * sum_j A[i,j] * (A @ A)[i,j]
    AA = A @ A
    tri_count = 0.5 * np.sum(A * AA, axis=1)

    # Check time
    if time.time() > deadline - 30:
        # Fallback: simple structural score
        deg_norm = degree / (np.max(degree) + 1e-10)
        mnd_norm = mean_neighbor_degree / (np.max(mean_neighbor_degree) + 1e-10)
        tri_norm = tri_count / (np.max(tri_count) + 1e-10)
        score = -0.4 * deg_norm + 0.1 * mnd_norm - 0.1 * tri_norm
        return score.astype(float)

    # Simulated greedy value: run multiple perturbed greedy simulations
    # and accumulate a "value" score for each vertex based on how early
    # it gets selected (earlier = more valuable).
    
    # Base ordering: low degree first (heuristic for good IS)
    base_order = np.argsort(degree)  # ascending degree
    
    # We'll run several simulations with different tie-breaking / perturbations
    num_sims = 5
    value_score = np.zeros(n, dtype=float)
    
    for sim in range(num_sims):
        if time.time() > deadline - 20:
            break
        
        if sim == 0:
            # Pure degree-based ordering
            order = base_order.copy()
        elif sim == 1:
            # Degree + mean neighbor degree (prefer high MND among same degree)
            combined = degree - 0.5 * mean_neighbor_degree
            order = np.argsort(combined)
        elif sim == 2:
            # Degree + triangle count (prefer low triangles)
            combined = degree + 0.3 * tri_count
            order = np.argsort(combined)
        elif sim == 3:
            # Random perturbation of degree order
            noise = np.random.uniform(0, 1, size=n)
            combined = degree + noise
            order = np.argsort(combined)
        else:
            # Inverse: try high-degree first to see what gets blocked
            order = np.argsort(-degree)
        
        # Run greedy and record selection order (position when selected)
        selected = np.zeros(n, dtype=bool)
        select_pos = np.full(n, -1, dtype=float)  # -1 means not selected
        pos_counter = 0
        for v in order:
            v = int(v)
            if selected[v]:
                continue
            select_pos[v] = float(pos_counter)
            pos_counter += 1
            nbrs = np.where(A[v] > 0.5)[0]
            selected[nbrs] = True
        
        # Value: vertices selected early get higher credit
        # Use inverse position: early selection = high value
        # For unselected vertices, assign low value
        selected_mask = select_pos >= 0
        if np.any(selected_mask):
            # Normalize positions to [0, 1] where 0 = earliest, 1 = latest
            min_pos = np.min(select_pos[selected_mask])
            max_pos = np.max(select_pos[selected_mask])
            if max_pos > min_pos:
                norm_pos = (select_pos - min_pos) / (max_pos - min_pos)
            else:
                norm_pos = np.zeros(n)
            # Value = 1 - normalized position for selected, 0 for unselected
            sim_value = np.where(selected_mask, 1.0 - norm_pos, 0.0)
        else:
            sim_value = np.zeros(n)
        
        value_score += sim_value
    
    # Average value across simulations
    value_score /= num_sims
    
    # Normalize value score
    max_val = np.max(value_score)
    if max_val > 0:
        value_score_norm = value_score / max_val
    else:
        value_score_norm = value_score
    
    # Normalize structural features
    deg_norm = degree / (np.max(degree) + 1e-10)
    mnd_norm = mean_neighbor_degree / (np.max(mean_neighbor_degree) + 1e-10)
    tri_norm = tri_count / (np.max(tri_count) + 1e-10)
    
    # Combine: value score is primary (directly targets metric),
    # structural features as secondary tiebreakers
    # Low degree, low triangles, high value
    w_value = 0.6
    w_deg = 0.2
    w_tri = 0.1
    w_mnd = 0.1
    
    score = w_value * value_score_norm - w_deg * deg_norm - w_tri * tri_norm + w_mnd * mnd_norm
    
    # Check time
    if time.time() > deadline:
        return score.astype(float)
    
    return score.astype(float)
# EVOLVE-BLOCK-END
