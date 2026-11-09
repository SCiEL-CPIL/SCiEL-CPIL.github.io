import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 135.0
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
        
    degrees = adjacency.sum(axis=1).astype(float)
    
    # Precompute features
    sum_neighbor_degrees = adjacency.dot(degrees)
    avg_neighbor_degree = np.zeros(n)
    nonzero_mask = degrees > 0
    avg_neighbor_degree[nonzero_mask] = sum_neighbor_degrees[nonzero_mask] / degrees[nonzero_mask]
    
    # Local edge density: number of edges among neighbors
    local_edges = np.zeros(n)
    for i in range(n):
        if time.time() > deadline - 40:
            break
        d = int(degrees[i])
        if d > 1:
            neighbors = np.where(adjacency[i])[0]
            sub = adjacency[np.ix_(neighbors, neighbors)]
            local_edges[i] = sub.sum() / 2.0
    
    # Normalized local edge density
    possible_edges = degrees * (degrees - 1) / 2.0
    local_density = np.zeros(n)
    valid = possible_edges > 0
    local_density[valid] = local_edges[valid] / possible_edges[valid]
    
    # Number of degree-1 neighbors (leaves) - selecting v with many leaf neighbors is wasteful
    leaf_neighbors = np.zeros(n)
    for i in range(n):
        if time.time() > deadline - 40:
            break
        if degrees[i] > 0:
            neighbors = np.where(adjacency[i])[0]
            leaf_neighbors[i] = np.sum(degrees[neighbors] == 1)
    
    # Average local density of neighbors
    avg_neighbor_local_density = np.zeros(n)
    nz = degrees > 0
    if nz.any():
        sum_nl = adjacency.dot(local_density)
        avg_neighbor_local_density[nz] = sum_nl[nz] / degrees[nz]
    
    # Precompute neighbor masks for fast greedy simulation
    neighbor_masks = [adjacency[i].copy() for i in range(n)]
    
    # Vectorized greedy decode
    def greedy_decode(scores):
        order = np.argsort(-scores)
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if blocked[v]:
                continue
            count += 1
            blocked[v] = True
            blocked |= neighbor_masks[v]
        return count
    
    # Iterative relaxation with damping
    relax_scores = np.ones(n) / (1.0 + degrees.mean())
    for it in range(50):
        if time.time() > deadline - 35:
            break
        neighbor_score_sums = adjacency.dot(relax_scores)
        relax_scores_new = 1.0 / (1.0 + neighbor_score_sums)
        relax_scores = 0.7 * relax_scores + 0.3 * relax_scores_new
    
    # Feature matrix
    # Normalize features to have zero mean, unit variance for stable weighting
    def normalize(x):
        m = x.mean()
        s = x.std()
        if s < 1e-10:
            return x - m
        return (x - m) / s
    
    deg_norm = normalize(degrees)
    avg_nd_norm = normalize(avg_neighbor_degree)
    ld_norm = normalize(local_density)
    le_norm = normalize(local_edges)
    anld_norm = normalize(avg_neighbor_local_density)
    relax_norm = normalize(relax_scores)
    leaf_norm = normalize(leaf_neighbors)
    
    # Start with a good baseline: prefer low degree, high avg neighbor degree
    # score = -deg + avg_nd - 0.5*ld - 0.3*leaf_neighbors
    base_scores = (-1.0 * deg_norm + 1.0 * avg_nd_norm - 0.5 * ld_norm - 0.3 * leaf_norm)
    best_count = greedy_decode(base_scores)
    best_scores = base_scores.copy()
    
    # Local search around the base weights
    # Weights: [w_deg, w_avgnd, w_ld, w_le, w_anld, w_relax, w_leaf]
    base_w = np.array([-1.0, 1.0, -0.5, 0.0, 0.0, 0.0, -0.3])
    
    features = np.column_stack([deg_norm, avg_nd_norm, ld_norm, le_norm, anld_norm, relax_norm, leaf_norm])
    
    def evaluate_weights(w):
        scores = features.dot(w)
        return greedy_decode(scores), scores
    
    # Fine local search
    for iteration in range(50):
        if time.time() > deadline - 10:
            break
        
        improved = False
        for dim in range(7):
            for delta in [-0.3, -0.15, -0.05, 0.05, 0.15, 0.3]:
                if time.time() > deadline - 5:
                    break
                w = base_w.copy()
                w[dim] += delta
                count, scores = evaluate_weights(w)
                if count > best_count:
                    best_count = count
                    best_scores = scores.copy()
                    base_w = w
                    improved = True
        
        if not improved:
            # Try larger steps
            for dim in range(7):
                for delta in [-1.0, -0.5, 0.5, 1.0]:
                    if time.time() > deadline - 5:
                        break
                    w = base_w.copy()
                    w[dim] += delta
                    count, scores = evaluate_weights(w)
                    if count > best_count:
                        best_count = count
                        best_scores = scores.copy()
                        base_w = w
                        improved = True
    
    # Try some specialized strategies
    if time.time() < deadline - 5:
        # Pure relaxation-based
        scores_r = relax_norm * 2.0 - deg_norm
        count_r = greedy_decode(scores_r)
        if count_r > best_count:
            best_count = count_r
            best_scores = scores_r.copy()
        
        # -degree * (1 + local_density)
        scores_ld = -deg_norm * (1.0 + 0.5 * ld_norm) + 0.5 * avg_nd_norm
        count_ld = greedy_decode(scores_ld)
        if count_ld > best_count:
            best_count = count_r
            best_scores = scores_ld.copy()
        
        # Prefer vertices whose neighbors are high-degree (they'd block a lot anyway)
        scores_hnd = -deg_norm + 1.5 * avg_nd_norm - 0.5 * leaf_norm
        count_hnd = greedy_decode(scores_hnd)
        if count_hnd > best_count:
            best_count = count_hnd
            best_scores = scores_hnd.copy()
        
        # Combine relaxation with structural features
        scores_combo = 1.5 * relax_norm - deg_norm + 0.5 * avg_nd_norm - 0.3 * ld_norm
        count_combo = greedy_decode(scores_combo)
        if count_combo > best_count:
            best_count = count_combo
            best_scores = scores_combo.copy()
    
    # Final polish: small random perturbations to break ties
    if time.time() < deadline - 3:
        for trial in range(15):
            if time.time() > deadline - 2:
                break
            noise = np.random.randn(n) * 0.02
            perturbed = best_scores + noise
            count_p = greedy_decode(perturbed)
            if count_p > best_count:
                best_count = count_p
                best_scores = perturbed.copy()
    
    if best_scores is None:
        return -degrees
    
    return best_scores
