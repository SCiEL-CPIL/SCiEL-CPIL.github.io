import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
        
    degrees = adjacency.sum(axis=1).astype(float)
    
    # Precompute features
    sum_neighbor_degrees = adjacency.dot(degrees)
    avg_neighbor_degree = np.zeros(n)
    nonzero_mask = degrees > 0
    avg_neighbor_degree[nonzero_mask] = sum_neighbor_degrees[nonzero_mask] / degrees[nonzero_mask]
    
    # Local edge density: number of edges among neighbors (measures "waste" of selecting v)
    local_edges = np.zeros(n)
    for i in range(n):
        if time.time() > deadline - 20:
            break
        d = int(degrees[i])
        if d > 1:
            neighbors = np.where(adjacency[i])[0]
            sub = adjacency[np.ix_(neighbors, neighbors)]
            local_edges[i] = sub.sum() / 2.0
        elif d == 1:
            local_edges[i] = 0.0
        else:
            local_edges[i] = 0.0
    
    # Normalized local edge density
    possible_edges = degrees * (degrees - 1) / 2.0
    local_density = np.zeros(n)
    valid = possible_edges > 0
    local_density[valid] = local_edges[valid] / possible_edges[valid]
    
    # Second-order: average local density of neighbors
    avg_neighbor_local_density = np.zeros(n)
    nz = degrees > 0
    if nz.any():
        sum_nl = adjacency.dot(local_density)
        avg_neighbor_local_density[nz] = sum_nl[nz] / degrees[nz]
    
    # Degree squared (penalize high-degree vertices more)
    degree_sq = degrees ** 2

    # Precompute neighbor lists for fast greedy simulation
    neighbor_lists = [np.where(adjacency[i])[0] for i in range(n)]
    
    def greedy_decode(scores):
        order = np.argsort(-scores)
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if selected[v]:
                continue
            selected[v] = True
            count += 1
            if len(neighbor_lists[v]) > 0:
                selected[neighbor_lists[v]] = True
        return count
    
    # Feature matrix: [deg, avg_nd, local_density, local_edges, avg_nl]
    # Try linear combinations with various weights
    best_score_val = -1
    best_scores = None
    
    # First pass: try a set of hand-crafted scoring functions
    def make_scorer(w_deg, w_avgnd, w_ld, w_le, w_anld):
        def scorer():
            return (w_deg * degrees + w_avgnd * avg_neighbor_degree + 
                    w_ld * local_density + w_le * local_edges + w_anld * avg_neighbor_local_density)
        return scorer
    
    # Key insight: we want to select vertices that have low degree and whose neighbors are poorly connected
    # (low local_edges means neighbors don't block each other as much)
    # Actually, for MIS: selecting v removes v and all its neighbors from consideration.
    # We want v to "cover" as few other potential candidates as possible.
    # So low degree is good, and we want neighbors that are also high-degree (so they'd be bad candidates anyway).
    
    weight_sets = [
        # (w_deg, w_avgnd, w_ld, w_le, w_anld)
        (-1.0, 0.0, 0.0, 0.0, 0.0),       # min degree
        (-1.0, 1.0, 0.0, 0.0, 0.0),       # avg_nd - deg
        (-1.0, 0.5, 0.0, 0.0, 0.0),       # 0.5*avg_nd - deg
        (-0.5, 0.5, 0.0, 0.0, 0.0),       # 0.5*(avg_nd - deg)
        (-1.0, 0.0, 0.0, -1.0, 0.0),      # -deg - local_edges
        (-1.0, 0.0, -1.0, 0.0, 0.0),      # -deg - local_density
        (-1.0, 0.5, -0.5, 0.0, 0.0),      # 0.5*avg_nd - deg - 0.5*local_density
        (-1.0, 1.0, -1.0, 0.0, 0.0),      # avg_nd - deg - local_density
        (-1.0, 0.0, 0.0, -0.5, 0.0),      # -deg - 0.5*local_edges
        (-1.0, 0.5, 0.0, -0.5, 0.0),      # 0.5*avg_nd - deg - 0.5*local_edges
        (-1.0, 0.0, 0.0, 0.0, -1.0),      # -deg - avg_neighbor_local_density
        (-1.0, 0.5, 0.0, 0.0, -0.5),      # 0.5*avg_nd - deg - 0.5*avg_nl
        (-1.0, 1.0, -0.5, -0.5, 0.0),     # avg_nd - deg - 0.5*ld - 0.5*le
        (-1.0, 0.5, -0.5, -0.5, 0.0),     # 0.5*avg_nd - deg - 0.5*ld - 0.5*le
        (-0.5, 0.5, -0.5, -0.5, 0.0),     # 0.5*(avg_nd - deg) - 0.5*ld - 0.5*le
        (-1.0, 0.0, 0.0, -1.0, -0.5),     # -deg - le - 0.5*avg_nl
        (-1.0, 1.0, 0.0, -1.0, 0.0),      # avg_nd - deg - le
        (-1.0, 0.5, 0.0, -1.0, 0.0),      # 0.5*avg_nd - deg - le
        (-2.0, 1.0, 0.0, 0.0, 0.0),       # avg_nd - 2*deg
        (-1.0, 2.0, 0.0, 0.0, 0.0),       # 2*avg_nd - deg
    ]
    
    for w_deg, w_avgnd, w_ld, w_le, w_anld in weight_sets:
        if time.time() > deadline - 10:
            break
        scores = (w_deg * degrees + w_avgnd * avg_neighbor_degree + 
                  w_ld * local_density + w_le * local_edges + w_anld * avg_neighbor_local_density)
        count = greedy_decode(scores)
        if count > best_score_val:
            best_score_val = count
            best_scores = scores.copy()
    
    # Second pass: local search around the best weights found
    if best_scores is not None:
        # Find which weight set was best
        best_w = None
        for w_deg, w_avgnd, w_ld, w_le, w_anld in weight_sets:
            scores = (w_deg * degrees + w_avgnd * avg_neighbor_degree + 
                      w_ld * local_density + w_le * local_edges + w_anld * avg_neighbor_local_density)
            if np.allclose(scores, best_scores, atol=1e-10):
                best_w = (w_deg, w_avgnd, w_ld, w_le, w_anld)
                break
        
        if best_w is None:
            best_w = (-1.0, 0.5, 0.0, -0.5, 0.0)  # default
        
        # Local search: perturb each weight
        base_w = list(best_w)
        for dim in range(5):
            for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0]:
                if time.time() > deadline - 5:
                    break
                w = list(base_w)
                w[dim] += delta
                scores = (w[0] * degrees + w[1] * avg_neighbor_degree + 
                          w[2] * local_density + w[3] * local_edges + w[4] * avg_neighbor_local_density)
                count = greedy_decode(scores)
                if count > best_score_val:
                    best_score_val = count
                    best_scores = scores.copy()
                    base_w = w  # update base for further perturbations
    
    # Third pass: try a few more specialized strategies
    if time.time() < deadline - 5:
        # Strategy: score by -degree * (1 + local_density) 
        # Vertices with many edges among neighbors are "wasteful" to select
        scores1 = -degrees * (1.0 + local_density)
        count = greedy_decode(scores1)
        if count > best_score_val:
            best_score_val = count
            best_scores = scores1.copy()
        
        # Strategy: score by -degree / (1 + local_edges)
        # High local edges means neighbors are interconnected, so selecting v is less wasteful
        # Wait, that's wrong. If neighbors are interconnected, selecting v removes many potential candidates.
        # Actually for MIS, we want to select vertices whose neighbors are NOT good candidates.
        # So we want neighbors with high degree (they'd be skipped anyway) and low local connectivity.
        
        # Strategy: -degree + sum of neighbor degrees (prefer vertices whose neighbors are high-degree)
        # This is already captured by avg_nd - deg
        scores2 = -degrees + sum_neighbor_degrees / (degrees + 1.0)
        count = greedy_decode(scores2)
        if count > best_score_val:
            best_score_val = count
            best_scores = scores2.copy()
        
        # Strategy: prefer low degree, and among equal degree, prefer those with neighbors that are high-degree
        scores3 = -degrees * 10.0 + avg_neighbor_degree
        count = greedy_decode(scores3)
        if count > best_score_val:
            best_score_val = count
            best_scores = scores3.copy()
        
        # Strategy: -degree * (2 - local_density) — penalize more when local density is low
        # (when neighbors are not connected, selecting v wastes more potential picks)
        scores4 = -degrees * (2.0 - local_density)
        count = greedy_decode(scores4)
        if count > best_score_val:
            best_score_val = count
            best_scores = scores4.copy()
        
        # Strategy: combine -degree with preference for neighbors having high avg neighbor degree
        scores5 = -degrees + 0.5 * avg_neighbor_local_density
        count = greedy_decode(scores5)
        if count > best_score_val:
            best_score_val = count
            best_scores = scores5.copy()
    
    if best_scores is None:
        return -degrees

    return best_scores
