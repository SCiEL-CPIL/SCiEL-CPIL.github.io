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
        if time.time() > deadline - 30:
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
    
    # Precompute neighbor masks for fast vectorized greedy simulation
    # Each row is a bitmask of neighbors (using uint64 for n<=64, but n=200 so use object or multiple)
    # Better: use a list of boolean arrays for neighbor masks
    neighbor_masks = [adjacency[i].copy() for i in range(n)]
    
    # Vectorized greedy decode using precomputed neighbor masks
    def greedy_decode(scores):
        order = np.argsort(-scores)
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if blocked[v]:
                continue
            count += 1
            blocked[v] = True
            # Mark all neighbors as blocked
            blocked |= neighbor_masks[v]
        return count
    
    # Iterative relaxation score: captures global structural balance
    # score_i = 1 / (1 + sum_{j in N(i)} score_j)
    # Start with uniform scores, iterate
    relax_scores = np.ones(n)
    for _ in range(20):
        if time.time() > deadline - 25:
            break
        neighbor_score_sums = adjacency.dot(relax_scores)
        relax_scores_new = 1.0 / (1.0 + neighbor_score_sums)
        relax_scores = relax_scores_new
    
    # Feature matrix: [deg, avg_nd, local_density, local_edges, avg_nl, relax]
    # Try linear combinations with various weights
    best_score_val = -1
    best_scores = None
    
    # First pass: try a set of hand-crafted scoring functions
    weight_sets = [
        # (w_deg, w_avgnd, w_ld, w_le, w_anld, w_relax)
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0),       # min degree
        (-1.0, 1.0, 0.0, 0.0, 0.0, 0.0),       # avg_nd - deg
        (-1.0, 0.5, 0.0, 0.0, 0.0, 0.0),       # 0.5*avg_nd - deg
        (-0.5, 0.5, 0.0, 0.0, 0.0, 0.0),       # 0.5*(avg_nd - deg)
        (-1.0, 0.0, 0.0, -1.0, 0.0, 0.0),      # -deg - local_edges
        (-1.0, 0.0, -1.0, 0.0, 0.0, 0.0),      # -deg - local_density
        (-1.0, 0.5, -0.5, 0.0, 0.0, 0.0),      # 0.5*avg_nd - deg - 0.5*local_density
        (-1.0, 1.0, -1.0, 0.0, 0.0, 0.0),      # avg_nd - deg - local_density
        (-1.0, 0.0, 0.0, -0.5, 0.0, 0.0),      # -deg - 0.5*local_edges
        (-1.0, 0.5, 0.0, -0.5, 0.0, 0.0),      # 0.5*avg_nd - deg - 0.5*local_edges
        (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0),      # -deg - avg_neighbor_local_density
        (-1.0, 0.5, 0.0, 0.0, -0.5, 0.0),      # 0.5*avg_nd - deg - 0.5*avg_nl
        (-1.0, 1.0, -0.5, -0.5, 0.0, 0.0),     # avg_nd - deg - 0.5*ld - 0.5*le
        (-1.0, 0.5, -0.5, -0.5, 0.0, 0.0),     # 0.5*avg_nd - deg - 0.5*ld - 0.5*le
        (-0.5, 0.5, -0.5, -0.5, 0.0, 0.0),     # 0.5*(avg_nd - deg) - 0.5*ld - 0.5*le
        (-1.0, 0.0, 0.0, -1.0, -0.5, 0.0),     # -deg - le - 0.5*avg_nl
        (-1.0, 1.0, 0.0, -1.0, 0.0, 0.0),      # avg_nd - deg - le
        (-1.0, 0.5, 0.0, -1.0, 0.0, 0.0),      # 0.5*avg_nd - deg - le
        (-2.0, 1.0, 0.0, 0.0, 0.0, 0.0),       # avg_nd - 2*deg
        (-1.0, 2.0, 0.0, 0.0, 0.0, 0.0),       # 2*avg_nd - deg
        # New: with relaxation scores
        (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),        # relax only
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.5),       # -deg + 0.5*relax
        (-1.0, 0.5, 0.0, 0.0, 0.0, 0.5),       # 0.5*avg_nd - deg + 0.5*relax
        (-1.0, 1.0, 0.0, 0.0, 0.0, 0.5),       # avg_nd - deg + 0.5*relax
        (-1.0, 0.5, -0.5, 0.0, 0.0, 0.5),      # 0.5*avg_nd - deg - 0.5*ld + 0.5*relax
        (-1.0, 1.0, -1.0, 0.0, 0.0, 0.5),      # avg_nd - deg - ld + 0.5*relax
        (-1.0, 0.5, 0.0, -0.5, 0.0, 0.5),      # 0.5*avg_nd - deg - 0.5*le + 0.5*relax
        (-1.0, 1.0, 0.0, -1.0, 0.0, 0.5),      # avg_nd - deg - le + 0.5*relax
        (-1.0, 0.5, -0.5, -0.5, 0.0, 0.5),     # 0.5*avg_nd - deg - 0.5*ld - 0.5*le + 0.5*relax
        (-1.0, 1.0, -0.5, -0.5, 0.0, 0.5),     # avg_nd - deg - 0.5*ld - 0.5*le + 0.5*relax
        # Higher weight on relaxation
        (-1.0, 0.0, 0.0, 0.0, 0.0, 1.0),       # -deg + relax
        (-1.0, 0.5, 0.0, 0.0, 0.0, 1.0),       # 0.5*avg_nd - deg + relax
        (-1.0, 1.0, 0.0, 0.0, 0.0, 1.0),       # avg_nd - deg + relax
        (-1.0, 0.5, -0.5, 0.0, 0.0, 1.0),      # 0.5*avg_nd - deg - 0.5*ld + relax
        (-1.0, 1.0, -1.0, 0.0, 0.0, 1.0),      # avg_nd - deg - ld + relax
        (-1.0, 0.5, 0.0, -0.5, 0.0, 1.0),      # 0.5*avg_nd - deg - 0.5*le + relax
        (-1.0, 1.0, 0.0, -1.0, 0.0, 1.0),      # avg_nd - deg - le + relax
        (-1.0, 0.5, -0.5, -0.5, 0.0, 1.0),     # 0.5*avg_nd - deg - 0.5*ld - 0.5*le + relax
        (-1.0, 1.0, -0.5, -0.5, 0.0, 1.0),     # avg_nd - deg - 0.5*ld - 0.5*le + relax
        # Even higher weight
        (-1.0, 0.0, 0.0, 0.0, 0.0, 2.0),       # -deg + 2*relax
        (-1.0, 0.5, 0.0, 0.0, 0.0, 2.0),       # 0.5*avg_nd - deg + 2*relax
        (-1.0, 1.0, 0.0, 0.0, 0.0, 2.0),       # avg_nd - deg + 2*relax
        (-1.0, 0.5, -0.5, 0.0, 0.0, 2.0),      # 0.5*avg_nd - deg - 0.5*ld + 2*relax
        (-1.0, 1.0, -1.0, 0.0, 0.0, 2.0),      # avg_nd - deg - ld + 2*relax
    ]
    
    for w_deg, w_avgnd, w_ld, w_le, w_anld, w_relax in weight_sets:
        if time.time() > deadline - 15:
            break
        scores = (w_deg * degrees + w_avgnd * avg_neighbor_degree + 
                  w_ld * local_density + w_le * local_edges + 
                  w_anld * avg_neighbor_local_density + w_relax * relax_scores)
        count = greedy_decode(scores)
        if count > best_score_val:
            best_score_val = count
            best_scores = scores.copy()
            best_w = (w_deg, w_avgnd, w_ld, w_le, w_anld, w_relax)
    
    # Second pass: local search around the best weights found
    if best_scores is not None and time.time() < deadline - 10:
        if 'best_w' not in dir() or best_w is None:
            best_w = (-1.0, 0.5, 0.0, -0.5, 0.0, 0.5)
        
        base_w = list(best_w)
        for dim in range(6):
            for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0]:
                if time.time() > deadline - 5:
                    break
                w = list(base_w)
                w[dim] += delta
                scores = (w[0] * degrees + w[1] * avg_neighbor_degree + 
                          w[2] * local_density + w[3] * local_edges + 
                          w[4] * avg_neighbor_local_density + w[5] * relax_scores)
                count = greedy_decode(scores)
                if count > best_score_val:
                    best_score_val = count
                    best_scores = scores.copy()
                    base_w = w
    
    # Third pass: try a few more specialized strategies
    if time.time() < deadline - 5:
        # Strategy: score by -degree * (1 + local_density) 
        scores1 = -degrees * (1.0 + local_density)
        count = greedy_decode(scores1)
        if count > best_score_val:
            best_score_val = count
            best_scores = scores1.copy()
        
        # Strategy: -degree + sum of neighbor degrees
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
        
        # Strategy: -degree * (2 - local_density)
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
        
        # Strategy: relax_scores combined with degree
        scores6 = relax_scores - 0.5 * degrees
        count = greedy_decode(scores6)
        if count > best_score_val:
            best_score_val = count
            best_scores = scores6.copy()
        
        scores7 = relax_scores - degrees
        count = greedy_decode(scores7)
        if count > best_score_val:
            best_score_val = count
            best_scores = scores7.copy()
        
        scores8 = relax_scores * 2.0 - degrees
        count = greedy_decode(scores8)
        if count > best_score_val:
            best_score_val = count
            best_scores = scores8.copy()
        
        # Strategy: normalize relax_scores and combine
        relax_norm = relax_scores - relax_scores.mean()
        scores9 = relax_norm - 0.5 * (degrees - degrees.mean())
        count = greedy_decode(scores9)
        if count > best_score_val:
            best_score_val = count
            best_scores = scores9.copy()
        
        scores10 = relax_norm - (degrees - degrees.mean())
        count = greedy_decode(scores10)
        if count > best_score_val:
            best_score_val = count
            best_scores = scores10.copy()
        
        # Final polish: try small perturbations of the best scores found
        if best_scores is not None and time.time() < deadline - 3:
            # Add small random noise to break ties and explore nearby orderings
            for trial in range(10):
                if time.time() > deadline - 2:
                    break
                noise = np.random.randn(n) * 0.01
                perturbed = best_scores + noise
                count = greedy_decode(perturbed)
                if count > best_score_val:
                    best_score_val = count
                    best_scores = perturbed.copy()
    
    if best_scores is None:
        return -degrees

    return best_scores
