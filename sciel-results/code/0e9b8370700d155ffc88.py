"""Guided local search penalty for TSP: geometric curvature disruption of sharp turns."""
import numpy as np
import time

_DEADLINE = time.perf_counter() + 310.0  # safety margin before the 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    n = edge_distance.shape[0]
    if n <= 2:
        return edge_distance.copy()

    # True distances (always input is the true matrix)
    d = np.asarray(edge_distance, dtype=float)

    # Build the edge list of the current tour (closed cycle)
    tour = np.asarray(local_opt_tour, dtype=int)
    # Ensure length == n and is a permutation; fallback to identity if malformed
    if tour.shape[0] != n:
        tour = np.arange(n)

    # Edge endpoints for the closed tour
    i = tour[:-1]
    j = tour[1:]
    # closing edge
    i = np.append(i, tour[0])
    j = np.append(j, tour[-1])

    # Lengths of each tour edge (true)
    tour_len = d[i, j]

    # Build a penalty matrix (additive), zero by default
    penalty = np.zeros((n, n), dtype=float)

    # Base usage-based mild penalty for all edges seen before (symmetric)
    usage = np.asarray(edge_n_used, dtype=float)
    usage = np.maximum(usage, 0.0)
    # Use smooth saturation: u/(1+u) which is bounded in [0,1) and smooth
    usage_scaled = usage / (1.0 + usage)
    # Add a small uniform-ish penalty weighted by usage (symmetric)
    penalty += 0.03 * usage_scaled

    # Geometric curvature penalty: for each node in the tour, compute the angle
    # between the two incident edges. Sharp turns (small angle) indicate
    # topological inefficiency. Penalize edges incident to high-curvature nodes.
    #
    # For node tour[k], the two incident edges are (tour[k-1], tour[k]) and (tour[k], tour[k+1]).
    # The "turn angle" is the angle at node tour[k] between vectors to its neighbors.
    # A straight path has angle 180° (pi), a U-turn has angle 0°.
    # We want to penalize small angles (sharp turns).
    
    # Compute vectors from each node to its two tour neighbors
    # For node at position k in tour: prev = tour[k-1], next = tour[(k+1)%n]
    # Vector v1 = pos[prev] - pos[node], v2 = pos[next] - pos[node]
    # cos(angle) = dot(v1, v2) / (|v1| * |v2|)
    
    # We don't have coordinates, only the distance matrix. But we can use
    # the law of cosines: for triangle (prev, node, next):
    # d(prev,next)^2 = d(prev,node)^2 + d(node,next)^2 - 2*d(prev,node)*d(node,next)*cos(angle)
    # cos(angle) = (d(prev,node)^2 + d(node,next)^2 - d(prev,next)^2) / (2*d(prev,node)*d(node,next))
    
    # For each position k in the tour:
    # prev = tour[(k-1) % n], node = tour[k], next = tour[(k+1) % n]
    prev_nodes = np.roll(tour, 1)   # tour[k-1]
    next_nodes = np.roll(tour, -1)  # tour[k+1]
    
    d_prev_node = d[prev_nodes, tour]   # d(prev, node)
    d_node_next = d[tour, next_nodes]   # d(node, next)
    d_prev_next = d[prev_nodes, next_nodes]  # d(prev, next)
    
    # cos(angle) via law of cosines
    denom = 2.0 * d_prev_node * d_node_next
    # Guard against zero denominators
    denom_safe = np.where(denom > 1e-12, denom, 1.0)
    cos_angle = (d_prev_node**2 + d_node_next**2 - d_prev_next**2) / denom_safe
    # Clip to [-1, 1] for numerical safety
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Curvature measure: deviation from straight (pi). 
    # angle = arccos(cos_angle). Sharp turn = small angle.
    # We use (1 - cos_angle) as a proxy: 0 for straight, 2 for U-turn.
    # Actually, let's use sin(angle/2)^2 = (1 - cos_angle) / 2 as curvature.
    # Or simply: curvature = 1 - cos_angle (ranges 0 to 2)
    curvature = 1.0 - cos_angle  # 0 = straight, 2 = U-turn
    
    # Rank nodes by curvature and penalize the sharpest turns
    # Top fraction of nodes with highest curvature get heavy penalties on their incident edges
    k_curv = max(1, int(round(0.30 * n)))  # top 30% of nodes by curvature
    # Get indices of top-k curvature nodes
    top_curv_idx = np.argsort(curvature)[-k_curv:]
    
    # For each high-curvature node, add penalty to its two incident tour edges
    # Penalty magnitude scales with curvature and local edge lengths
    for k in top_curv_idx:
        node = tour[k]
        prev = prev_nodes[k]
        next = next_nodes[k]
        
        # Penalty on edge (prev, node) and edge (node, next)
        # Magnitude proportional to curvature and average local edge length
        local_avg_len = 0.5 * (d_prev_node[k] + d_node_next[k])
        mag = 0.4 * curvature[k] * local_avg_len + 0.15 * np.mean(tour_len)
        
        penalty[prev, node] += mag
        penalty[node, prev] += mag
        penalty[node, next] += mag
        penalty[next, node] += mag
    
    # Symmetrize to be safe
    penalty = 0.5 * (penalty + penalty.T)

    # Deadline check: if we're running out of time, return immediately
    if time.perf_counter() > _DEADLINE:
        return d + penalty
    
    # Result: true distance + penalty. Keep finite.
    result = d + penalty
    # Ensure finiteness (defensive)
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=0.0)

    return result
