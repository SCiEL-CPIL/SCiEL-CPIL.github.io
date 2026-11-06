"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline safety
_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    # Initialize deadline tracking
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: stop if we've exceeded 145s (5s buffer before 150s limit)
    if time.time() - _start_time > 145:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    # Identify feasible nodes (demand fits in remaining capacity)
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # If only one feasible node, return it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # --- Regret-2 Heuristic Strategy ---
    # Goal: Prioritize nodes that have a large difference between their best
    # and second-best "connection cost" among feasible alternatives.
    # This captures the "opportunity cost" of not visiting a node now:
    # if a node is much better connected to the current node than to other
    # remaining nodes (or vice versa), visiting it now avoids a large detour later.
    
    # For each candidate node i, we estimate the cost of inserting it into the route.
    # A simple proxy is the distance from the current node to i.
    # However, regret-based selection looks at the difference between the best
    # and second-best options for *that specific node* in terms of how well it
    # fits into the remaining structure.
    
    # Let's define a "connection cost" for a candidate node i relative to the current state.
    # The most immediate cost is d(current, i).
    # But to use regret, we need to compare how "good" node i is relative to other nodes.
    
    # Alternative interpretation of Regret-2 in VRP context:
    # For each unvisited node i, consider the cost of visiting it from the current node.
    # Let c_i = d(current, i).
    # This is just nearest neighbor.
    
    # A better regret metric:
    # For each candidate i, look at the two smallest distances from i to any other feasible node j (j != i).
    # If the difference (2nd_min - 1st_min) is large, it means node i is isolated from the rest of the cluster.
    # Visiting isolated nodes early might be bad because you have to go far to get back to the cluster.
    # Conversely, if the difference is small, node i is well-connected.
    
    # Actually, the standard Regret-k heuristic in CVRP usually works on *edges* or *insertions*.
    # Since we are picking a single next node, let's use a "Regret of Delay" approach.
    # Score(i) = d(current, i) - penalty_if_delayed(i)
    # penalty_if_delayed(i) is high if i has few close neighbors among the *remaining* feasible nodes.
    
    # Let's compute for each feasible node i:
    # 1. d_cur[i] = distance_matrix[current_node, i]
    # 2. Find the K closest feasible neighbors for i (excluding i itself).
    # 3. Let min_dist_1[i] be the distance to the closest neighbor.
    # 4. Let min_dist_2[i] be the distance to the second closest neighbor.
    # 5. Regret[i] = min_dist_2[i] - min_dist_1[i]
    #    If Regret is high, the node is on the periphery (isolated).
    #    If Regret is low, the node is in a dense cluster.
    
    # We want to minimize total distance.
    # Visiting a node in a dense cluster allows us to "batch" demands.
    # Visiting an isolated node forces a long detour.
    # So we should prefer nodes with LOW regret (well-connected) or nodes that are CLOSE to current.
    
    # Let's combine:
    # Score(i) = d_cur[i] + beta * Regret[i]
    # We want to minimize Score.
    # Low d_cur is good.
    # Low Regret (dense) is good.
    
    F = len(feasible_nodes)
    if F < 2:
        return int(feasible_nodes[0])
    
    # Distance from current to all feasible
    d_cur = distance_matrix[current_node, feasible_nodes]
    
    # Pairwise distances among feasible nodes
    dist_sub = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    dist_for_knn = dist_sub.copy()
    np.fill_diagonal(dist_for_knn, np.inf)
    
    # Find the two smallest distances for each node
    # We only need the 1st and 2nd smallest.
    # np.argpartition is faster than full sort.
    # We want the indices of the 2 smallest elements in each row.
    # But we just need the values.
    # Let's use np.partition to get the 2 smallest values.
    # axis=1, kth=1 (0-indexed, so 2nd smallest is at index 1)
    
    # Note: np.partition returns the array with the kth element in its sorted position,
    # but elements before and after are arbitrary.
    # So partition[:, :2] will contain the two smallest, but not necessarily sorted.
    # We can sort that small subarray.
    
    # If F is small, sorting is fine. F <= 50.
    # Let's just sort for simplicity and robustness, it's very fast for 50x50.
    sorted_dists = np.sort(dist_for_knn, axis=1)
    
    # 1st closest distance (index 0)
    min_dist_1 = sorted_dists[:, 0]
    
    # 2nd closest distance (index 1)
    if F > 2:
        min_dist_2 = sorted_dists[:, 1]
    else:
        # If only 2 nodes, the 2nd closest is the other one.
        # Actually if F==2, sorted_dists[:, 0] is dist to the other, sorted_dists[:, 1] is inf (self).
        # Wait, we set diagonal to inf.
        # For F=2:
        # Node 0: dist to 1 is d. dist to 0 is inf. Sorted: [d, inf].
        # Node 1: dist to 0 is d. dist to 1 is inf. Sorted: [d, inf].
        # So min_dist_2 is inf. Regret is inf.
        # This would make both nodes have infinite regret.
        # In this case, we should just fall back to nearest neighbor or handle it.
        # If F==2, let's just pick the one closer to current.
        return int(feasible_nodes[np.argmin(d_cur)])

    # Regret = 2nd min - 1st min
    regret = min_dist_2 - min_dist_1
    
    # Normalize regret to be comparable to distance?
    # Distances are likely in [0, 100]. Regret is also in that range.
    # Let's use a weighted sum.
    # Score = d_cur + alpha * regret
    
    # We need to tune alpha.
    # If alpha=0, it's nearest neighbor.
    # If alpha is high, we strongly prefer dense clusters.
    # Let's try alpha = 0.5.
    
    # However, regret can be very large for isolated nodes, which is good (we avoid them).
    # But if ALL nodes are isolated, regret is high for all, and d_cur dominates.
    
    # Let's normalize d_cur and regret to [0, 1] to balance them properly.
    d_min, d_max = np.min(d_cur), np.max(d_cur)
    r_min, r_max = np.min(regret), np.max(regret)
    
    if d_max > d_min:
        norm_d = (d_cur - d_min) / (d_max - d_min)
    else:
        norm_d = np.zeros_like(d_cur)
        
    if r_max > r_min:
        norm_r = (regret - r_min) / (r_max - r_min)
    else:
        norm_r = np.zeros_like(regret)
        
    # Score = norm_d + alpha * norm_r
    # We want to minimize this.
    alpha = 0.5
    scores = norm_d + alpha * norm_r
    
    idx = np.argmin(scores)
    best_node = feasible_nodes[idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
