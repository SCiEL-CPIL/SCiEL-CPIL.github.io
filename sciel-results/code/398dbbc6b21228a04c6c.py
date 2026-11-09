"""Capacitated vehicle routing: pick the next customer or return to depot."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Deadline guard: the evaluator kills the process at 150s wall-clock.
# We stop doing heavy work once we get close to that limit.
_T0 = time.perf_counter()
_DEADLINE = _T0 + 145.0  # 5s safety margin


def _deadline_exceeded():
    return time.perf_counter() > _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer to visit, or return 0 (depot) to close the route.
    
    Strategy: Hybrid of Clarke-Wright savings and density-based clustering.
    
    1. Primary score: savings-based (minimize dist_current - dist_depot)
       This maximizes the Clarke-Wright savings of extending the route.
    
    2. Secondary adjustment: density-weighted centroid deviation.
       Compute the centroid of all feasible unvisited nodes (weighted by inverse
       demand to favor high-density, low-demand clusters). Score candidates by
       their deviation from this centroid relative to distance from current node.
       This encourages maintaining spatial density and minimizing intra-cluster
       travel variance.
    
    3. Capacity-aware depot penalty: when capacity is tight, add a penalty for
       nodes far from depot to avoid ending up stranded.
    
    The final score is a weighted combination:
        score = w1 * savings_score + w2 * density_score + w3 * depot_penalty
    
    where weights adapt based on capacity ratio.
    """
    if len(unvisited_nodes) == 0:
        return 0
    
    # Filter to feasible nodes (demand <= rest_capacity)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0
    
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Deadline check: if we're close to the limit, just pick nearest feasible node
    if _deadline_exceeded():
        dists = distance_matrix[current_node, feasible_nodes]
        best_idx = np.argmin(dists)
        return int(feasible_nodes[best_idx])
    
    # Compute distances
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    dists_to_depot = distance_matrix[feasible_nodes, depot]
    demands_feasible = demands[feasible_nodes]
    
    # Total demand of all unvisited feasible nodes
    total_demand = np.sum(demands_feasible)
    
    # Capacity ratio: how much capacity we have relative to remaining demand
    capacity_ratio = rest_capacity / max(total_demand, 1.0)
    
    # === Component 1: Clarke-Wright savings score ===
    # savings = dist_current_to_depot - dist_current_to_j + dist_j_to_depot
    # But we want to minimize, so: score_savings = dist_current_to_j - dist_j_to_depot
    # This is equivalent to maximizing savings.
    dist_current_to_depot = distance_matrix[current_node, depot]
    savings_score = dists_from_current - dists_to_depot
    
    # Normalize savings_score to [0, 1] for combining
    min_sav = np.min(savings_score)
    max_sav = np.max(savings_score)
    if max_sav > min_sav:
        savings_norm = (savings_score - min_sav) / (max_sav - min_sav)
    else:
        savings_norm = np.zeros_like(savings_score)
    
    # === Component 2: Density-weighted centroid deviation ===
    # Compute centroid of feasible nodes, weighted by inverse demand
    # (higher weight for low-demand nodes to encourage dense, easy-to-serve clusters)
    inv_demands = 1.0 / np.maximum(demands_feasible, 1e-6)
    inv_demands_norm = inv_demands / np.sum(inv_demands)
    
    # Get coordinates via distance matrix structure
    # Since we only have distance_matrix, we approximate position using
    # distances to depot and current node as a 2D proxy
    # Use (dist_to_depot, dist_from_current) as a 2D coordinate proxy
    x_coords = dists_to_depot
    y_coords = dists_from_current
    
    # Weighted centroid
    centroid_x = np.sum(inv_demands_norm * x_coords)
    centroid_y = np.sum(inv_demands_norm * y_coords)
    
    # Deviation from centroid (Euclidean distance in proxy space)
    deviations = np.sqrt((x_coords - centroid_x)**2 + (y_coords - centroid_y)**2)
    
    # Normalize deviation to [0, 1]
    min_dev = np.min(deviations)
    max_dev = np.max(deviations)
    if max_dev > min_dev:
        density_score = (deviations - min_dev) / (max_dev - min_dev)
    else:
        density_score = np.zeros_like(deviations)
    
    # === Component 3: Depot proximity penalty (capacity-aware) ===
    # When capacity is tight, penalize nodes far from depot
    # Normalize dist_to_depot to [0, 1]
    min_dd = np.min(dists_to_depot)
    max_dd = np.max(dists_to_depot)
    if max_dd > min_dd:
        depot_penalty = (dists_to_depot - min_dd) / (max_dd - min_dd)
    else:
        depot_penalty = np.zeros_like(dists_to_depot)
    
    # === Combine components with adaptive weights ===
    # When capacity is plentiful (high ratio), prioritize savings and density
    # When capacity is tight (low ratio), add depot penalty
    if capacity_ratio >= 1.0:
        w_savings = 0.5
        w_density = 0.4
        w_depot = 0.1
    else:
        # As capacity gets tighter, shift weight toward depot proximity
        tightness = 1.0 - min(capacity_ratio, 1.0)
        w_savings = 0.4 + 0.1 * (1.0 - tightness)
        w_density = 0.3 + 0.1 * (1.0 - tightness)
        w_depot = 0.2 + 0.3 * tightness
    
    final_score = w_savings * savings_norm + w_density * density_score + w_depot * depot_penalty
    
    best_idx = np.argmin(final_score)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
