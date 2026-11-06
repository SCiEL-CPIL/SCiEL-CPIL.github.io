"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time once at module load
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds
_SAFETY_MARGIN = 2.0  # return safe result this far before deadline

def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: Pick next customer or depot (0) using Sweep-Angle Priority.
    
    Strategy:
    1. Filter unvisited nodes by capacity.
    2. If none fit, return depot.
    3. If only one fits, return it.
    4. Compute the "sweep angle" for each feasible node relative to the current direction.
       The current direction is defined by the vector from the depot to the current node.
       We prefer nodes that maintain a consistent angular sector, minimizing the change in
       direction (sweep). This helps create contiguous routes that cover specific areas
       of the map, reducing backtracking and fragmentation.
    5. To handle the circular nature of angles and the fact that we might be closing a route,
       we also consider the distance to the depot. A node that is close in angle but far away
       might be bad if it forces a long return trip. However, the primary signal is angular
       consistency.
       
       Score = alpha * angular_deviation + beta * (distance_to_depot / max_dist)
       
       Actually, pure angle might lead to picking a very far node if it's in the same sector.
       We need to balance angular consistency with proximity. 
       
       Let's refine: 
       The "Sweep" heuristic typically sorts nodes by angle around the depot and visits them
       in that order. Here we are making a greedy choice.
       
       Let's compute the angle of the vector (depot -> current) and (depot -> candidate).
       The deviation is the absolute difference in angles (wrapped to [-pi, pi]).
       
       We also want to avoid picking a node that is very far from the current node if the
       angular benefit is small.
       
       Let's use a composite score:
       score = angular_deviation * scale + normalized_distance_to_candidate
       
       Wait, the standard "Sweep" is a construction heuristic that sorts by angle.
       If we are already in the middle of a route, we should probably continue in the same
       angular direction.
       
       Let's define:
       vec_current = coords[current] - coords[depot]
       vec_cand = coords[cand] - coords[depot]
       
       angle_current = atan2(vec_current.y, vec_current.x)
       angle_cand = atan2(vec_cand.y, vec_cand.x)
       
       delta_angle = abs(angle_cand - angle_current)
       delta_angle = min(delta_angle, 2*pi - delta_angle)
       
       We want small delta_angle.
       
       However, if we are near the end of the route (low rest capacity or few nodes left),
       we should prioritize closing the route (returning to depot).
       
       Let's check rest_capacity. If rest_capacity is low, we should pick a node that allows
       us to return to depot? No, we just pick the next node. The next node is picked, then
       we check if we can continue.
       
       If we pick a node, we add its demand to the load.
       
       Let's stick to the angular deviation as the primary metric, combined with a penalty
       for distance from current node to prevent jumping across the map if the angle is
       similar but distance is huge.
       
       score = w1 * delta_angle + w2 * (dist(current, cand) / max_dist)
       
       Also, if the number of remaining feasible nodes is small, we should consider
       the cost of returning to depot from the candidate.
       
       Let's add a term: w3 * dist(cand, depot)
       
       Weights:
       w1 = 1.0 (Angle is key for sweep)
       w2 = 0.5 (Distance from current)
       w3 = 0.5 (Distance to depot, to avoid getting stuck far away)
    """
    if len(unvisited_nodes) == 0:
        return depot

    # If time is running out, fall back to simple nearest feasible
    if _time_exceeded():
        feasible_mask = np.array([demands[i] <= rest_capacity for i in unvisited_nodes])
        feasible_nodes = unvisited_nodes[feasible_mask]
        if len(feasible_nodes) == 0:
            return depot
        return int(feasible_nodes[np.argmin(distance_matrix[current_node, feasible_nodes])])

    # Filter by capacity
    feasible_mask = np.array([demands[i] <= rest_capacity for i in unvisited_nodes])
    feasible_nodes = unvisited_nodes[feasible_mask]

    if len(feasible_nodes) == 0:
        return depot

    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # We need coordinates to compute angles.
    # The distance_matrix is provided. We can infer coordinates if we assume Euclidean?
    # No, distance_matrix is arbitrary. 
    # WAIT. The task says "distance_matrix". It does NOT provide coordinates.
    # We CANNOT compute angles without coordinates!
    
    # Re-reading the task:
    # "select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)"
    # Only distance_matrix is provided.
    # 
    # If coordinates are not provided, we cannot compute polar angles directly.
    # We must derive a proxy for "sweep angle" or "directional consistency" using only distances.
    #
    # Is there a way to infer direction?
    # If the metric is Euclidean, we could try to reconstruct coordinates (MDS), but that's expensive and might not be Euclidean.
    #
    # Let's look at the inspirations.
    # Inspiration 1: Savings-based (Clarke-Wright). Uses distances.
    # Inspiration 2: Savings-based.
    #
    # The directive says: "Sweep-Angle Priority Heuristic ... compute the polar angle ... relative to the depot ...".
    # This implies coordinates might be available or inferable?
    # "distance_matrix" is the only input.
    #
    # If I cannot compute angles, I cannot implement the Sweep-Angle heuristic as described in the directive literally.
    # However, the directive is a *strategy to try*. If the interface doesn't support it, I must adapt.
    #
    # Can I approximate "angular consistency" with distances?
    # Triangle inequality: d(A,B) + d(B,C) >= d(A,C).
    # If A, B, C are collinear and B is between A and C, d(A,C) = d(A,B) + d(B,C).
    # If they form a wide angle, d(A,C) is close to sqrt(d(A,B)^2 + d(B,C)^2).
    #
    # "Sweeping" implies visiting nodes in a specific angular order around the depot.
    # This is equivalent to minimizing the "turning angle" or maintaining a consistent direction.
    #
    # Without coordinates, a common heuristic that mimics sweep behavior is the **Nearest Neighbor** with a **Lookahead** or **Savings**.
    #
    # Let's look at the "Savings" approach again.
    # Savings s(i,j) = d(depot, i) + d(depot, j) - d(i, j).
    # High savings means i and j are close to each other relative to their distance to depot.
    # This effectively groups nodes that are "close" in angular and radial terms.
    #
    # The current program (score 0.99) uses:
    # score = d(current, cand) + 0.2 * min(d(cand, other_feasible)) + 0.1 * d(cand, depot)
    #
    # The inspirations (score 1.08) use Savings.
    #
    # The directive asks for a STRUCTURAL change to Sweep-Angle.
    # Since I can't compute angles, I will implement a **Directional Consistency** metric using the triangle inequality as a proxy for "straightness" or "sweep".
    #
    # If we assume the route is a path, "sweeping" means we don't double back.
    # A node C is "in the sweep direction" from A (current) if the vector A->C is similar to the previous direction.
    # We don't have the previous direction explicitly, but we have the depot.
    #
    # Alternative interpretation:
    # Maybe the distance_matrix IS Euclidean and I should extract coordinates?
    # No, that's risky and slow.
    #
    # Let's try a different structural change that fits the "Sweep" spirit without coordinates:
    # **Cluster-based Selection**.
    # Nodes that are close to the depot and close to each other form a "sweep sector".
    #
    # Actually, let's look at the constraints. "distance_matrix" is the only data.
    # If I cannot compute angles, I must simulate the effect of sweeping.
    # Sweeping minimizes the sum of turning angles.
    #
    # Let's try **Nearest Neighbor with a "Momentum" term**.
    # We don't have momentum.
    #
    # Let's try **Savings Heuristic** (Clarke-Wright) as a greedy construction step.
    # This is a different family from the current "distance + lookahead".
    # The current program minimizes immediate distance + local connectivity.
    # Savings maximizes the benefit of merging two paths.
    #
    # In a greedy construction step (adding one node at a time), the Savings heuristic is not directly applicable in the same way as in the merge phase.
    # However, we can score a candidate node `c` by how well it fits into the existing route.
    #
    # Let's try: **Score = d(current, c) + d(c, depot) - lambda * min(d(c, other_feasible))**
    # This is similar to the current one but with different signs/weights?
    # Current: d(cur,c) + 0.2*min(d(c,other)) + 0.1*d(c,depot)
    #
    # Let's try a **Farthest-Neighbor** variant? No, that's bad for TSP.
    #
    # Let's go back to the directive: "Sweep-Angle ... partitioning the problem".
    # If I can't use angles, I can use **Radial Sort** proxy.
    # Sort feasible nodes by distance to depot.
    # Visit nodes in order of increasing distance to depot? No, that's not sweep.
    #
    # Let's assume the distance matrix allows us to infer a 2D embedding via MDS if we have time?
    # No, 50 nodes, MDS is O(N^3) or similar, might be too slow per step if done repeatedly.
    #
    # Wait, the inspirations used Savings and scored 1.08.
    # The current program scored 0.99.
    # The current program is *better* than the inspirations.
    #
    # The directive says: "Make a STRUCTURAL change... A first try in a new direction may score worse; that is acceptable."
    #
    # I will implement a **Greedy Savings-like** score.
    # Instead of minimizing distance, we maximize "savings" relative to serving nodes individually.
    #
    # Score for candidate `c`:
    # We are extending the current route.
    # The "cost" of adding `c` to the current route ending at `current` is `d(current, c)`.
    # The "cost" of serving `c` individually from depot is `d(depot, c)`.
    # But we are not starting a new route, we are extending.
    #
    # Let's use a **Hybrid Angular-Proxy**:
    # We define the "direction" from depot to current node.
    # We want the next node to be "aligned" with this direction.
    # Using distances, we can check if `c` is "ahead" of `current` or "behind".
    #
    # Let `d_d_c` = distance(depot, current)
    # Let `d_d_n` = distance(depot, candidate)
    # Let `d_c_n` = distance(current, candidate)
    #
    # If `c` is further from depot than `current` and `d_c_n` is small, it's a good sweep extension.
    # If `c` is closer to depot, we might be backtracking.
    #
    # Score = d(current, candidate) + penalty_if_backtracking
    # Backtracking penalty: if d_d_n < d_d_c, add penalty.
    #
    # Let's try this: **Monotonic Distance to Depot Heuristic**.
    # Prefer nodes that are further from the depot than the current node, to avoid backtracking towards the depot before the route is complete.
    #
    # Score = d(current, cand) + gamma * max(0, d(depot, current) - d(depot, cand))
    #
    # This encourages moving away from the depot, mimicking a sweep that goes out and then back.
    
    if _time_exceeded():
        feasible_mask = np.array([demands[i] <= rest_capacity for i in unvisited_nodes])
        feasible_nodes = unvisited_nodes[feasible_mask]
        if len(feasible_nodes) == 0:
            return depot
        return int(feasible_nodes[np.argmin(distance_matrix[current_node, feasible_nodes])])

    feasible_mask = np.array([demands[i] <= rest_capacity for i in unvisited_nodes])
    feasible_nodes = unvisited_nodes[feasible_mask]

    if len(feasible_nodes) == 0:
        return depot

    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Distances
    d_cur_cand = distance_matrix[current_node, feasible_nodes]
    d_depot_cur = distance_matrix[depot, current_node]
    d_depot_cand = distance_matrix[depot, feasible_nodes]
    
    # Monotonicity penalty: Penalize moving closer to depot if we are not near the end
    # How do we know if we are near the end? 
    # If rest_capacity is small, or if few nodes are left.
    # 
    # Let's use a fixed penalty for now.
    # penalty = max(0, d_depot_cur - d_depot_cand)
    # 
    # If we are at the depot (start), d_depot_cur = 0. Penalty is 0.
    # As we move out, d_depot_cur increases.
    # If we pick a node closer to depot, penalty > 0.
    
    penalty = np.maximum(0, d_depot_cur - d_depot_cand)
    
    # We also want to avoid picking a node that is very far from current if the monotonicity gain is small.
    # The d_cur_cand term handles proximity.
    
    # Let's also consider the distance from candidate to depot for the return trip?
    # If we pick a node far from depot, the return trip is long.
    # But that's a global cost, not a local one.
    
    # Score = d_cur_cand + alpha * penalty
    alpha = 0.5
    scores = d_cur_cand + alpha * penalty
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
