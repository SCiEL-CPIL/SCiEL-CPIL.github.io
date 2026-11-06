"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline tracking and caching
_start_time = None
_deadline = 145.0  # 5s buffer before 150s limit

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    # Initialize deadline tracking on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: stop if we've exceeded deadline
    elapsed = time.time() - _start_time
    if elapsed > _deadline:
        return depot
    
    if len(unvisited_nodes) == 0:
        return depot
    
    # Check feasible nodes (demand <= rest_capacity)
    unvisited_arr = np.array(unvisited_nodes)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot
    
    # Radial Sweep Strategy:
    # 1. Calculate the angle of the current node relative to the depot.
    # 2. Calculate the angles of all feasible nodes.
    # 3. Select the node that is closest to the current node in terms of 
    #    angular distance, but also considering the distance from the current node.
    # This enforces a "sweep" pattern around the depot.
    
    # If current_node is the depot (0), we can start with any node.
    # To maintain consistency, if current is depot, we pick the node with the smallest angle
    # (or simply the closest one to start the sweep).
    
    # We need coordinates to compute angles. The distance_matrix is Euclidean usually.
    # However, we don't have coordinates directly. We only have distance_matrix.
    # We can infer a relative position or just use the distance matrix to simulate the sweep?
    # Actually, without coordinates, "angular sweep" is hard.
    # Alternative: Use the distance to the depot as a proxy for "radius" and 
    # distance to current as a proxy for "progress".
    
    # Let's try a different structural change: "Farthest Insertion" variant or "Nearest Neighbor with Capacity Lookahead".
    # The prompt suggests "capacity-aware angular sweep".
    # Since we don't have coordinates, we can approximate the "angle" by looking at the 
    # relative distances to the depot.
    # Nodes that are "far" from the depot and "close" to the current node are good candidates.
    
    # Let's stick to a robust heuristic that works with just distance matrices:
    # "Regret-based Nearest Neighbor"
    # For each feasible node, calculate the cost of visiting it now vs. the cost if we visited it later (approximated by its distance to the 2nd best node).
    # Actually, simple regret: 
    # score(node) = d(current, node) + d(node, depot) - d(current, depot) is the savings.
    # The current program uses max savings.
    
    # Let's try "Sweep" using the distance matrix properties.
    # If we assume the depot is at (0,0), we can't get angles.
    # But we can use a "Directional" heuristic:
    # Prioritize nodes that are in the same "general direction" from the depot as the current node.
    # How to measure direction without coords?
    # Triangle inequality: d(depot, node) is roughly consistent if they are in the same direction.
    # If node A and B are in the same direction from depot, d(A, B) is small compared to d(depot, A) + d(depot, B).
    # If they are in opposite directions, d(A, B) is large.
    
    # So, a "same direction" score for node c given current node cur is:
    # dir_score(c) = d(depot, cur) + d(depot, c) - d(cur, c)
    # This is exactly the Clarke-Wright savings!
    # The current program already maximizes this.
    
    # Why did it score 1.0820? Maybe it's good but not optimal.
    # The directive asks for a STRUCTURAL change.
    # "Capacity-aware angular sweep" implies we should prefer nodes that keep us in a "cluster" or "sector".
    
    # Let's try a "Cluster-based" selection.
    # Identify the "center" of the current route (or the current node) and pick the feasible node
    # that minimizes the distance to the current node, BUT penalize nodes that are too far from the depot 
    # if they would cause us to return to the depot with little capacity left? No, that's complex.
    
    # Let's try "Nearest Neighbor with Capacity Constraint" but with a twist:
    # Instead of just picking the nearest feasible node, pick the node that minimizes 
    # d(current, node) + alpha * d(node, depot)
    # where alpha encourages going back to the depot sooner if the node is far out.
    # But we want to maximize route length efficiency (minimize total length).
    # Going to a far node and then back to depot is expensive.
    # So, d(current, node) + d(node, depot) is the marginal cost of adding this node to the route 
    # if we were to close the route after it.
    # However, we might visit more nodes.
    
    # Let's try "Minimizing the angle" using a geometric approximation if possible, 
    # or falling back to a robust metric.
    # Since we don't have coordinates, we can't do true angular sweep.
    # We must use the distance matrix.
    
    # New Strategy: "Lookahead Nearest Neighbor"
    # For each feasible node c:
    # 1. Calculate immediate cost: d(current, c)
    # 2. Calculate "regret": How much worse is it to visit c later?
    #    Approximate by: d(current, c) - min_{c' in feasible, c' != c} d(current, c')
    #    This is the standard regret-1 heuristic.
    #    Nodes with high regret (i.e., if we don't visit them now, we'll have to travel much further to get to them later) should be visited now.
    
    # Let's implement Regret-1.
    # It is structurally different from pure Savings or pure Nearest Neighbor.
    
    d_current_cand = distance_matrix[current_node, feasible_nodes]
    
    # If only one feasible node, pick it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Calculate regret for each node
    # Regret(i) = cost(i) - cost(second_best)
    # We want to maximize regret.
    
    # Sort distances
    sorted_dists = np.sort(d_current_cand)
    min_dist = sorted_dists[0]
    second_min_dist = sorted_dists[1]
    
    # For each node, regret = d(current, node) - min_dist
    # Wait, standard regret is: 
    # Regret(k) = cost(k, first) - cost(k, second)
    # Here, the "cost" of visiting a node is the distance from current.
    # So for node i:
    # cost_1 = d(current, i)
    # cost_2 = min_{j != i} d(current, j)
    # Regret(i) = cost_1 - cost_2
    
    # If i is the nearest node, cost_1 = min_dist, cost_2 = second_min_dist. Regret = min_dist - second_min_dist (negative).
    # If i is the farthest node, cost_1 is large, cost_2 is min_dist. Regret is large.
    # This encourages visiting nodes that are "unique" or "hard to reach later" (i.e., far from the current best option).
    # This prevents getting stuck in a local cluster while ignoring distant nodes that might be better served now.
    
    # However, we also need to respect capacity and eventually return to depot.
    # A pure regret heuristic might ignore the return to depot.
    # Let's combine Regret with a "Depot Return" penalty.
    
    # Score = Regret(i) - beta * d(i, depot)
    # This balances "urgency" (regret) with "efficiency" (distance to depot).
    
    # Calculate cost_2 for each node
    # cost_2[i] is the minimum distance from current to any node OTHER than i.
    # This is O(N^2) to do naively, but N is small (50).
    
    regrets = np.zeros(len(feasible_nodes))
    for i in range(len(feasible_nodes)):
        # Mask out the current node i
        dists_copy = d_current_cand.copy()
        dists_copy[i] = np.inf
        min_other = np.min(dists_copy)
        regrets[i] = d_current_cand[i] - min_other
        
    # Now, incorporate the distance to the depot.
    # If we pick a node, we will eventually have to go back to the depot.
    # The "savings" heuristic already accounts for this partially.
    # Let's try a weighted sum:
    # Score = w1 * regrets - w2 * d(feasible_node, depot)
    
    d_depot_cand = distance_matrix[depot, feasible_nodes]
    
    # Normalize regrets and distances to make them comparable?
    # Or just use a simple weight.
    # Let's try to maximize: regrets - 0.5 * d_depot_cand
    
    scores = regrets - 0.5 * d_depot_cand
    
    idx = np.argmax(scores)
    best_node = feasible_nodes[idx]
    
    return int(best_node)

# EVOLVE-BLOCK-END
