"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from datetime import datetime, timedelta

# Global deadline guard
_EVAL_START = datetime.now()
DEADLINE = _EVAL_START + timedelta(seconds=145)  # Safety margin before 150s kill


def _time_remaining():
    return (DEADLINE - datetime.now()).total_seconds()


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for selecting the next node in a capacitated vehicle routing problem.
    Uses a savings-based lookahead approach with capacity constraints.
    """
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    # Check if we're running out of time
    if _time_remaining() < 1.0:
        # Fallback to simple nearest feasible
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return int(depot)
        # Safe distance calculation for list-of-lists or array
        dists = [distance_matrix[current_node][n] for n in feasible]
        return int(feasible[np.argmin(dists)])
    
    # Find feasible nodes (can fit in remaining capacity)
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if len(feasible) == 0:
        return int(depot)
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    # Use a savings-based heuristic with lookahead
    # The idea: for each feasible node i, compute a "savings" score that considers:
    # 1. Distance from current to i
    # 2. Distance from i back to depot (if we close after i)
    # 3. Potential savings from not going back to depot
    
    # Simple approach: use nearest neighbor with a twist
    # For each candidate, estimate the cost of visiting it vs. returning to depot
    # We want to minimize total route length, so we prefer nodes that are close
    # and also have good "continuation" potential
    
    # Compute distances from current to all feasible nodes safely
    # Assuming distance_matrix is a list of lists or 2D array accessible by index
    dist_to_feasible = [distance_matrix[current_node][n] for n in feasible]
    
    if len(feasible) <= 20:
        # For small sets, do proper lookahead
        scores = []
        for idx, i in enumerate(feasible):
            dist_ci = dist_to_feasible[idx]
            # Find the nearest other feasible node from i
            if len(feasible) > 1:
                # Get distances from i to all other feasible nodes
                other_nodes = [feasible[j] for j in range(len(feasible)) if j != idx]
                dists_from_i = [distance_matrix[i][n] for n in other_nodes]
                min_next = min(dists_from_i)
            else:
                min_next = 0.0
            # Combine: prefer close to current, and close to next
            scores.append(dist_ci + 0.3 * min_next)
        best_idx = np.argmin(scores)
        return int(feasible[best_idx])
    else:
        # For larger sets, use a faster approximation
        # Just use nearest neighbor with a slight penalty for nodes far from depot
        dists = dist_to_feasible
        # Add a small bias toward nodes closer to depot (to keep routes compact)
        depot_dists = [distance_matrix[depot][n] for n in feasible]
        # Convert to numpy for vectorized addition and argmin
        scores = np.array(dists) + 0.1 * np.array(depot_dists)
        best_idx = np.argmin(scores)
        return int(feasible[best_idx])
# EVOLVE-BLOCK-END
