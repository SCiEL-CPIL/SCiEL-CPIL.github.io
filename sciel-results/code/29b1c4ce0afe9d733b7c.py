"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute heuristic matrix for ACO orienteering.
    
    The heuristic guides ants toward nodes with high prize and low effective cost.
    Effective cost of going from i to j and returning to 0 is d[i,j] + d[j,0].
    Only edges where this cost fits within max_len are truly feasible.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # Effective cost: d[i,j] + d[j,0] - the cost of going to j and coming back
    # This is what matters for feasibility since ant must return to 0
    # cost[i,j] = d[i,j] + d[j,0]
    cost = distance_matrix + distance_matrix[:, 0:1]  # d[i,j] + d[j,0]
    
    # Feasibility: can we go from i to j and still return to 0 within max_len?
    # The ant has already traveled some path to i, but we don't know that in the heuristic.
    # We use a conservative estimate: assume the ant is at i, needs d[i,j] to go to j,
    # and d[j,0] to return. So the remaining budget must cover d[i,j] + d[j,0].
    # But the ant has already used some budget to get to i. We don't track that in the heuristic.
    # So we use a softer constraint: edges with cost <= max_len get high heuristic,
    # edges with cost > max_len get very low heuristic.
    
    # Base heuristic: prize at destination / effective cost
    # Add small epsilon to avoid division by zero
    eps = 1e-9
    
    # prize[j] is the prize at node j (destination)
    # We want high prize, low cost
    # heuristic[i,j] = prize[j] / (cost[i,j] + eps)
    
    # But we need to handle infeasible edges
    # Let's create a feasibility mask
    feasible = cost <= max_len
    
    # For feasible edges, use prize/cost ratio
    # For infeasible edges, use a very small value
    
    # Compute base ratio
    base = prizes[None, :] / (cost + eps)
    
    # Apply feasibility: infeasible edges get minimal value
    # Use a smooth transition or hard cutoff
    # Hard cutoff: infeasible -> 1e-9
    result = np.where(feasible, base, 1e-9)
    
    # Ensure node 0 has low heuristic as a destination (prize is 0, so base[*,0] should be ~0)
    # But let's explicitly set it to be safe
    result[:, 0] = 1e-9
    
    # Ensure all values are positive and finite
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)
    
    # Optionally, normalize or scale to make differences more pronounced
    # Use a power to amplify high-prize nodes
    # Let's try: result = result ** 2 to amplify differences
    # But be careful not to make values too extreme
    
    # Actually, let's think about what makes a good heuristic:
    # 1. It should favor high prize nodes
    # 2. It should favor close nodes (low cost)
    # 3. It should strongly penalize infeasible edges
    
    # The current formula does this. Let's also consider:
    # - Nodes with prize 0 should have very low heuristic
    # - Nodes with high prize and low cost should have very high heuristic
    
    # Let's add a bonus for high prize nodes to make the ACO more directed
    # Scale by prize to make high-prize nodes more attractive
    result = result * (prizes[None, :] + eps)
    
    # Re-apply feasibility and positivity
    result = np.where(feasible, result, 1e-9)
    result[:, 0] = 1e-9
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
