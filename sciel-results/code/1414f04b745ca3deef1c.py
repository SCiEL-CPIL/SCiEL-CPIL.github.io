# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix (larger = more attractive) for ACO CVRP.
    
    Approach:
      1. Base heuristic: Inverse distance (standard).
      2. Clarke-Wright Savings: Boost customer-customer edges based on savings 
         (d_0i + d_0j - d_ij) to encourage merging routes.
      3. Depot-Pressure Field: 
         - Scale depot edges (0->j) by 1/sqrt(demand_j) to prioritize serving 
           high-demand nodes early when the truck is full (reducing risk of 
           getting stuck with unplaceable heavy loads later).
         - Scale customer-customer edges (i->j) by a "capacity potential" factor 
           that rewards visiting high-demand nodes when the truck is likely 
           still relatively empty (early in the route). This is approximated 
           statically by weighting edges towards high-demand nodes more heavily 
           if they are likely to be visited early. However, since we don't know 
           the order, we use a simpler heuristic: 
           - High demand nodes are "harder" to place. 
           - We boost edges *into* high-demand nodes slightly less if they are 
             very far, but more if they are close? 
           - Actually, the directive suggests: "scale edge weights by the product 
             of the departure node's remaining capacity potential and the arrival 
             node's demand efficiency". 
           - Since we can't know remaining capacity statically, we approximate:
             - "Departure node's remaining capacity potential": If we assume a 
               typical route length, early nodes have more remaining capacity. 
               We can proxy "early" by distance from depot? Nodes closer to depot 
               are likely visited earlier. So, nodes close to depot have high 
               "remaining capacity potential".
             - "Arrival node's demand efficiency": High demand is efficient if 
               it fills the truck, but risky if it overfills. 
           - Let's simplify the structural change: 
             - Depot edges: 1/d * (1 + demand_j / capacity) -> Wait, high demand 
               should be visited EARLY. So edges FROM depot TO high demand should 
               be MORE attractive? Yes, to get them out of the way.
             - Customer-Customer edges: 
                 - If we visit i then j, we need cap >= d_i + d_j.
                 - Edges leading to high demand nodes are risky if the truck is 
                   already full. 
                 - Let's boost edges *from* low-demand nodes *to* high-demand 
                   nodes? No, that's complex.
    
    Let's stick to a robust combination:
    1. Inverse Distance.
    2. Savings boost for CC edges.
    3. Demand-aware scaling:
       - Depot -> j: Boost if demand_j is high (visit early).
         Factor: 1.0 + 0.5 * (demand_j / capacity)
       - i -> j (CC): 
         - Savings boost.
         - Demand efficiency: If d_i + d_j is close to capacity, it's a good 
           pair. If it's very small, it's less efficient but safer. 
           Let's boost pairs that fill the truck well: 
           efficiency = 1.0 + 0.5 * ((d_i + d_j) / capacity)
    
    This combines the "Savings" structure from Inspiration 1/2 with a 
    "Depot-Pressure" refinement for the start of the route.
    """
    start_time = time.time()
    deadline = start_time + 280.0

    n = distance_matrix.shape[0]
    if distance_matrix.shape[1] != n:
        return np.full((n, n), 1e-9)

    demands = np.asarray(demands, dtype=np.float64)
    if demands.shape[0] != n:
        demands = demands[:n]

    # Safe distance matrix to avoid division by zero
    dist_safe = np.maximum(distance_matrix, 1e-9)
    
    # Initialize heuristic with inverse distance (standard)
    heuristic = 1.0 / dist_safe
    
    if n > 1:
        # Extract depot row/col
        d0 = distance_matrix[0, :]
        
        # Compute savings for customer-customer edges
        # Savings S_ij = d_0i + d_0j - d_ij
        # We only apply this to i, j != 0
        
        # Compute savings matrix for all pairs
        savings = d0[:, None] + d0[None, :] - distance_matrix
        
        # Only consider positive savings
        savings = np.maximum(savings, 0.0)
        
        # Mask out depot edges (row 0 and col 0) because savings logic doesn't apply directly
        depot_mask = np.zeros((n, n), dtype=bool)
        depot_mask[0, :] = True
        depot_mask[:, 0] = True
        
        # --- Depot Edge Refinement: "Depot-Pressure" ---
        # High demand nodes should be visited early.
        # Boost edges from depot to high-demand customers.
        # demand_j is the arrival node's demand.
        # Scale: 1 + alpha * (demand_j / capacity)
        # This makes high-demand nodes more attractive to leave the depot.
        alpha_dep = 0.5
        dep_boost = 1.0 + alpha_dep * (demands[None, :] / capacity)
        # Apply to row 0 (depot -> j), excluding depot itself (0,0)
        heuristic[0, 1:] *= dep_boost[0, 1:]
        
        # --- Customer-Customer Edge Refinement: Savings + Demand Efficiency ---
        # Combine savings with demand efficiency.
        # Demand efficiency: Higher if demands[i] + demands[j] is a larger fraction of capacity.
        # This encourages grouping customers that fill the truck well.
        demand_sum = demands[:, None] + demands[None, :]
        
        # Weight savings by demand efficiency
        # efficiency = 1 + beta * (demand_sum / capacity)
        beta_cc = 0.5
        efficiency = 1.0 + beta_cc * (demand_sum / capacity)
        
        # Normalize savings to be comparable
        max_savings = np.max(savings[~depot_mask]) if np.any(~depot_mask) else 1.0
        if max_savings < 1e-9:
            max_savings = 1.0
            
        norm_savings = savings / max_savings
        
        # Combine: heuristic_cc = (1/d_ij) * efficiency * (1 + norm_savings)
        cc_boost = efficiency * (1.0 + norm_savings)
        
        # Apply to non-depot edges
        heuristic[~depot_mask] = (1.0 / dist_safe[~depot_mask]) * cc_boost[~depot_mask]

    # Ensure all entries are strictly positive
    heuristic = np.maximum(heuristic, 1e-9)

    # Final check: no NaN or Inf
    if not np.all(np.isfinite(heuristic)):
        heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
        heuristic = np.maximum(heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
