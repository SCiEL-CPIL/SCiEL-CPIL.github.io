# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility using a min-cost flow formulation.
    We model this as a transportation problem with facility opening costs.
    
    Approach:
    1. Use linear programming / min-cost flow via scipy or manual implementation.
    2. Since we have binary opening decisions, we'll use a heuristic that's close to optimal:
       - First, solve the assignment ignoring opening costs (min-cost matching with capacities)
       - Then, check which facilities are used, and if opening cost savings justify reassignment
    
    Actually, let's use a proper min-cost flow approach.
    
    The problem is:
    min sum_f open_cost[f] * y[f] + sum_{i,f} assign_cost[i,f] * x[i,f]
    s.t. sum_f x[i,f] = 1 for all i (each customer assigned to one facility)
         sum_i demands[i] * x[i,f] <= capacities[f] * y[f] for all f
         y[f] in {0,1}
         x[i,f] in {0,1}
    
    This is an integer program. For 100 customers and 20 facilities, we can use:
    - A greedy heuristic with local search
    - Or use scipy's linear programming with rounding
    
    Let's use a min-cost flow approach where we treat each facility as having
    capacity = capacities[f], and we need to decide which facilities to open.
    
    Since the opening cost is fixed per facility, we can:
    1. Try all subsets of facilities? No, 2^20 is too large.
    2. Use a heuristic: start with all facilities open, solve min-cost flow,
       then try to close facilities and re-solve.
    
    Let's implement a min-cost flow using successive shortest paths with potentials.
    
    Network:
    - Source -> each customer (capacity 1, cost 0)
    - Each customer -> each facility (capacity 1, cost = assign_cost[i,f])
    - Each facility -> sink (capacity = capacities[f], cost 0)
    - Additionally, facility opening: if facility f is used, pay open_cost[f]
    
    To handle opening costs in min-cost flow, we can add a "opening" node for each facility:
    - Source -> customer (cap 1, cost 0)
    - Customer -> facility (cap 1, cost assign_cost[i,f])
    - Facility -> opening_node_f (cap infinity, cost 0)
    - Opening_node_f -> sink (cap capacities[f], cost open_cost[f]/capacities[f]?) 
    
    No, that doesn't work cleanly because the opening cost is fixed, not per unit.
    
    Alternative: Use a min-cost flow where we have two layers:
    - Layer 1: Decide which facilities to open (binary)
    - Layer 2: Assign customers to open facilities
    
    For a cleaner approach, let's use a heuristic that's close to optimal:
    1. Solve the transportation problem (min-cost flow) with all facilities available
    2. Identify which facilities are used
    3. If the total cost (assignment + opening) can be improved by closing some facilities,
       re-solve with fewer facilities
    
    Let's implement a min-cost flow using the successive shortest path algorithm.
    
    Actually, for this size (100 customers, 20 facilities), we can use scipy's
    linear_sum_assignment for the assignment part, but that's for square matrices.
    
    Let's use a different approach: min-cost flow via network simplex or successive shortest paths.
    
    I'll implement a simple min-cost flow using scipy's linprog or a custom successive shortest path.
    
    Given the constraints, let's use a practical approach:
    1. Use min-cost flow to assign customers to facilities, treating facility capacity as flow capacity
    2. The cost for sending flow from customer i to facility f is assign_cost[i,f]
    3. The facility has a fixed cost open_cost[f] if any flow goes through it
    
    To handle the fixed cost, we can:
    - Start with all facilities "open" (pay all opening costs)
    - Solve min-cost flow
    - Then, for each facility that has low utilization, try closing it and re-solving
    
    Let's implement min-cost flow using successive shortest paths with Bellman-Ford for initial potentials.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Build the min-cost flow network
    # Nodes: 0 = source, 1..n_customers = customers, n_customers+1..n_customers+n_facilities = facilities, n_customers+n_facilities+1 = sink
    # Edges:
    # source -> customer i: cap=1, cost=0
    # customer i -> facility f: cap=1, cost=assign_cost[i,f]
    # facility f -> sink: cap=capacities[f], cost=0
    # 
    # But we also need to account for opening costs.
    # We'll handle opening costs separately: after solving, we pay open_cost for each facility used.
    # Then we try to improve by closing facilities.
    
    # For now, let's solve the min-cost flow without opening costs, then add a local search for opening costs.
    
    # Min-cost flow using successive shortest paths
    # This is complex to implement from scratch. Let's use scipy's linear programming instead.
    
    # Actually, let's use a simpler approach: 
    # 1. Use a greedy algorithm with lookahead
    # 2. Then do local search to improve
    
    # Given the time constraint, let's use a well-tuned greedy + local search.
    
    # Step 1: Greedy assignment (largest demand first, cheapest feasible facility)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    
    for i in order:
        # Find the cheapest feasible facility
        best_f = -1
        best_cost = np.inf
        for f in range(n_facilities):
            if load[f] + demands[i] <= capacities[f]:
                cost = assign_cost[i, f]
                if cost < best_cost:
                    best_cost = cost
                    best_f = f
        if best_f >= 0:
            load[best_f] += demands[i]
            result[i] = best_f
        else:
            # Should not happen if total capacity >= total demand
            result[i] = 0
    
    # Step 2: Local search - try to swap assignments to reduce cost
    # Also try to close facilities if it reduces total cost
    
    # Compute current total cost
    def total_cost(result, load):
        cost = 0.0
        used_facilities = set(result.tolist())
        for f in used_facilities:
            cost += open_cost[f]
        for i in range(n_customers):
            cost += assign_cost[i, result[i]]
        return cost
    
    current_cost = total_cost(result, load)
    
    # Local search: try swapping pairs of customers
    improved = True
    max_iterations = 50
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(n_customers):
            for j in range(i + 1, n_customers):
                fi = result[i]
                fj = result[j]
                if fi == fj:
                    continue
                # Try swapping: i -> fj, j -> fi
                if load[fi] - demands[i] + demands[j] <= capacities[fi] and \
                   load[fj] - demands[j] + demands[i] <= capacities[fj]:
                    new_cost = current_cost
                    # Remove old costs
                    new_cost -= assign_cost[i, fi] + assign_cost[j, fj]
                    # Add new costs
                    new_cost += assign_cost[i, fj] + assign_cost[j, fi]
                    # Check if facilities change
                    # Facility fi: was used if load[fi] > 0, now load[fi] - demands[i] + demands[j]
                    # Facility fj: was used if load[fj] > 0, now load[fj] - demands[j] + demands[i]
                    old_load_fi = load[fi]
                    old_load_fj = load[fj]
                    new_load_fi = old_load_fi - demands[i] + demands[j]
                    new_load_fj = old_load_fj - demands[j] + demands[i]
                    
                    # Opening cost changes
                    if old_load_fi > 0 and new_load_fi == 0:
                        new_cost -= open_cost[fi]
                    elif old_load_fi == 0 and new_load_fi > 0:
                        new_cost += open_cost[fi]
                    if old_load_fj > 0 and new_load_fj == 0:
                        new_cost -= open_cost[fj]
                    elif old_load_fj == 0 and new_load_fj > 0:
                        new_cost += open_cost[fj]
                    
                    if new_cost < current_cost - 1e-9:
                        # Make the swap
                        result[i] = fj
                        result[j] = fi
                        load[fi] = new_load_fi
                        load[fj] = new_load_fj
                        current_cost = new_cost
                        improved = True
    
    # Step 3: Try to close facilities and reassign
    for f in range(n_facilities):
        if load[f] == 0:
            continue
        # Try closing facility f
        customers_at_f = np.where(result == f)[0]
        if len(customers_at_f) == 0:
            continue
        
        # Reassign these customers to other facilities
        new_load = load.copy()
        new_load[f] = 0
        new_result = result.copy()
        
        feasible = True
        for i in customers_at_f:
            # Find best alternative facility
            best_alt = -1
            best_alt_cost = np.inf
            for alt_f in range(n_facilities):
                if alt_f == f:
                    continue
                if new_load[alt_f] + demands[i] <= capacities[alt_f]:
                    cost = assign_cost[i, alt_f]
                    if cost < best_alt_cost:
                        best_alt_cost = cost
                        best_alt = alt_f
            if best_alt < 0:
                feasible = False
                break
            new_load[best_alt] += demands[i]
            new_result[i] = best_alt
        
        if not feasible:
            continue
        
        # Compute new cost
        new_total = 0.0
        for alt_f in range(n_facilities):
            if new_load[alt_f] > 0:
                new_total += open_cost[alt_f]
        for i in range(n_customers):
            new_total += assign_cost[i, new_result[i]]
        
        if new_total < current_cost - 1e-9:
            result = new_result
            load = new_load
            current_cost = new_total
    
    return result
# EVOLVE-BLOCK-END
