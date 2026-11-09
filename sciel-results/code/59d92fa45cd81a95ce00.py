# EVOLVE-BLOCK-START
import numpy as np
import time
import networkx as nx
from itertools import combinations

def _solve_flow(demands, capacities, assign_cost, open_cost, allowed_subset):
    """
    Solve the capacitated facility location problem for a fixed set of allowed facilities.
    Returns (assignment_array, total_cost) or (None, inf) if infeasible.
    """
    n_cust = len(demands)
    n_fac = len(allowed_subset)
    
    if n_fac == 0:
        return None, float('inf')
        
    # Check total capacity feasibility
    total_cap = sum(capacities[f] for f in allowed_subset)
    total_demand = sum(demands)
    if total_cap < total_demand:
        return None, float('inf')

    # Build flow network
    # Source -> Facility (capacity=cap, cost=0)
    # Facility -> Customer (capacity=demand, cost=assign_cost)
    # Customer -> Sink (capacity=demand, cost=0)
    # Note: We need to handle the "open cost" separately or incorporate it.
    # Since the subset is fixed, the open cost is constant for this subset.
    # We just need to minimize assignment cost.
    
    G = nx.DiGraph()
    source = -1
    sink = n_cust + n_fac
    
    G.add_node(source)
    G.add_node(sink)
    
    # Map facility indices in allowed_subset to 0..n_fac-1 for node IDs
    fac_ids = {}
    for i, f in enumerate(allowed_subset):
        fac_ids[f] = i
        
    # Add facility nodes
    for f in allowed_subset:
        G.add_node(fac_ids[f])
        # Source to Facility
        G.add_edge(source, fac_ids[f], capacity=capacities[f], cost=0)
        
    # Add customer nodes
    for c in range(n_cust):
        G.add_node(c)
        # Customer to Sink
        G.add_edge(c, sink, capacity=demands[c], cost=0)
        
    # Facility to Customer edges
    for c in range(n_cust):
        for f in allowed_subset:
            cost = assign_cost[c, f]
            G.add_edge(fac_ids[f], c, capacity=demands[c], cost=cost)
            
    try:
        flow_value, flow_dict = nx.min_cost_flow(G)
    except Exception:
        return None, float('inf')
        
    # Check if max flow equals total demand
    if flow_value < total_demand:
        return None, float('inf')
        
    # Extract assignment
    assignment = np.zeros(n_cust, dtype=int)
    total_assign_cost = 0.0
    
    for c in range(n_cust):
        assigned_f = None
        for f in allowed_subset:
            fid = fac_ids[f]
            flow_amt = flow_dict[fid][c]
            if flow_amt > 0:
                # Since customer c sends exactly demands[c] to sink, and receives from exactly one facility (due to integer flow and unit capacities on customer side? No, capacity is demand[c])
                # Actually, a customer can be split? No, min_cost_flow with integer costs/capacities yields integer flows.
                # But can a customer be split across facilities?
                # The edge c->sink has capacity demand[c]. The edges f->c have capacity demand[c].
                # If the flow is fractional, we have a problem. But min_cost_flow returns integer flows if capacities and costs are integers.
                # Here capacities are integers. Costs are floats.
                # If costs are floats, the flow might be fractional?
                # NetworkX min_cost_flow uses successive shortest path. It works with floats.
                # However, the problem requires each customer to be assigned to ONE facility.
                # If the flow splits a customer, we need to repair it.
                # But usually, with these structures, the optimal solution is integral if the graph is bipartite and capacities are integers?
                # Actually, the "customer" node has incoming edges from facilities and outgoing to sink.
                # If we send 10 units to customer 1 from fac A and 5 from fac B, that's a split.
                # We need to ensure we assign to a single facility.
                # We can post-process: for each customer, if split, assign all to the one with highest flow? Or cheapest?
                # Better: modify the graph so that customer cannot be split.
                # This is hard with standard flow.
                # Alternative: Since n_cust=100, we can check if any customer is split.
                # If split, we can try to merge flows.
                
                # Let's assume for now that the flow is not split, or we handle it.
                # Actually, if we set capacity of f->c to demand[c], it allows full assignment to one.
                # If the optimal solution splits, it's because of cost averaging?
                # With linear costs, the extreme points of the polytope are integral if the matrix is totally unimodular.
                # The incidence matrix of a bipartite graph is TU.
                # So the flow should be integral if capacities are integers.
                # So each customer receives exactly demand[c] from exactly one facility?
                # Not necessarily. It could receive 5 from A and 5 from B if demand is 10.
                # But since the cost is linear, there is an optimal solution where no customer is split.
                # We can verify this. If a customer is split, we can shift the entire demand to the cheaper facility if capacity allows?
                # No, capacity might be tight.
                # However, the standard result for bipartite matching/flow with integral capacities is that there exists an integral optimal flow.
                # An integral flow means flow on each edge is integer.
                # It does NOT mean that a customer is served by a single facility.
                # Example: Customer demand 10. Fac A cap 10, Fac B cap 10. Cost A=1, Cost B=1.
                # Flow could be 5 from A, 5 from B.
                # This is a valid integral flow.
                # But the problem requires "one facility index per customer".
                # So we need to repair the solution.
                
                assigned_f = f
                # We need to be careful. Let's collect all flows for this customer.
                break
        
        # Proper extraction:
        flows = []
        for f in allowed_subset:
            fid = fac_ids[f]
            amt = flow_dict[fid][c]
            if amt > 1e-6:
                flows.append((f, amt, assign_cost[c, f]))
                
        if not flows:
            return None, float('inf')
            
        if len(flows) == 1:
            f_assigned = flows[0][0]
            assignment[c] = f_assigned
            total_assign_cost += flows[0][1] * assign_cost[c, f_assigned]
        else:
            # Split flow detected. We need to assign to a single facility.
            # Strategy: Try to assign to the facility with the highest flow? Or cheapest?
            # Since the flow is optimal for the relaxed problem, the split is due to symmetry or tight constraints.
            # We can try to "unsplit" by moving the entire demand to one of the facilities.
            # We must check if the target facility has enough capacity.
            # This is a repair step.
            
            # Sort by cost
            flows.sort(key=lambda x: x[2])
            
            # Try to assign to the cheapest one
            # We need to know the current load of each facility from the flow.
            # Let's calculate loads from flow_dict.
            # This is getting complicated. 
            # Simpler approach: If split, just pick the one with largest flow. 
            # This might violate capacity? No, the flow respects capacity.
            # But if we consolidate, we might exceed capacity?
            # If we consolidate demand c to facility f1, we remove flow from other facilities.
            # So load on f1 increases by (total_demand_c - flow_c_f1).
            # Load on others decreases.
            # We need to check if f1 has enough remaining capacity.
            
            # Let's compute the load on each facility from the flow.
            load = {f: 0 for f in allowed_subset}
            for f in allowed_subset:
                fid = fac_ids[f]
                # Sum flow from source to f
                load[f] = flow_dict[source][fid]
                
            # Try to consolidate to the cheapest facility in the split
            for f_target, amt, cost in flows:
                # We want to move all demand[c] to f_target
                # Additional load needed = demand[c] - amt
                extra = demands[c] - amt
                if load[f_target] + extra <= capacities[f_target] + 1e-6:
                    # Check if we can reduce load on other facilities
                    # We don't explicitly need to check others because we are just re-assigning c.
                    # The capacity constraint is on the total load.
                    # If load[f_target] + demand[c] <= capacities[f_target], then it's feasible.
                    # Wait, load[f_target] includes amt. So new load = load[f_target] + (demand[c] - amt).
                    # This is exactly what we checked.
                    assignment[c] = f_target
                    total_assign_cost += demand[c] * cost
                    break
            else:
                # Could not consolidate to cheapest. Try next?
                # Or just pick the one with max flow as a fallback.
                f_max = max(flows, key=lambda x: x[1])[0]
                assignment[c] = f_max
                total_assign_cost += demand[c] * assign_cost[c, f_max]
                
    # Calculate open cost
    open_cost_total = 0.0
    used_facilities = set(assignment.tolist())
    for f in used_facilities:
        if f in allowed_subset:
            open_cost_total += open_cost[f]
            
    total_cost = total_assign_cost + open_cost_total
    return assignment, total_cost


def _local_search(assignment, demands, capacities, assign_cost, open_cost, used_facilities):
    """
    Local search to improve the assignment.
    Tries single moves and swaps.
    """
    n_cust = len(demands)
    # Precompute loads
    load = np.zeros(len(capacities))
    for c in range(n_cust):
        f = assignment[c]
        load[f] += demands[c]
        
    improved = True
    iters = 0
    max_iters = 1000
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Single moves
        for c in range(n_cust):
            current_f = assignment[c]
            current_cost = assign_cost[c, current_f]
            
            # Find best alternative
            best_delta = 0
            best_f = -1
            
            # Only consider facilities that are already open or have low open cost?
            # For simplicity, consider all facilities. If we open a new one, we pay open_cost.
            # But opening a new facility is a "big" move.
            # Let's stick to moves to existing open facilities first.
            # Then maybe try opening new ones.
            
            for f in used_facilities:
                if f == current_f:
                    continue
                # Check capacity
                if load[f] + demands[c] <= capacities[f]:
                    delta = assign_cost[c, f] - current_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_f = f
                        
            if best_f != -1:
                # Make move
                load[current_f] -= demands[c]
                load[best_f] += demands[c]
                assignment[c] = best_f
                improved = True
                
        # Swaps
        for c1 in range(n_cust):
            f1 = assignment[c1]
            for c2 in range(c1 + 1, n_cust):
                f2 = assignment[c2]
                if f1 == f2:
                    continue
                
                # Check capacity after swap
                # load[f1] - demands[c1] + demands[c2] <= cap[f1]
                # load[f2] - demands[c2] + demands[c1] <= cap[f2]
                
                if load[f1] - demands[c1] + demands[c2] <= capacities[f1] and \
                   load[f2] - demands[c2] + demands[c1] <= capacities[f2]:
                   
                    cost_before = assign_cost[c1, f1] + assign_cost[c2, f2]
                    cost_after = assign_cost[c1, f2] + assign_cost[c2, f1]
                    
                    if cost_after < cost_before - 1e-9:
                        # Make swap
                        load[f1] = load[f1] - demands[c1] + demands[c2]
                        load[f2] = load[f2] - demands[c2] + demands[c1]
                        assignment[c1] = f2
                        assignment[c2] = f1
                        improved = True
                        break # Restart inner loop? Or just continue?
                        # Breaking inner loop is safer to avoid invalidating indices?
                        # Actually, we update loads, so it's fine.
                        # But we should probably restart the whole iteration to be safe?
                        # No, just continue.
                        
        # Update used_facilities
        used_facilities = set(assignment.tolist())
        
    return assignment


def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign 100 customers to 20 facilities.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Precompute average assignment cost for each facility
    avg_assign_cost = assign_cost.mean(axis=0)
    
    # Generate candidate subsets
    # We want to balance open cost and assignment cost.
    # Score = open_cost[f] + avg_assign_cost[f]
    scores = open_cost + avg_assign_cost
    sorted_facs = np.argsort(scores)
    
    candidate_subsets = []
    
    # 1. All facilities
    candidate_subsets.append(tuple(range(n_fac)))
    
    # 2. Top k by score
    for k in [5, 8, 10, 12, 15, 18]:
        if k <= n_fac:
            subset = tuple(sorted_facs[:k].tolist())
            candidate_subsets.append(subset)
            
    # 3. Top k by open cost
    by_open = np.argsort(open_cost)
    for k in [5, 8, 10, 12, 15]:
        if k <= n_fac:
            subset = tuple(by_open[:k].tolist())
            candidate_subsets.append(subset)
            
    # 4. Top k by avg assign cost
    by_assign = np.argsort(avg_assign_cost)
    for k in [5, 8, 10, 12, 15]:
        if k <= n_fac:
            subset = tuple(by_assign[:k].tolist())
            candidate_subsets.append(subset)
            
    # Remove duplicates
    unique_subsets = list(set(candidate_subsets))
    
    best_result = None
    best_total = float('inf')
    
    for subset in unique_subsets:
        if time.time() > deadline - 5:
            break
            
        result, total_cost = _solve_flow(demands, capacities, assign_cost, open_cost, subset)
        
        if result is not None:
            if total_cost < best_total:
                best_total = total_cost
                best_result = result.copy()
                
    if best_result is None:
        # Fallback: Greedy
        best_result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    best_result[i] = f
                    break
            else:
                # Should not happen if total capacity is sufficient
                pass
                
        # Calculate cost for fallback
        used = set(best_result.tolist())
        best_total = sum(open_cost[f] for f in used) + sum(assign_cost[i, best_result[i]] for i in range(n_cust))
        
    # Local search
    if time.time() < deadline - 2:
        used_facilities = set(best_result.tolist())
        best_result = _local_search(best_result, demands, capacities, assign_cost, open_cost, used_facilities)
        
        # Recalculate cost
        used_facilities = set(best_result.tolist())
        final_cost = sum(open_cost[f] for f in used_facilities) + sum(assign_cost[i, best_result[i]] for i in range(n_cust))
        
    return best_result
# EVOLVE-BLOCK-END
