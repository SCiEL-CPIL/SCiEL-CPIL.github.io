"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import sys

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Minimum-cost flow formulation:
    # Source -> Customer nodes (supply=1 each)
    # Customer nodes -> Facility nodes (cost=assign_cost, capacity=1)
    # Facility nodes -> Sink (capacity=capacities[j], cost=open_cost/capacities[j] as surrogate)
    #
    # We'll use LP relaxation of the flow problem. Since the constraint matrix
    # is totally unimodular for network flow, the LP relaxation gives integer solutions
    # for the flow variables. The fixed-charge is approximated by spreading open_cost
    # across capacity units.
    
    # Build the LP for min-cost flow:
    # Variables: flow[i, j] for customer i to facility j, and flow_out[j] for facility j to sink
    # Actually, let's just use the assignment variables directly with proper constraints.
    
    # Nodes: 0=source, 1..n_customers=customers, n_customers+1..n_customers+n_facilities=facilities, 
    #         n_customers+n_facilities+1=sink
    # Edges:
    # source -> customer i: capacity 1, cost 0
    # customer i -> facility j: capacity 1, cost assign_cost[i,j]
    # facility j -> sink: capacity capacities[j], cost open_cost[j] / capacities[j] (surrogate)
    
    # Total flow = n_customers
    # Let's formulate as LP with variables for each edge flow
    
    # Edge indexing:
    # Edges 0..n_customers-1: source -> customer i
    # Edges n_customers..n_customers+n_customers*n_facilities-1: customer i -> facility j
    # Edges n_customers+n_customers*n_facilities..n_customers+n_customers*n_facilities+n_facilities-1: facility j -> sink
    
    n_edges_src = n_customers
    n_edges_cust_fac = n_customers * n_facilities
    n_edges_fac_sink = n_facilities
    n_total_edges = n_edges_src + n_edges_cust_fac + n_edges_fac_sink
    
    # Objective: minimize sum of edge costs * flows
    c = np.zeros(n_total_edges)
    # Source->customer edges: cost 0
    # Customer->facility edges: cost = assign_cost
    c[n_edges_src:n_edges_src + n_edges_cust_fac] = assign_cost.flatten()
    # Facility->sink edges: cost = open_cost[j] / capacities[j] (surrogate for fixed charge)
    # We spread the open cost across the capacity
    for j in range(n_facilities):
        c[n_edges_src + n_edges_cust_fac + j] = open_cost[j] / max(capacities[j], 1)
    
    # Flow conservation constraints:
    # For source: outflow = n_customers (all customers must be served)
    # For customer i: inflow from source = outflow to facilities = 1
    # For facility j: inflow from customers = outflow to sink
    # For sink: inflow from facilities = n_customers
    
    # Let's build the constraint matrix.
    # Nodes: 0=source, 1..n_customers=customers, n_customers+1..n_customers+n_facilities=facilities, 
    #         n_customers+n_facilities+1=sink
    n_nodes = n_customers + n_facilities + 2
    node_source = 0
    node_customer_start = 1
    node_facility_start = 1 + n_customers
    node_sink = 1 + n_customers + n_facilities
    
    # A matrix: n_nodes x n_total_edges
    # For each edge, +1 at tail node, -1 at head node (or vice versa for conservation)
    # Standard: for node k, sum of inflows - sum of outflows = b[k]
    # where b[source] = -n_customers, b[customer i] = 0, b[facility j] = 0, b[sink] = n_customers
    
    A = np.zeros((n_nodes, n_total_edges))
    b = np.zeros(n_nodes)
    
    # Source -> customer i edges
    for i in range(n_customers):
        edge_idx = i
        A[node_source, edge_idx] = -1  # outflow from source
        A[node_customer_start + i, edge_idx] = 1  # inflow to customer
    
    # Customer i -> facility j edges
    for i in range(n_customers):
        for j in range(n_facilities):
            edge_idx = n_edges_src + i * n_facilities + j
            A[node_customer_start + i, edge_idx] = -1  # outflow from customer
            A[node_facility_start + j, edge_idx] = 1  # inflow to facility
    
    # Facility j -> sink edges
    for j in range(n_facilities):
        edge_idx = n_edges_src + n_edges_cust_fac + j
        A[node_facility_start + j, edge_idx] = -1  # outflow from facility
        A[node_sink, edge_idx] = 1  # inflow to sink
    
    # b vector:
    # source: net outflow = n_customers => sum(out) - sum(in) = n_customers => -sum(out) + sum(in) = -n_customers
    # So b[source] = -n_customers
    # customers: net flow = 0
    # facilities: net flow = 0
    # sink: net inflow = n_customers => sum(in) - sum(out) = n_customers => b[sink] = n_customers
    
    b[node_source] = -n_customers
    b[node_sink] = n_customers
    
    # Bounds:
    # Source->customer: [0, 1]
    # Customer->facility: [0, 1]
    # Facility->sink: [0, capacities[j]]
    
    bounds = []
    for i in range(n_edges_src):
        bounds.append((0, 1))
    for i in range(n_edges_cust_fac):
        bounds.append((0, 1))
    for j in range(n_edges_fac_sink):
        bounds.append((0, capacities[j]))
    
    # Solve LP
    try:
        res = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method='highs')
        
        if res.success:
            # Extract customer->facility flows
            x_vals = res.x[n_edges_src:n_edges_src + n_edges_cust_fac].reshape(n_customers, n_facilities)
            
            # The LP should give near-integer solutions due to total unimodularity
            # But the facility->sink edges have continuous capacity, so let's check
            
            # Round the assignment variables
            x_rounded = np.round(x_vals).astype(int)
            
            # Verify each customer is assigned to exactly one facility
            if not np.all(x_rounded.sum(axis=1) == 1):
                # Try a different rounding strategy
                # For each customer, assign to the facility with highest flow
                x_rounded = np.zeros((n_customers, n_facilities), dtype=int)
                for i in range(n_customers):
                    best_j = np.argmax(x_vals[i])
                    x_rounded[i, best_j] = 1
            
            # Check capacity constraints
            loads = x_rounded.T @ demands
            if np.any(loads > capacities + 1e-6):
                # Need to repair - use greedy repair
                result = np.zeros(n_customers, dtype=int)
                load = np.zeros(n_facilities)
                order = np.argsort(-demands)
                for i in order:
                    costs = assign_cost[i].copy()
                    for f in range(n_facilities):
                        if load[f] + demands[i] > capacities[f]:
                            costs[f] = np.inf
                    min_idx = np.argmin(costs)
                    if np.isinf(costs[min_idx]):
                        # Should not happen if total capacity >= total demand
                        min_idx = np.argmin(assign_cost[i])
                    result[i] = min_idx
                    load[min_idx] += demands[i]
                return result
            
            result = x_rounded.argmax(axis=1)
            return result
        else:
            raise Exception("LP failed")
            
    except Exception:
        pass
    
    # Fallback: greedy assignment sorted by demand (largest first)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    order = np.argsort(-demands)
    for i in order:
        costs = assign_cost[i].copy()
        for f in range(n_facilities):
            if load[f] + demands[i] > capacities[f]:
                costs[f] = np.inf
        min_idx = np.argmin(costs)
        if np.isinf(costs[min_idx]):
            # Should not happen, but handle gracefully
            min_idx = np.argmin(assign_cost[i])
        result[i] = min_idx
        load[min_idx] += demands[i]
        
    return result
# EVOLVE-BLOCK-END
