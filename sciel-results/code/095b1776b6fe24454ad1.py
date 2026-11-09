# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from scipy.optimize import linear_sum_assignment


def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign 100 customers to 20 facilities.
    Uses min-cost flow for exact assignment given a facility subset,
    then local search for facility open/close decisions.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (150s limit, 5s buffer)
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    total_demand = demands.sum()
    
    def _total_cost(result):
        """Compute total cost: open cost + assignment cost"""
        used = set(result.tolist())
        oc = sum(open_cost[f] for f in used)
        ac = sum(assign_cost[i, result[i]] for i in range(n_cust))
        return oc + ac
    
    def _is_feasible(result):
        """Check if assignment is feasible (no capacity violations)"""
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[result[i]] += demands[i]
        return np.all(load <= capacities + 1e-9)
    
    def _min_cost_flow_for_subset(allowed_facilities):
        """
        Solve min-cost flow to assign all customers to the given facility subset.
        Uses successive shortest path with potentials (Dijkstra).
        Returns (assignment array, total_assign_cost) or (None, inf) if infeasible.
        """
        if len(allowed_facilities) == 0:
            return None, float('inf')
        
        # Check total capacity
        total_cap = sum(capacities[f] for f in allowed_facilities)
        if total_cap < total_demand - 1e-9:
            return None, float('inf')
        
        n_allowed = len(allowed_facilities)
        
        # Build flow network:
        # Source -> customer nodes (capacity = demand[i], cost = 0)
        # Customer nodes -> facility nodes (capacity = inf, cost = assign_cost[i,f])
        # Facility nodes -> sink (capacity = capacities[f], cost = 0)
        
        # Node indices:
        # 0: source
        # 1..n_cust: customers
        # n_cust+1..n_cust+n_allowed: facilities
        # n_cust+n_allowed+1: sink
        n_nodes = 1 + n_cust + n_allowed + 1
        source = 0
        sink = n_nodes - 1
        
        # Adjacency list for residual graph
        # Each edge: [to, capacity, cost, reverse_edge_index]
        graph = [[] for _ in range(n_nodes)]
        
        def add_edge(u, v, cap, cost):
            graph[u].append([v, cap, cost, len(graph[v])])
            graph[v].append([u, 0, -cost, len(graph[u]) - 1])
        
        # Source to customers
        for i in range(n_cust):
            add_edge(source, 1 + i, demands[i], 0.0)
        
        # Customers to facilities
        for i in range(n_cust):
            for j, f in enumerate(allowed_facilities):
                add_edge(1 + i, 1 + n_cust + j, float('inf'), assign_cost[i, f])
        
        # Facilities to sink
        for j, f in enumerate(allowed_facilities):
            add_edge(1 + n_cust + j, sink, capacities[f], 0.0)
        
        # Successive shortest path with potentials
        # Initialize potentials using Bellman-Ford (or just 0 since no negative edges initially)
        potential = np.zeros(n_nodes)
        
        # We need to send total_demand units of flow
        flow_sent = 0.0
        total_cost = 0.0
        assignment = np.full(n_cust, -1, dtype=int)
        
        # Track flow through each customer-facility edge
        # customer_flow[i][j] = flow from customer i to facility j
        customer_flow = np.zeros((n_cust, n_allowed))
        
        while flow_sent < total_demand - 1e-9:
            # Dijkstra with potentials
            dist = np.full(n_nodes, float('inf'))
            dist[source] = 0.0
            prev_node = np.full(n_nodes, -1, dtype=int)
            prev_edge = np.full(n_nodes, -1, dtype=int)
            visited = np.zeros(n_nodes, dtype=bool)
            
            # Use a priority queue
            pq = [(0.0, source)]
            
            while pq:
                d, u = heapq.heappop(pq)
                if visited[u]:
                    continue
                visited[u] = True
                if d > dist[u]:
                    continue
                
                for e_idx, edge in enumerate(graph[u]):
                    v, cap, cost, rev_idx = edge
                    if cap > 1e-9:
                        new_dist = d + cost + potential[u] - potential[v]
                        if new_dist < dist[v] - 1e-12:
                            dist[v] = new_dist
                            prev_node[v] = u
                            prev_edge[v] = e_idx
                            heapq.heappush(pq, (new_dist, v))
            
            if dist[sink] >= float('inf') - 1:
                break  # No augmenting path
            
            # Update potentials
            for i in range(n_nodes):
                if dist[i] < float('inf'):
                    potential[i] += dist[i]
            
            # Find bottleneck capacity along the path
            path_flow = float('inf')
            v = sink
            while v != source:
                u = prev_node[v]
                e_idx = prev_edge[v]
                cap = graph[u][e_idx][1]
                path_flow = min(path_flow, cap)
                v = u
            
            if path_flow <= 1e-9:
                break
            
            # Augment flow
            v = sink
            while v != source:
                u = prev_node[v]
                e_idx = prev_edge[v]
                edge = graph[u][e_idx]
                rev_idx = edge[3]
                
                # Forward edge
                edge[1] -= path_flow
                # Reverse edge
                rev_edge = graph[v][rev_idx]
                rev_edge[1] += path_flow
                
                # Track customer-facility flow
                if u == source:
                    # This is source -> customer edge
                    cust_idx = v - 1
                elif v == sink:
                    # This is facility -> sink edge
                    fac_idx = u - 1 - n_cust
                elif 1 <= u <= n_cust and 1 + n_cust <= v <= n_cust + n_allowed:
                    # This is customer -> facility edge
                    cust_idx = u - 1
                    fac_idx = v - 1 - n_cust
                    customer_flow[cust_idx, fac_idx] += path_flow
                elif 1 + n_cust <= u <= n_cust + n_allowed and 1 <= v <= n_cust:
                    # This is reverse facility -> customer edge (shouldn't happen in forward pass)
                    pass
                
                v = u
            
            flow_sent += path_flow
            total_cost += path_flow * potential[sink]
        
        # Extract assignment from customer_flow
        # For each customer, find which facility got their flow
        for i in range(n_cust):
            max_flow_idx = np.argmax(customer_flow[i])
            if customer_flow[i, max_flow_idx] > 1e-6:
                assignment[i] = allowed_facilities[max_flow_idx]
            else:
                # Fallback: assign to cheapest facility
                min_cost_idx = np.argmin([assign_cost[i, f] for f in allowed_facilities])
                assignment[i] = allowed_facilities[min_cost_idx]
        
        # Verify feasibility
        if not _is_feasible(assignment):
            return None, float('inf')
        
        return assignment, total_cost
    
    def _local_search(result, max_iters=500):
        """
        Local search with 1-move and 2-swap improvements,
        including facility open/close decisions.
        """
        result = result.copy()
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[result[i]] += demands[i]
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # 1-move improvements
            for i in range(n_cust):
                current_f = result[i]
                current_cost = assign_cost[i, current_f]
                
                # Try moving to each other facility
                for f in range(n_fac):
                    if f == current_f:
                        continue
                    if load[f] + demands[i] <= capacities[f] + 1e-9:
                        new_cost = assign_cost[i, f]
                        delta_assign = new_cost - current_cost
                        
                        # Check if opening this facility adds cost
                        delta_open = 0
                        # Count how many customers are at each facility
                        # We need to track facility usage counts
                        # For efficiency, maintain a count array
                        # But for now, let's use a set check
                        if load[f] < 1e-9:  # facility f is currently empty
                            delta_open += open_cost[f]
                        if load[current_f] - demands[i] < 1e-9:  # current facility becomes empty
                            delta_open -= open_cost[current_f]
                        
                        if delta_assign + delta_open < -1e-9:
                            load[current_f] -= demands[i]
                            load[f] += demands[i]
                            result[i] = f
                            improved = True
                            break
            
            # 2-swap improvements
            for i in range(n_cust):
                if improved:
                    break
                fi = result[i]
                for j in range(i + 1, n_cust):
                    fj = result[j]
                    if fi == fj:
                        continue
                    
                    # Check feasibility of swap
                    new_load_fi = load[fi] - demands[i] + demands[j]
                    new_load_fj = load[fj] - demands[j] + demands[i]
                    
                    if new_load_fi <= capacities[fi] + 1e-9 and new_load_fj <= capacities[fj] + 1e-9:
                        delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - \
                                       (assign_cost[i, fi] + assign_cost[j, fj])
                        
                        delta_open = 0
                        # Check if fi becomes empty
                        if new_load_fi < 1e-9:
                            delta_open -= open_cost[fi]
                        # Check if fj becomes empty
                        if new_load_fj < 1e-9:
                            delta_open -= open_cost[fj]
                        
                        if delta_assign + delta_open < -1e-9:
                            load[fi] = new_load_fi
                            load[fj] = new_load_fj
                            result[i] = fj
                            result[j] = fi
                            improved = True
                            break
        
        return result
    
    # Generate facility subsets to try
    subsets_to_try = []
    
    # All facilities
    subsets_to_try.append(list(range(n_fac)))
    
    # Sort by open cost, try subsets of cheapest
    by_open = np.argsort(open_cost)
    for k in [8, 10, 12, 14, 16, 18]:
        subsets_to_try.append(by_open[:k].tolist())
    
    # Sort by average assignment cost
    avg_assign = assign_cost.mean(axis=0)
    by_assign = np.argsort(avg_assign)
    for k in [8, 10, 12, 14, 16, 18]:
        subsets_to_try.append(by_assign[:k].tolist())
    
    # Combined metric: open_cost + avg_assign
    combined = open_cost + avg_assign
    by_combined = np.argsort(combined)
    for k in [8, 10, 12, 14, 16, 18]:
        subsets_to_try.append(by_combined[:k].tolist())
    
    # Ratio metric: open_cost / (avg_assign + epsilon)
    ratio = open_cost / (avg_assign + 1e-9)
    by_ratio = np.argsort(ratio)
    for k in [10, 12, 14, 16]:
        subsets_to_try.append(by_ratio[:k].tolist())
    
    # Remove duplicates
    unique_subsets = []
    seen = set()
    for s in subsets_to_try:
        key = frozenset(s)
        if key not in seen:
            seen.add(key)
            unique_subsets.append(s)
    
    # Try each subset with min-cost flow, then local search
    best_result = None
    best_cost = float('inf')
    
    for subset in unique_subsets:
        if time.time() > deadline - 15:
            break
        
        result, flow_cost = _min_cost_flow_for_subset(subset)
        if result is None:
            continue
        
        # Add open cost
        used = set(result.tolist())
        total = flow_cost + sum(open_cost[f] for f in used)
        
        if total < best_cost:
            best_cost = total
            best_result = result.copy()
    
    # If we have a feasible solution, improve it with local search
    if best_result is not None:
        if time.time() < deadline - 5:
            improved = _local_search(best_result, max_iters=500)
            if _is_feasible(improved):
                new_cost = _total_cost(improved)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_result = improved
        
        # Second round of local search with more iterations if time permits
        if time.time() < deadline - 3:
            improved2 = _local_search(best_result, max_iters=500)
            if _is_feasible(improved2):
                new_cost2 = _total_cost(improved2)
                if new_cost2 < best_cost:
                    best_cost = new_cost2
                    best_result = improved2
    
    # Fallback: simple greedy if nothing else worked
    if best_result is None:
        # Try all facilities with largest-first ordering
        load = np.zeros(n_fac)
        best_result = np.full(n_cust, -1, dtype=int)
        order = np.argsort(-demands)
        
        for i in order:
            placed = False
            costs = assign_cost[i, :]
            sorted_idx = np.argsort(costs)
            for f in sorted_idx:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    load[f] += demands[i]
                    best_result[i] = f
                    placed = True
                    break
            if not placed:
                # Try any facility with remaining capacity
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f] + 1e-9:
                        load[f] += demands[i]
                        best_result[i] = f
                        placed = True
                        break
            if not placed:
                # Last resort: force assign to facility with most remaining capacity
                remaining = capacities - load
                f = np.argmax(remaining)
                load[f] += demands[i]
                best_result[i] = f
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END
