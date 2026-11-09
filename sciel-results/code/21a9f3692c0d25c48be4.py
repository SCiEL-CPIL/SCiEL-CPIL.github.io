import numpy as np
from collections import deque

def min_cost_flow_ssp(n_customers, n_facilities, capacities, assign_cost):
    """
    Solves the min-cost flow problem:
    Min sum assign_cost[i,f] * x[i,f]
    s.t. sum_f x[i,f] = 1
         sum_i x[i,f] <= capacities[f]  (Note: this is unit capacity, not weighted)
         x[i,f] in {0,1}
    
    Uses Successive Shortest Path with Potentials.
    """
    n_nodes = n_customers + n_facilities + 2
    source = 0
    sink = n_nodes - 1
    
    # Graph representation: adjacency list of (to, cap, cost, rev_index)
    # We need to handle residual graph.
    # graph[u] = list of [v, cap, cost, rev_index]
    graph = [[] for _ in range(n_nodes)]
    
    def add_edge(u, v, cap, cost):
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])
        
    # Add edges
    for i in range(n_customers):
        add_edge(source, 1 + i, 1, 0)
        
    for f in range(n_facilities):
        add_edge(1 + n_customers + f, sink, capacities[f], 0)
        
    for i in range(n_customers):
        for f in range(n_facilities):
            add_edge(1 + i, 1 + n_customers + f, 1, assign_cost[i, f])
            
    # Potentials
    potential = np.zeros(n_nodes)
    
    # We need to send n_customers units of flow
    flow = 0
    assignment = np.zeros(n_customers, dtype=int)
    
    # Dijkstra with potentials
    # dist[v] = shortest distance from source to v in reduced cost graph
    while flow < n_customers:
        dist = np.full(n_nodes, np.inf)
        dist[source] = 0
        prev_node = np.full(n_nodes, -1)
        prev_edge = np.full(n_nodes, -1)
        
        # Priority queue: (distance, node)
        import heapq
        pq = [(0, source)]
        in_queue = np.zeros(n_nodes, dtype=bool)
        in_queue[source] = True
        
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            in_queue[u] = False # Not strictly needed for standard Dijkstra, but good practice
            for e_idx, (v, cap, cost, rev) in enumerate(graph[u]):
                if cap <= 0:
                    continue
                # Reduced cost
                rc = cost + potential[u] - potential[v]
                new_dist = dist[u] + rc
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev_node[v] = u
                    prev_edge[v] = e_idx
                    heapq.heappush(pq, (new_dist, v))
                    
        if dist[sink] == np.inf:
            break # No more flow possible
            
        # Update potentials
        for v in range(n_nodes):
            if dist[v] < np.inf:
                potential[v] += dist[v]
                
        # Augment flow
        # Find bottleneck
        # Since customer edges have cap 1, bottleneck is 1
        v = sink
        while v != source:
            u = prev_node[v]
            e_idx = prev_edge[v]
            # graph[u][e_idx] is the edge
            graph[u][e_idx][1] -= 1
            graph[v][graph[u][e_idx][3]][1] += 1
            v = u
            
        flow += 1
        
        # Record assignment
        # The last edge added to sink was from a facility
        # We need to trace back to find which customer went to which facility
        # Actually, we can just extract the final flow from the graph
        # Edges from customer i (node 1+i) to facility f (node 1+n_customers+f)
        # If graph[1+i][edge_to_f][1] == 0 (original cap 1, now 0), then flow is 1.
        # Wait, residual cap 0 means forward flow is 1.
        # Let's just check the original edges.
        
    # Extract assignment
    for i in range(n_customers):
        node_i = 1 + i
        for e_idx, (v, cap, cost, rev) in enumerate(graph[node_i]):
            # v should be a facility node
            if 1 + n_customers <= v < 1 + n_customers + n_facilities:
                # If residual capacity is 0, it means 1 unit of flow was sent
                # Original cap was 1.
                if cap == 0:
                    f = v - (1 + n_customers)
                    assignment[i] = f
                    break
                    
    return assignment

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Get initial assignment from Min-Cost Flow (Unit Capacity Relaxed)
    result = min_cost_flow_ssp(n_customers, n_facilities, capacities, assign_cost)
    
    # 2. Calculate load based on result
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[result[i]] += demands[i]
        
    # Check for capacity violations (Weighted)
    # The MCF solution satisfies sum(x_if) <= cap_f
    # But we need sum(d_i * x_if) <= cap_f
    # If violated, local search will fix it.
    
    def total_cost(result, load):
        cost = 0.0
        used_facilities = set(result.tolist())
        for f in used_facilities:
            cost += open_cost[f]
        for i in range(n_customers):
            cost += assign_cost[i, result[i]]
        return cost
        
    current_cost = total_cost(result, load)
    
    # 3. Local Search to fix violations and optimize
    # We need to ensure capacity is respected.
    # The previous local search code assumed feasibility or handled it?
    # The previous code checked feasibility in swaps.
    # We should add a step to fix infeasibilities first if any.
    
    # Simple repair: If load[f] > capacities[f], move customers from f to other facilities with lowest cost increase.
    # But the local search swaps might handle it if we allow "infeasible" states?
    # No, the current local search only applies swaps if the new state is feasible.
    # So if we start infeasible, we might get stuck.
    # Let's add a repair phase.
    
    for f in range(n_facilities):
        if load[f] > capacities[f]:
            # Move customers from f
            customers_at_f = np.where(result == f)[0]
            # Sort by demand descending? Or by cost difference?
            # Move largest demands first to free up space quickly?
            # Or move those with cheapest alternative?
            # Let's move those with cheapest alternative first to minimize cost increase.
            # Calculate alternative cost for each customer at f
            alts = []
            for i in customers_at_f:
                best_alt = -1
                best_cost = np.inf
                for alt_f in range(n_facilities):
                    if alt_f == f:
                        continue
                    if load[alt_f] + demands[i] <= capacities[alt_f]:
                        c = assign_cost[i, alt_f]
                        if c < best_cost:
                            best_cost = c
                            best_alt = alt_f
                if best_alt >= 0:
                    alts.append((best_cost - assign_cost[i, f], i, best_alt))
                else:
                    # No feasible alternative? Must move someone else?
                    # For now, just pick any
                    pass
            
            # Sort by cost increase
            alts.sort()
            for delta_c, i, alt_f in alts:
                if load[f] <= capacities[f]:
                    break
                # Move i to alt_f
                load[f] -= demands[i]
                load[alt_f] += demands[i]
                result[i] = alt_f
                current_cost += delta_c
                
    # Now standard local search
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
                    new_cost -= assign_cost[i, fi] + assign_cost[j, fj]
                    new_cost += assign_cost[i, fj] + assign_cost[j, fi]
                    
                    old_load_fi = load[fi]
                    old_load_fj = load[fj]
                    new_load_fi = old_load_fi - demands[i] + demands[j]
                    new_load_fj = old_load_fj - demands[j] + demands[i]
                    
                    if old_load_fi > 0 and new_load_fi == 0:
                        new_cost -= open_cost[fi]
                    elif old_load_fi == 0 and new_load_fi > 0:
                        new_cost += open_cost[fi]
                    if old_load_fj > 0 and new_load_fj == 0:
                        new_cost -= open_cost[fj]
                    elif old_load_fj == 0 and new_load_fj > 0:
                        new_cost += open_cost[fj]
                    
                    if new_cost < current_cost - 1e-9:
                        result[i] = fj
                        result[j] = fi
                        load[fi] = new_load_fi
                        load[fj] = new_load_fj
                        current_cost = new_cost
                        improved = True
                        
    # Step 4: Try to close facilities
    for f in range(n_facilities):
        if load[f] == 0:
            continue
        customers_at_f = np.where(result == f)[0]
        if len(customers_at_f) == 0:
            continue
        
        new_load = load.copy()
        new_load[f] = 0
        new_result = result.copy()
        
        feasible = True
        for i in customers_at_f:
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
