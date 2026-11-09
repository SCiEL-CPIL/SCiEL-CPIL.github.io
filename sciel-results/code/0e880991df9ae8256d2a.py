# EVOLVE-BLOCK-START
import numpy as np
import time

def _lagrangian_relaxation(distance_matrix, demands, capacity, n_ants=20, n_iterations=30, seed=42, max_dual_iters=50, max_time=280):
    """
    Solve a Lagrangian relaxation of CVRP to identify edges that are locally optimal
    under relaxed capacity constraints. Returns an edge attractiveness matrix.
    
    The Lagrangian relaxation penalizes capacity violations via dual variables,
    transforming the constrained problem into an unconstrained one where we can
    identify which edges are "valuable" under the relaxed formulation.
    """
    start_time = time.time()
    n = distance_matrix.shape[0]
    
    # Initialize dual variable (lambda) for capacity constraint
    lambda_val = 0.0
    best_lambda = 0.0
    best_relaxed_cost = float('inf')
    
    # Track edge usage across dual iterations
    edge_attractiveness = np.zeros((n, n), dtype=float)
    
    for dual_iter in range(max_dual_iters):
        if time.time() - start_time > max_time:
            break
        
        # Solve relaxed problem: minimize sum(d(i,j)*x(i,j)) + lambda * sum(route_load - capacity)
        # This is equivalent to finding routes where each edge has modified cost:
        # c'(i,j) = d(i,j) + lambda * (demand_j - capacity/n) for customer j
        # We use a greedy construction with modified costs
        
        # Modified distance matrix for relaxed problem
        # For each edge to customer j, add penalty proportional to demand_j
        # The idea: edges to high-demand customers get penalized more when lambda > 0
        modified_dm = distance_matrix.copy()
        for j in range(1, n):
            # Penalty for visiting customer j
            penalty = lambda_val * demands[j]
            modified_dm[:, j] += penalty
            modified_dm[j, :] += penalty  # Symmetric for undirected
        
        # Ensure positive distances
        modified_dm = np.maximum(modified_dm, 1e-9)
        
        # Greedy construction: build routes using nearest neighbor with modified costs
        # but respecting capacity (this gives us a feasible solution under relaxation)
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        current_routes = []
        
        while not visited[1:].all():
            route = [0]
            current = 0
            remaining_cap = capacity
            
            while True:
                # Find eligible customers
                eligible = (~visited) & (np.arange(n) > 0)
                eligible &= (demands <= remaining_cap)
                eligible_indices = np.where(eligible)[0]
                
                if len(eligible_indices) == 0:
                    break
                
                # Pick nearest using modified costs
                costs = modified_dm[current, eligible_indices]
                next_node = eligible_indices[np.argmin(costs)]
                
                route.append(next_node)
                visited[next_node] = True
                remaining_cap -= demands[next_node]
                current = next_node
                
                # If we've visited all, break
                if visited[1:].all():
                    break
            
            route.append(0)
            current_routes.append(route)
        
        # Calculate total cost of this solution
        total_cost = 0.0
        for route in current_routes:
            for k in range(len(route) - 1):
                total_cost += distance_matrix[route[k], route[k+1]]
        
        # Update dual variable using subgradient method
        # Subgradient: sum of (route_load - capacity) for all routes
        subgradient = 0.0
        for route in current_routes:
            route_load = sum(demands[c] for c in route[1:-1])
            subgradient += (route_load - capacity)
        
        # Step size (Polyak-type)
        gap = total_cost - best_relaxed_cost
        if gap > 1e-6:
            step_size = 2.0 * gap / (subgradient**2 + 1e-9)
        else:
            step_size = 0.1
        
        # Update lambda
        lambda_val = max(0.0, lambda_val + step_size * subgradient)
        
        # Track best
        if total_cost < best_relaxed_cost:
            best_relaxed_cost = total_cost
            best_lambda = lambda_val
        
        # Accumulate edge attractiveness: edges used in good solutions get boosted
        # Weight by how "good" this solution is relative to the best
        solution_quality = max(0.0, 1.0 - (total_cost - best_relaxed_cost) / (best_relaxed_cost + 1e-9))
        
        for route in current_routes:
            for k in range(len(route) - 1):
                u, v = route[k], route[k+1]
                # Boost edges that appear in high-quality solutions
                edge_attractiveness[u, v] += solution_quality
                edge_attractiveness[v, u] += solution_quality * 0.5  # Asymmetric: forward direction matters more
    
    # Normalize edge attractiveness
    max_attr = np.max(edge_attractiveness)
    if max_attr > 0:
        edge_attractiveness = edge_attractiveness / max_attr
    
    return edge_attractiveness

def _run_aco(distance_matrix, coordinates, demands, capacity, heuristic, n_ants=20, n_iterations=30, seed=42):
    """Run ACO and return the best (lowest) total route distance found by any ant."""
    rng = np.random.RandomState(seed)
    n = distance_matrix.shape[0]
    pheromone = np.ones_like(heuristic, dtype=float)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_overall = float('inf')
    
    for it in range(n_iterations):
        best_iter_cost = float('inf')
        
        for ant in range(n_ants):
            current = 0
            remaining_cap = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            total_dist = 0.0
            
            while True:
                eligible_mask = (~visited) & (np.arange(n) > 0)
                eligible_mask &= (demands[:n] <= remaining_cap)
                eligible_indices = np.where(eligible_mask)[0]
                
                if len(eligible_indices) == 0:
                    total_dist += distance_matrix[current, 0]
                    current = 0
                    remaining_cap = capacity
                    if visited[1:].all():
                        break
                    continue
                
                probs = pheromone[current, eligible_indices] * heuristic[current, eligible_indices]
                probs = np.where(probs > 0, probs, 1e-9)
                probs /= probs.sum()
                
                idx = rng.choice(len(eligible_indices), p=probs)
                next_node = eligible_indices[idx]
                
                total_dist += distance_matrix[current, next_node]
                remaining_cap -= demands[next_node]
                visited[next_node] = True
                current = next_node
                
                if visited[1:].all():
                    total_dist += distance_matrix[current, 0]
                    break
        
        if total_dist < best_iter_cost:
            best_iter_cost = total_dist
        if total_dist < best_overall:
            best_overall = total_dist
        
        pheromone *= 0.9
        pheromone += 1.0 / best_iter_cost
    
    return best_overall

def _two_opt_improve(routes, distance_matrix, demands, capacity, max_passes=10):
    """
    Improves a set of routes using 2-Opt within each route.
    routes: list of lists of nodes, each route is [0, c1, ..., ck, 0]
    """
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            n_route = len(route)
            if n_route < 4:
                continue
            for i in range(1, n_route - 2):
                for j in range(i + 1, n_route - 1):
                    u = route[i-1]
                    v = route[i]
                    w = route[j]
                    x = route[j+1]
                    
                    old_dist = distance_matrix[u, v] + distance_matrix[w, x]
                    new_dist = distance_matrix[u, w] + distance_matrix[v, x]
                    
                    if new_dist < old_dist - 1e-10:
                        routes[r_idx][i:j+1] = routes[r_idx][i:j+1][::-1]
                        improved = True
    return routes

def _or_opt_improve(routes, distance_matrix, demands, capacity, max_passes=20):
    """
    OR-opt: relocate 1, 2, or 3 consecutive customers from one position to another
    within the same route.
    """
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            n_route = len(route)
            if n_route < 5:
                continue
            for seg_len in [1, 2, 3]:
                for i in range(1, n_route - 1 - seg_len):
                    segment = route[i:i+seg_len]
                    # Try inserting after position j (j != i-1 and j != i+seg_len-1 to avoid trivial)
                    for j in range(1, n_route - 1):
                        if j == i - 1 or j == i + seg_len - 1:
                            continue
                        # Compute old cost: edges around the segment
                        old_cost = distance_matrix[route[i-1], route[i]] + distance_matrix[route[i+seg_len-1], route[i+seg_len]]
                        # Compute new cost: insert segment after position j
                        # Need to be careful about indices after removal
                        # New sequence: ... route[j], seg[0], ..., seg[-1], route[j+1] ...
                        # But if j < i, the indices shift. Let's handle by constructing new route.
                        # For efficiency, compute cost delta:
                        # Remove segment from position i: connect route[i-1] to route[i+seg_len]
                        # Insert segment after position j: connect route[j] to seg[0] and seg[-1] to route[j+1]
                        # But j might be before i, so indices change.
                        
                        # Let's just do it by constructing the new route and checking capacity (same total demand)
                        # Since we're moving within the same route, capacity is preserved.
                        new_route = route[:i] + route[i+seg_len:]
                        # Now insert segment after position j in the new route
                        # But j might have shifted. If j < i, j stays. If j >= i, j shifts by seg_len.
                        if j >= i:
                            new_j = j - seg_len
                        else:
                            new_j = j
                        # new_j should be valid index in new_route
                        if new_j < 1 or new_j >= len(new_route) - 1:
                            continue
                        new_route2 = new_route[:new_j+1] + segment + new_route[new_j+1:]
                        # Verify length
                        if len(new_route2) != len(route):
                            continue
                        # Compute cost difference
                        # Old: edges (i-1,i), (i+seg_len-1, i+seg_len)
                        # New: edges (j, seg[0]), (seg[-1], j+1), and (i-1, i+seg_len)
                        # But if j < i-1 or j > i+seg_len, the edge (i-1, i+seg_len) is new
                        # This is getting complex. Let's just compute total route cost difference approximately.
                        # Actually, let's compute the full route cost for both and compare.
                        # For 50 nodes, this is cheap.
                        def route_cost(r):
                            c = 0.0
                            for k in range(len(r) - 1):
                                c += distance_matrix[r[k], r[k+1]]
                            return c
                        old_total = route_cost(route)
                        new_total = route_cost(new_route2)
                        if new_total < old_total - 1e-10:
                            routes[r_idx] = new_route2
                            improved = True
                            break  # Restart segment search for this route
                    if improved:
                        break
                if improved:
                    break
    return routes

def _savings_heuristic(distance_matrix, demands, capacity, n=51):
    """
    Clarke-Wright Savings heuristic for CVRP.
    Returns list of routes, each [0, c1, ..., ck, 0].
    """
    customers = list(range(1, n))
    active_routes = []
    route_index = {}
    
    for c in customers:
        if demands[c] <= capacity:
            active_routes.append([0, c, 0])
            route_index[c] = len(active_routes) - 1
    
    if not active_routes:
        return []
        
    savings = {}
    for i in customers:
        for j in customers:
            if i == j:
                continue
            s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
            savings[(i, j)] = s
            
    sorted_pairs = sorted(savings.keys(), key=lambda x: savings[x], reverse=True)
    
    def get_route_of(node):
        if node in route_index:
            return active_routes[route_index[node]]
        return None
        
    def is_endpoint(route, node):
        return len(route) >= 3 and (route[1] == node or route[-2] == node)

    for i, j in sorted_pairs:
        ri = route_index.get(i)
        rj = route_index.get(j)
        
        if ri is None or rj is None:
            continue
        if ri == rj:
            continue
            
        route_i = active_routes[ri]
        route_j = active_routes[rj]
        
        i_adj_depot = (route_i[1] == i or route_i[-2] == i)
        j_adj_depot = (route_j[1] == j or route_j[-2] == j)
        
        if not (i_adj_depot and j_adj_depot):
            continue
            
        cap_i = sum(demands[c] for c in route_i[1:-1])
        cap_j = sum(demands[c] for c in route_j[1:-1])
        
        if cap_i + cap_j > capacity:
            continue
            
        if route_i[1] == i and route_j[1] == j:
            inner_j = route_j[1:-1][::-1]
            inner_i = route_i[1:-1]
            new_inner = inner_j + inner_i
            merged = [0] + new_inner + [0]
            
        elif route_i[1] == i and route_j[-2] == j:
            inner_i = route_i[1:-1]
            inner_j = route_j[1:-1]
            merged = [0] + inner_j + inner_i + [0]
            
        elif route_i[-2] == i and route_j[1] == j:
            inner_i = route_i[1:-1]
            inner_j = route_j[1:-1]
            merged = [0] + inner_i + inner_j + [0]
            
        else:
            inner_i = route_i[1:-1][::-1]
            inner_j = route_j[1:-1]
            merged = [0] + inner_i + inner_j + [0]
            
        active_routes[ri] = merged
        del active_routes[rj]
        
        for node in merged[1:-1]:
            route_index[node] = ri
    
    return active_routes

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 280
    
    n = distance_matrix.shape[0]
    
    # 1. Lagrangian Relaxation-based edge weights (primary structural change)
    # This derives attractiveness from dual optimization, not constructive heuristics
    try:
        lagrangian_weights = _lagrangian_relaxation(
            distance_matrix, demands, capacity,
            max_dual_iters=50, max_time=120
        )
    except Exception:
        lagrangian_weights = np.zeros((n, n), dtype=float)
    
    # 2. Generate reference solution using Savings Heuristic + 2-Opt + OR-opt for additional edge info
    try:
        routes = _savings_heuristic(distance_matrix, demands, capacity, n)
        routes = _two_opt_improve(routes, distance_matrix, demands, capacity, max_passes=15)
        routes = _or_opt_improve(routes, distance_matrix, demands, capacity, max_passes=10)
        # Final 2-Opt polish after OR-opt
        routes = _two_opt_improve(routes, distance_matrix, demands, capacity, max_passes=10)
        
        # Extract edges
        edge_matrix = np.zeros((n, n), dtype=float)
        for route in routes:
            for k in range(len(route) - 1):
                u = route[k]
                v = route[k+1]
                edge_matrix[u, v] = 1.0
    except Exception:
        edge_matrix = np.zeros((n, n), dtype=float)
        
    # 3. Base heuristic: 1/d
    dm_safe = distance_matrix + 1e-9
    heuristic = 1.0 / dm_safe
    max_base = np.max(heuristic)
    if max_base > 0:
        heuristic = heuristic / max_base
        
    # 4. Combine: Lagrangian weights (primary) + savings edges (secondary) + base
    # Lagrangian weights are the main signal from dual optimization
    alpha_lagrangian = 8.0
    alpha_savings = 5.0
    
    heuristic = heuristic + alpha_lagrangian * lagrangian_weights + alpha_savings * edge_matrix
    
    # Ensure positive
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Final check
    if not np.all(np.isfinite(heuristic)):
        heuristic = 1.0 / (distance_matrix + 1e-9)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
