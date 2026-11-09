import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin
    
    n = distance_matrix.shape[0]
    dm = distance_matrix
    d = demands
    customers = np.arange(1, n)
    n_cust = n - 1
    
    # Base heuristic: inverse distance (always positive, captures proximity)
    eps = 1e-9
    base_heur = 1.0 / (dm + eps)
    np.fill_diagonal(base_heur, 0.0)
    
    # Constructive heuristic: simulate capacity-constrained route construction
    # This captures demand-aware grouping that static features miss
    edge_freq = np.zeros((n, n), dtype=float)
    
    # Multi-start nearest-neighbor with capacity, then 2-opt improvement
    n_starts = 7
    rng = np.random.default_rng(42)
    
    for start_idx in range(n_starts):
        # Check time
        if time.time() > deadline - 30:
            break
        
        # Pick a random starting customer (or depot)
        if start_idx == 0:
            current = 0  # Start at depot
        else:
            current = rng.choice(customers)
        
        visited = set()
        remaining_load = capacity
        route = []
        
        # Build route with nearest-neighbor under capacity
        while len(visited) < n_cust:
            if current == 0:
                # At depot, pick nearest unvisited customer that fits
                candidates = []
                for c in customers:
                    if c not in visited and d[c] <= remaining_load:
                        candidates.append(c)
                if not candidates:
                    break
                # Pick nearest
                best_c = min(candidates, key=lambda c: dm[current, c])
            else:
                # At customer, pick next customer that fits
                candidates = []
                for c in customers:
                    if c not in visited and d[c] <= remaining_load:
                        candidates.append(c)
                if not candidates:
                    # Return to depot
                    break
                # Pick nearest
                best_c = min(candidates, key=lambda c: dm[current, c])
            
            # Add edge to route
            route.append((current, best_c))
            visited.add(best_c)
            remaining_load -= d[best_c]
            current = best_c
        
        # Return to depot
        route.append((current, 0))
        
        # Apply 2-opt improvement on the route
        route_customers = [r[0] for r in route[:-1]]  # Exclude final depot return
        if len(route_customers) > 2:
            improved = True
            max_iter = 50
            for _ in range(max_iter):
                if not improved:
                    break
                improved = False
                for i in range(len(route_customers) - 1):
                    for j in range(i + 1, len(route_customers)):
                        # Try reversing segment i+1 to j
                        new_route = route_customers[:i+1] + route_customers[i+1:j+1][::-1] + route_customers[j+1:]
                        # Check capacity feasibility
                        feasible = True
                        load = 0
                        for c in new_route:
                            load += d[c]
                            if load > capacity:
                                feasible = False
                                break
                        if feasible:
                            # Compute cost
                            old_cost = dm[0, route_customers[0]] + sum(dm[route_customers[k], route_customers[k+1]] for k in range(len(route_customers)-1)) + dm[route_customers[-1], 0]
                            new_cost = dm[0, new_route[0]] + sum(dm[new_route[k], new_route[k+1]] for k in range(len(new_route)-1)) + dm[new_route[-1], 0]
                            if new_cost < old_cost - 1e-12:
                                route_customers = new_route
                                improved = True
                                break
                    if improved:
                        break
        
        # Record edge frequencies
        for i in range(len(route_customers) - 1):
            edge_freq[route_customers[i], route_customers[i+1]] += 1
            edge_freq[route_customers[i+1], route_customers[i]] += 1
        if route_customers:
            edge_freq[0, route_customers[0]] += 1
            edge_freq[route_customers[0], 0] += 1
            edge_freq[route_customers[-1], 0] += 1
            edge_freq[0, route_customers[-1]] += 1
    
    # Normalize edge frequencies
    max_freq = np.max(edge_freq)
    if max_freq > 0:
        norm_freq = edge_freq / max_freq
    else:
        norm_freq = np.zeros_like(edge_freq)
    
    # Combine: base heuristic boosted by constructive edge frequency
    # Edges that appear frequently in good constructive solutions get higher heuristic values
    final_heur = base_heur * (1.0 + 2.0 * norm_freq)
    
    # Ensure positivity and finiteness
    final_heur = np.where(final_heur > 0, final_heur, 1e-9)
    final_heur = np.where(np.isfinite(final_heur), final_heur, 1e-9)
    
    if final_heur.shape != distance_matrix.shape:
        final_heur = 1.0 / (distance_matrix + eps)
    
    return final_heur
