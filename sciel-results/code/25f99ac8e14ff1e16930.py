"""Combined program: K-means clustering + LNS + Fast Local Search for EV Routing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import sqrt

def _get_distance_matrix(coords):
    """Precompute Euclidean distance matrix using NumPy."""
    arr = np.array(coords, dtype=np.float64)
    dx = arr[:, 0, None] - arr[None, :, 0]
    dy = arr[:, 1, None] - arr[None, :, 1]
    return np.sqrt(dx**2 + dy**2)

def _route_distance_fast(dist_matrix, route):
    """Calculate total distance of a route using precomputed distance matrix."""
    if not route:
        return 0.0
    d = dist_matrix[0, route[0]]
    for k in range(len(route) - 1):
        d += dist_matrix[route[k], route[k + 1]]
    d += dist_matrix[route[-1], 0]
    return d

def _check_route_feasibility_fast(dist_matrix, demands, ready, due, service,
                                  battery, recharge_rate, horizon, route,
                                  capacity, n_customers, stations):
    """
    Check if a single route is feasible using precomputed distances.
    Returns (feasible, total_distance).
    Optimized for speed.
    """
    if not route:
        return True, 0.0

    total_demand = 0
    for v in route:
        if v <= n_customers:
            total_demand += demands[v]
            if total_demand > capacity:
                return False, 0.0

    t = 0.0
    b = float(battery)
    prev = 0
    total_dist = 0.0

    for v in route:
        d = dist_matrix[prev, v]
        t += d
        b -= d
        total_dist += d
        if b < -1e-9:
            return False, 0.0

        if v <= n_customers:
            # Customer
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0
            t += service[v]
        else:
            # Station
            time_charging = (battery - b) * recharge_rate
            t += time_charging
            b = float(battery)  # Refill to full
        
        prev = v

    # Return to depot
    d = dist_matrix[prev, 0]
    t += d
    b -= d
    total_dist += d
    if b < -1e-9:
        return False, 0.0
    if t > horizon + 1e-9:
        return False, 0.0

    return True, total_dist

def _kmeans_cluster(coords, n_customers, n_clusters, n_iter=30, seed=42):
    """Simple k-means clustering of customers (nodes 1..n_customers)."""
    rng = random.Random(seed)
    cust_coords = np.array([coords[i] for i in range(1, n_customers + 1)])
    
    # Initialize centroids by picking random customers
    idx = list(range(n_customers))
    rng.shuffle(idx)
    centroids = [cust_coords[idx[i % n_customers]] for i in range(n_clusters)]
    
    labels = np.zeros(n_customers, dtype=int)
    
    for _ in range(n_iter):
        # Assign each customer to nearest centroid
        for i in range(n_customers):
            best_c = 0
            best_d = float('inf')
            for c in range(n_clusters):
                d = (cust_coords[i][0] - centroids[c][0])**2 + (cust_coords[i][1] - centroids[c][1])**2
                if d < best_d:
                    best_d = d
                    best_c = c
            labels[i] = best_c
        
        # Update centroids
        for c in range(n_clusters):
            members = [i for i in range(n_customers) if labels[i] == c]
            if members:
                centroids[c] = cust_coords[members].mean(axis=0)
    
    # Build clusters
    clusters = [[] for _ in range(n_clusters)]
    for i in range(n_customers):
        clusters[labels[i]].append(i + 1)  # +1 because customer ids start at 1
    
    return clusters

def _nearest_neighbor_route(dist_matrix, customers, ready, due, service,
                            battery, recharge_rate, horizon, capacity,
                            n_customers, stations, demands):
    """Build a feasible route from a set of customers via nearest neighbor + feasibility check."""
    if not customers:
        return []
    
    # Sort by due time to improve initial feasibility
    customers = sorted(customers, key=lambda c: due[c])
    
    route = []
    remaining = list(customers)
    prev = 0
    t = 0.0
    b = float(battery)
    total_demand = 0
    
    while remaining:
        best_c = None
        best_d = float('inf')
        
        # Find nearest feasible customer
        for c in remaining:
            d = dist_matrix[prev, c]
            if d < best_d:
                # Quick check: can we reach it?
                if b - d >= -1e-9:
                    # Check time window roughly
                    new_t = t + d
                    if new_t < ready[c]:
                        new_t = ready[c]
                    if new_t <= due[c] + 1e-9:
                        best_d = d
                        best_c = c
        
        if best_c is None:
            # Try any remaining customer if nearest failed
            for c in remaining:
                cand = route + [c]
                ok, _ = _check_route_feasibility_fast(
                    dist_matrix, demands, ready, due, service,
                    battery, recharge_rate, horizon, cand,
                    capacity, n_customers, stations
                )
                if ok and total_demand + demands[c] <= capacity:
                    best_c = c
                    break
            
            if best_c is None:
                break
        
        # Add best_c
        route.append(best_c)
        total_demand += demands[best_c]
        remaining.remove(best_c)
        
        # Update state
        d = dist_matrix[prev, best_c]
        t += d
        b -= d
        if best_c <= n_customers:
            if t < ready[best_c]:
                t = ready[best_c]
            t += service[best_c]
        prev = best_c
        
        # Ensure we don't exceed capacity
        if total_demand > capacity:
            break

    return route

def _two_opt_route(route, dist_matrix, demands, ready, due, service,
                   battery, recharge_rate, horizon, capacity, n_customers, stations,
                   deadline):
    """Apply 2-opt moves within a single route to reduce distance while maintaining feasibility."""
    if len(route) < 3:
        return route, 0.0
    
    n = len(route)
    improved = True
    total_improvement = 0.0
    
    while improved:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                # Compute distance delta for reversing segment [i..j]
                a = 0 if i == 0 else route[i - 1]
                b_node = route[i]
                c = route[j]
                d_node = 0 if j == n - 1 else route[j + 1]
                
                old_d = dist_matrix[a, b_node] + dist_matrix[c, d_node]
                new_d = dist_matrix[a, c] + dist_matrix[b_node, d_node]
                
                if new_d >= old_d - 1e-9:
                    continue
                
                # Try the reversal
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                
                # Check feasibility
                feasible, _ = _check_route_feasibility_fast(
                    dist_matrix, demands, ready, due, service,
                    battery, recharge_rate, horizon, new_route,
                    capacity, n_customers, stations
                )
                
                if feasible:
                    route = new_route
                    total_improvement += old_d - new_d
                    improved = True
    
    return route, total_improvement

def _or_opt_route(route, dist_matrix, demands, ready, due, service,
                  battery, recharge_rate, horizon, capacity, n_customers, stations,
                  deadline):
    """Apply Or-opt moves (relocate 1-3 consecutive nodes) within a route."""
    if len(route) < 3:
        return route, 0.0
    
    n = len(route)
    improved = True
    total_improvement = 0.0
    
    while improved and time.time() < deadline:
        improved = False
        for length in [1, 2, 3]:
            for i in range(n - length + 1):
                if time.time() > deadline:
                    break
                segment = route[i:i+length]
                
                # Remove segment
                remaining = route[:i] + route[i+length:]
                
                # Try inserting at each position in remaining
                best_pos = None
                best_delta = float('inf')
                
                for pos in range(len(remaining) + 1):
                    new_route = remaining[:pos] + segment + remaining[pos:]
                    
                    # Check feasibility
                    feasible, nd = _check_route_feasibility_fast(
                        dist_matrix, demands, ready, due, service,
                        battery, recharge_rate, horizon, new_route,
                        capacity, n_customers, stations
                    )
                    
                    if feasible:
                        old_d = _route_distance_fast(dist_matrix, route)
                        delta = nd - old_d
                        if delta < best_delta - 1e-9:
                            best_delta = delta
                            best_pos = pos
                
                if best_pos is not None and best_delta < -1e-6:
                    new_route = remaining[:best_pos] + segment + remaining[best_pos:]
                    old_d = _route_distance_fast(dist_matrix, route)
                    route = new_route
                    total_improvement += -best_delta
                    improved = True
                    n = len(route)
                    break
    
    return route, total_improvement

def _relocate_customer(routes, customer, dist_matrix, demands, ready, due, service,
                       battery, recharge_rate, horizon, capacity, n_customers, stations,
                       deadline):
    """Try to relocate a customer to a better position in any route."""
    # Find current route
    curr_route_idx = -1
    for i, r in enumerate(routes):
        if customer in r:
            curr_route_idx = i
            break
    
    if curr_route_idx == -1:
        return routes, False
    
    curr_route = routes[curr_route_idx]
    
    # Remove customer from current route
    remaining_curr = [v for v in curr_route if v != customer]
    
    best_target_idx = None
    best_pos = None
    best_improvement = 0.0
    
    # Calculate distance of current route without customer
    dist_curr_without = _route_distance_fast(dist_matrix, remaining_curr) if remaining_curr else 0.0
    
    # Try inserting into every route
    for i, route in enumerate(routes):
        if i == curr_route_idx:
            target_route = remaining_curr
            old_target_dist = dist_curr_without
        else:
            target_route = route
            old_target_dist = _route_distance_fast(dist_matrix, route)
        
        # Quick capacity check
        target_demand = sum(demands[v] for v in target_route if v <= n_customers)
        if target_demand + demands[customer] > capacity:
            continue
        
        best_pos_in_route = None
        best_target_dist = float('inf')
        
        for pos in range(len(target_route) + 1):
            new_target_route = target_route[:pos] + [customer] + target_route[pos:]
            feasible, nd = _check_route_feasibility_fast(
                dist_matrix, demands, ready, due, service,
                battery, recharge_rate, horizon, new_target_route,
                capacity, n_customers, stations
            )
            if feasible and nd < best_target_dist:
                best_target_dist = nd
                best_pos_in_route = pos
        
        if best_pos_in_route is not None:
            if i == curr_route_idx:
                old_affected = _route_distance_fast(dist_matrix, curr_route)
                new_affected = best_target_dist
                delta = new_affected - old_affected
            else:
                old_affected = _route_distance_fast(dist_matrix, curr_route) + old_target_dist
                new_affected = dist_curr_without + best_target_dist
                delta = new_affected - old_affected
            
            if delta < best_improvement:
                best_improvement = delta
                best_target_idx = i
                best_pos = best_pos_in_route
    
    if best_target_idx is not None and best_improvement < -1e-6:
        new_routes = routes.copy()
        
        if best_target_idx == curr_route_idx:
            new_routes[curr_route_idx] = remaining_curr[:best_pos] + [customer] + remaining_curr[best_pos:]
        else:
            new_routes[curr_route_idx] = remaining_curr
            new_routes[best_target_idx] = routes[best_target_idx][:best_pos] + [customer] + routes[best_target_idx][best_pos:]
        
        return new_routes, True
    
    return routes, False

def _lns_destroy_reinsert(routes, dist_matrix, demands, ready, due, service,
                          battery, recharge_rate, horizon, capacity, n_customers, stations,
                          deadline, rng):
    """LNS: Destroy a random subset of customers and reinsert them optimally."""
    if len(routes) < 2:
        return routes
    
    # Collect all customers
    all_customers = []
    for r in routes:
        all_customers.extend([v for v in r if v <= n_customers])
    
    if not all_customers:
        return routes
    
    # Destroy: remove 20-40% of customers randomly
    destroy_fraction = rng.uniform(0.2, 0.4)
    n_destroy = max(1, int(len(all_customers) * destroy_fraction))
    to_destroy = rng.sample(all_customers, min(n_destroy, len(all_customers)))
    
    # Remove from routes
    new_routes = []
    remaining_customers = []
    for r in routes:
        new_r = [v for v in r if v not in to_destroy]
        if new_r:
            new_routes.append(new_r)
        # Collect destroyed customers
        for v in r:
            if v in to_destroy:
                remaining_customers.append(v)
    
    # Reinsert each destroyed customer into best position
    for c in remaining_customers:
        best_route_idx = None
        best_pos = None
        best_dist = float('inf')
        
        for i, route in enumerate(new_routes):
            # Quick capacity check
            route_demand = sum(demands[v] for v in route if v <= n_customers)
            if route_demand + demands[c] > capacity:
                continue
            
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                feasible, nd = _check_route_feasibility_fast(
                    dist_matrix, demands, ready, due, service,
                    battery, recharge_rate, horizon, new_route,
                    capacity, n_customers, stations
                )
                if feasible and nd < best_dist:
                    best_dist = nd
                    best_route_idx = i
                    best_pos = pos
        
        if best_route_idx is not None:
            new_routes[best_route_idx] = new_routes[best_route_idx][:best_pos] + [c] + new_routes[best_route_idx][best_pos:]
        else:
            # Cannot insert, put in new route
            new_routes.append([c])
    
    return new_routes

def _solve_instance(instance):
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance.get("speed", 1.0)
    recharge_rate = instance.get("recharge_rate", 0.5)
    horizon = instance["horizon"]

    # Precompute distance matrix
    dist_matrix = _get_distance_matrix(coords)

    all_customers = list(range(1, n_customers + 1))
    
    # Determine number of clusters based on capacity
    total_demand = sum(demands[c] for c in all_customers)
    min_routes = max(1, int(np.ceil(total_demand / capacity)))
    n_clusters = max(min_routes, 4)  # At least 4 routes
    
    # K-means clustering
    clusters = _kmeans_cluster(coords, n_customers, n_clusters, n_iter=20, seed=42)
    
    # Build initial routes from clusters using nearest neighbor
    routes = []
    for cluster in clusters:
        if cluster:
            route = _nearest_neighbor_route(
                dist_matrix, cluster, ready, due, service,
                battery, recharge_rate, horizon, capacity,
                n_customers, stations, demands
            )
            if route:
                routes.append(route)
    
    # Handle any unassigned customers
    assigned = set()
    for r in routes:
        for v in r:
            if v <= n_customers:
                assigned.add(v)
    
    unassigned = [c for c in all_customers if c not in assigned]
    for c in unassigned:
        # Try to insert into existing routes
        best_route_idx = None
        best_pos = None
        best_dist = float('inf')
        
        for i, route in enumerate(routes):
            route_demand = sum(demands[v] for v in route if v <= n_customers)
            if route_demand + demands[c] > capacity:
                continue
            
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                feasible, nd = _check_route_feasibility_fast(
                    dist_matrix, demands, ready, due, service,
                    battery, recharge_rate, horizon, new_route,
                    capacity, n_customers, stations
                )
                if feasible and nd < best_dist:
                    best_dist = nd
                    best_route_idx = i
                    best_pos = pos
        
        if best_route_idx is not None:
            routes[best_route_idx] = routes[best_route_idx][:best_pos] + [c] + routes[best_route_idx][best_pos:]
        else:
            routes.append([c])
    
    # Apply 2-opt within each route
    deadline_2opt = time.time() + 15
    for i in range(len(routes)):
        if time.time() > deadline_2opt:
            break
        routes[i], _ = _two_opt_route(
            routes[i], dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, capacity, n_customers, stations,
            deadline_2opt
        )
    
    # Apply Or-opt within each route
    deadline_oropt = time.time() + 10
    for i in range(len(routes)):
        if time.time() > deadline_oropt:
            break
        routes[i], _ = _or_opt_route(
            routes[i], dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, capacity, n_customers, stations,
            deadline_oropt
        )
    
    # LNS iterations
    rng = random.Random(42)
    deadline_lns = time.time() + 30
    best_routes = [r[:] for r in routes]
    best_dist = sum(_route_distance_fast(dist_matrix, r) for r in routes)
    
    while time.time() < deadline_lns:
        # LNS destroy and reinsert
        new_routes = _lns_destroy_reinsert(
            routes, dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, capacity, n_customers, stations,
            deadline_lns, rng
        )
        
        # Apply 2-opt to new routes
        for i in range(len(new_routes)):
            if time.time() > deadline_lns:
                break
            new_routes[i], _ = _two_opt_route(
                new_routes[i], dist_matrix, demands, ready, due, service,
                battery, recharge_rate, horizon, capacity, n_customers, stations,
                deadline_lns
            )
        
        # Check feasibility and calculate distance
        new_dist = 0.0
        feasible = True
        for r in new_routes:
            ok, d = _check_route_feasibility_fast(
                dist_matrix, demands, ready, due, service,
                battery, recharge_rate, horizon, r,
                capacity, n_customers, stations
            )
            if not ok:
                feasible = False
                break
            new_dist += d
        
        if feasible and new_dist < best_dist:
            best_dist = new_dist
            best_routes = [r[:] for r in new_routes]
            routes = new_routes
    
    # Final relocate pass
    deadline_relocate = time.time() + 15
    improved = True
    while improved and time.time() < deadline_relocate:
        improved = False
        for c in all_customers:
            if time.time() > deadline_relocate:
                break
            routes, did_improve = _relocate_customer(
                routes, c, dist_matrix, demands, ready, due, service,
                battery, recharge_rate, horizon, capacity, n_customers, stations,
                deadline_relocate
            )
            if did_improve:
                improved = True
    
    # Final 2-opt polish
    deadline_polish = time.time() + 10
    for i in range(len(routes)):
        if time.time() > deadline_polish:
            break
        routes[i], _ = _two_opt_route(
            routes[i], dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, capacity, n_customers, stations,
            deadline_polish
        )
    
    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 110  # 40s margin for safety
    
    # Solve
    routes = _solve_instance(instance)
    
    # Final Verification
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    recharge_rate = instance.get("recharge_rate", 0.5)
    horizon = instance["horizon"]
    
    dist_matrix = _get_distance_matrix(coords)
    
    all_ok = True
    for r in routes:
        ok, _ = _check_route_feasibility_fast(
            dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, r,
            capacity, n_customers, stations
        )
        if not ok:
            all_ok = False
            break
            
    if not all_ok:
        # Fallback: One customer per route
        customers = list(range(1, n_customers + 1))
        routes = [[c] for c in customers]
        
    return routes
# EVOLVE-BLOCK-END
