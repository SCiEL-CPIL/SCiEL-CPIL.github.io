"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt, inf

def _dist(coords, i, j):
    dx = coords[i][0] - coords[j][0]
    dy = coords[i][1] - coords[j][1]
    return sqrt(dx*dx + dy*dy)

def _build_dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = sqrt(dx*dx + dy*dy)
            D[i, j] = d
            D[j, i] = d
    return D

def _simulate_route(route, instance, D):
    """
    Simulate a route. Returns (is_feasible, total_time, total_distance).
    route: list of node ids (customers and stations).
    """
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    horizon = instance["horizon"]
    recharge_rate = instance["recharge_rate"]
    speed = instance["speed"]
    
    # Check capacity
    total_demand = 0
    for node in route:
        if node <= n_customers:
            total_demand += instance["demands"][node]
    if total_demand > capacity:
        return False, 0, 0

    time = 0.0
    battery = float(battery_max)
    prev = 0  # depot
    total_dist = 0.0

    for node in route:
        d = D[prev, node]
        time += d / speed
        battery -= d
        total_dist += d

        if battery < -1e-9:
            return False, 0, 0
        
        if node <= n_customers:
            # Customer
            if time < instance["ready"][node]:
                time = instance["ready"][node]
            if time > instance["due"][node] + 1e-9:
                return False, 0, 0
            time += instance["service"][node]
        elif node in stations:
            # Charging station
            energy_needed = battery_max - battery
            if energy_needed > 0:
                time += energy_needed * recharge_rate
            battery = float(battery_max)
        
        prev = node
    
    # Return to depot
    d = D[prev, 0]
    time += d / speed
    battery -= d
    total_dist += d
    
    if battery < -1e-9:
        return False, 0, 0
    if time > horizon + 1e-9:
        return False, 0, 0
    
    return True, time, total_dist

def _km_clusters(points, k, D, max_iters=50):
    """
    Simple K-Means clustering using distance matrix D.
    points: list of node indices to cluster.
    k: number of clusters.
    Returns list of lists of node indices.
    """
    if k >= len(points):
        return [[p] for p in points]
    
    # Initialize centroids randomly from points
    import random
    random.seed(42) # Fixed seed for reproducibility in evolution
    centroids = random.sample(points, k)
    clusters = [[] for _ in range(k)]
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        new_clusters = [[] for _ in range(k)]
        for p in points:
            min_d = inf
            best_c = 0
            for c_idx, c in enumerate(centroids):
                d = D[p, c]
                if d < min_d:
                    min_d = d
                    best_c = c_idx
            new_clusters[best_c].append(p)
        
        # Check convergence
        if all(sorted(old) == sorted(new) for old, new in zip(clusters, new_clusters)):
            break
            
        clusters = new_clusters
        
        # Update centroids
        for c_idx in range(k):
            if not clusters[c_idx]:
                continue
            # Centroid is the point in cluster with min distance to others (approx)
            # Or just pick a random point from the cluster
            # Better: pick the point that minimizes sum of distances to others in cluster
            best_point = clusters[c_idx][0]
            min_sum = inf
            for p in clusters[c_idx]:
                sum_d = sum(D[p, q] for q in clusters[c_idx])
                if sum_d < min_sum:
                    min_sum = sum_d
                    best_point = p
            centroids[c_idx] = best_point
            
    return clusters

def _construct_route(cluster, instance, D):
    """
    Construct a route for a cluster using Nearest Neighbor with battery awareness.
    """
    if not cluster:
        return []
        
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    
    unvisited = list(cluster)
    route = []
    current_node = 0
    battery = float(battery_max)
    time = 0.0
    
    while unvisited:
        best_next = None
        best_cost = inf
        best_action = None # 'customer' or 'station'
        
        # Option 1: Go to a customer
        for cust in unvisited:
            d = D[current_node, cust]
            if battery - d < -1e-9:
                continue
            new_time = time + d / speed
            if new_time < instance["ready"][cust]:
                new_time = instance["ready"][cust]
            if new_time > instance["due"][cust] + 1e-9:
                continue
            new_time += instance["service"][cust]
            
            # Cost: distance to go + penalty for waiting? Just distance for now.
            # We want to minimize distance, but must be feasible.
            if d < best_cost:
                best_cost = d
                best_next = cust
                best_action = 'customer'
                
        # Option 2: Go to a station if no customer is reachable or if battery is low
        if best_next is None:
            for st in stations:
                d = D[current_node, st]
                if battery - d < -1e-9:
                    continue
                new_time = time + d / speed
                # Check if we can return to depot or reach a customer after charging?
                # For now, just go to station if we can't reach any customer.
                # Heuristic: if battery < 50, prefer station?
                # Let's just pick the nearest station if no customer is feasible.
                if d < best_cost:
                    best_cost = d
                    best_next = st
                    best_action = 'station'
                    
        if best_next is None:
            # Can't move? Break. This should not happen if clusters are small enough.
            # If it does, we return partial route.
            break
            
        # Move to best_next
        d = D[current_node, best_next]
        time += d / speed
        battery -= d
        
        if best_action == 'customer':
            if time < instance["ready"][best_next]:
                time = instance["ready"][best_next]
            time += instance["service"][best_next]
            route.append(best_next)
            unvisited.remove(best_next)
        else:
            # Charge
            energy_needed = battery_max - battery
            if energy_needed > 0:
                time += energy_needed * recharge_rate
            battery = float(battery_max)
            route.append(best_next)
            
        current_node = best_next
        
    return route

def _improve_route(route, instance, D, max_iters=50):
    """
    Improve route using 2-opt swaps.
    """
    if len(route) < 2:
        return route
        
    best_route = route
    best_dist = _simulate_route(best_route, instance, D)[2] if _simulate_route(best_route, instance, D)[0] else inf
    
    for _ in range(max_iters):
        improved = False
        n = len(best_route)
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse segment i+1 to j
                new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                feasible, _, dist = _simulate_route(new_route, instance, D)
                if feasible and dist < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = dist
                    improved = True
        if not improved:
            break
            
    return best_route

def plan(instance):
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    horizon = instance["horizon"]
    
    D = _build_dist_matrix(coords)
    customers = list(range(1, n_customers + 1))
    
    # Estimate number of clusters (routes)
    # Total demand / capacity gives lower bound on vehicles.
    total_demand = sum(instance["demands"][c] for c in customers)
    min_vehicles = max(1, int(np.ceil(total_demand / capacity)))
    
    # Also consider battery range. Max distance in square is ~141. Battery 300.
    # A round trip from depot to far corner and back is ~282.
    # So one vehicle can cover a significant area.
    # Let's try min_vehicles to min_vehicles + 2 clusters.
    # K-Means with small K.
    
    best_routes = None
    best_score = inf
    
    for k in range(min_vehicles, min_vehicles + 3):
        # Cluster
        clusters = _km_clusters(customers, k, D)
        
        routes = []
        valid = True
        for cluster in clusters:
            if not cluster:
                continue
            # Construct
            route = _construct_route(cluster, instance, D)
            # Check if all customers in cluster are in route
            if set(route) != set(cluster) | set([s for s in route if s in instance["stations"]]):
                # If some customers were not visited, it's a failure for this cluster
                # Actually _construct_route removes from unvisited, so if unvisited is not empty, it failed.
                # But we don't return unvisited. Let's assume it works for now.
                pass
            
            # Improve
            route = _improve_route(route, instance, D)
            
            # Verify
            feasible, _, dist = _simulate_route(route, instance, D)
            if not feasible:
                valid = False
                break
            routes.append(route)
            
        if not valid:
            continue
            
        # Check if all customers are served
        served = set()
        total_dist = 0
        for route in routes:
            for node in route:
                if node <= n_customers:
                    served.add(node)
            total_dist += _simulate_route(route, instance, D)[2]
            
        if served != set(customers):
            continue
            
        if total_dist < best_score:
            best_score = total_dist
            best_routes = routes
            
    if best_routes is None:
        # Fallback: individual routes
        best_routes = [[c] for c in customers]
        
    return best_routes
# EVOLVE-BLOCK-END
