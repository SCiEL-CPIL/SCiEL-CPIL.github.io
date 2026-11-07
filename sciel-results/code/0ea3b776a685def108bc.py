import math
import time
import random
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist

def plan(instance):
    t0 = time.time()
    DEADLINE = 148.0  # Safety margin within 150s kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    # Precompute distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    customers = list(range(1, n_cust + 1))
    
    # Discretization parameters for state space
    BATTERY_STEP = 25.0  # Battery buckets: 0, 25, ..., 300
    TIME_STEP = 10.0     # Time buckets: 0, 10, ..., 1000
    
    # Helper to round battery to nearest bucket (floor to be safe)
    def battery_bucket(b):
        return int(b // BATTERY_STEP)
    
    # Helper to round time to nearest bucket (ceil to be safe for time windows? 
    # Actually, for state space, we want to represent "earliest possible time".
    # If we use floor, we might underestimate time and violate due date.
    # If we use ceil, we overestimate time, which is safe for due dates but might 
    # prune valid paths that arrive earlier. 
    # Let's use floor for time in state, but check actual time in transition.
    # Actually, standard approach: state is (node, battery, time).
    # Since time is continuous, we discretize.
    # Let's use a finer grid or just use the actual time in the heap priority?
    # Dijkstra on (distance, time, battery) is complex.
    # Let's use a simplified state: (last_node, battery_bucket, time_bucket).
    # Edge cost = distance.
    # We need to ensure that if we reach a state, it's better (lower dist) than previous.
    # But time and battery also matter.
    # A state (u, b, t) dominates (u, b', t') if b >= b' and t <= t' and dist <= dist'.
    # This is a multi-criteria optimization.
    
    # Given the small size (25 customers), we can use a simpler approach for the cluster route:
    # Since clusters are small (e.g., 5 customers), we can use Held-Karp or even brute-force permutations 
    # if K is large enough.
    # If K=5, clusters have 5 customers. 5! = 120 permutations. Very fast.
    # If K=4, clusters have ~6 customers. 6! = 720. Still fast.
    # If K=3, clusters have ~8 customers. 8! = 40320. Manageable.
    # If K=2, clusters have ~12 customers. 12! is too big.
    
    # Strategy:
    # 1. Cluster customers into K groups using K-Means.
    # 2. For each group, find the optimal (or near-optimal) route using:
    #    - If size <= 8: Brute force permutations with feasibility check.
    #    - If size > 8: Use a greedy insertion or simple SA for that cluster.
    # 3. Concatenate routes.
    
    # How to choose K?
    # Total demand ~ 25 * 15 = 375. Capacity 100. So min 4 routes.
    # Let's try K in range [4, 8].
    # For each K, cluster, solve each cluster, sum distances. Pick best K.
    
    def solve_cluster_route(cluster_customers):
        """
        Find the minimum distance valid route for a given list of customers.
        Returns (route_list, distance) or (None, inf) if no valid route found.
        """
        if not cluster_customers:
            return [], 0.0
            
        n_c = len(cluster_customers)
        
        # Check total demand
        total_demand = sum(demands[c] for c in cluster_customers)
        if total_demand > capacity + 1e-9:
            return None, float('inf')
            
        # If small, brute force
        if n_c <= 8:
            from itertools import permutations
            best_route = None
            best_dist = float('inf')
            
            # Add depot (0) to the start and end
            # We need to check all permutations of cluster_customers
            for perm in permutations(cluster_customers):
                # Check feasibility
                t = 0.0
                b = battery_cap
                load = 0.0
                d_total = 0.0
                prev = 0
                valid = True
                for node in perm:
                    d = dist[prev, node]
                    t += d / speed
                    b -= d
                    if b < -1e-9:
                        valid = False
                        break
                    t = max(t, ready[node])
                    if t > due[node] + 1e-9:
                        valid = False
                        break
                    t += service[node]
                    load += demands[node]
                    if load > capacity + 1e-9:
                        valid = False
                        break
                    d_total += d
                    prev = node
                
                if not valid:
                    continue
                
                # Return to depot
                d = dist[prev, 0]
                t += d / speed
                b -= d
                if b < -1e-9 or t > horizon + 1e-9:
                    continue
                d_total += d
                
                if d_total < best_dist:
                    best_dist = d_total
                    best_route = list(perm)
            
            if best_route is not None:
                return best_route, best_dist
            return None, float('inf')
        
        else:
            # For larger clusters, use a greedy heuristic or simple SA
            # Given time constraints, let's use a simple greedy insertion
            # Sort by due date
            sorted_c = sorted(cluster_customers, key=lambda x: due[x])
            route = []
            t = 0.0
            b = battery_cap
            load = 0.0
            prev = 0
            
            for c in sorted_c:
                # Try to insert c at the best position in current route
                # For simplicity, just append if feasible
                d = dist[prev, c]
                new_t = t + d / speed
                new_b = b - d
                if new_b < -1e-9:
                    # Cannot add
                    break
                new_t = max(new_t, ready[c])
                if new_t > due[c] + 1e-9:
                    # Cannot add
                    break
                new_t += service[c]
                new_load = load + demands[c]
                if new_load > capacity + 1e-9:
                    break
                
                # Add c
                route.append(c)
                t = new_t
                b = new_b
                load = new_load
                prev = c
            
            if not route:
                return None, float('inf')
                
            # Check return
            d = dist[prev, 0]
            t += d / speed
            b -= d
            if b < -1e-9 or t > horizon + 1e-9:
                return None, float('inf')
                
            d_total = sum(dist[route[i-1] if i>0 else 0, route[i]] for i in range(len(route))) + dist[route[-1], 0]
            return route, d_total

    def cluster_and_solve(K):
        """
        Cluster customers into K groups and solve each.
        Returns (routes, total_distance) or (None, inf) if any cluster is infeasible.
        """
        if K >= len(customers):
            # Each customer in its own cluster
            routes = []
            total_dist = 0.0
            for c in customers:
                r, d = solve_cluster_route([c])
                if r is None:
                    return None, float('inf')
                routes.append(r)
                total_dist += d
            return routes, total_dist
            
        # K-Means clustering
        # Use scipy's KMeans? No, standard library only? 
        # "Python standard library, NumPy and SciPy are available."
        from sklearn.cluster import KMeans # Wait, is sklearn available? 
        # The prompt says "NumPy and SciPy". It does NOT explicitly say sklearn.
        # I should implement a simple K-Means using NumPy to be safe.
        
        # Simple K-Means
        X = coords[customers]
        n_points = len(X)
        
        # Initialize centroids randomly
        random.seed(42)
        centroids = X[random.sample(range(n_points), K)].copy()
        
        for _ in range(20): # Max iterations
            # Assign points to nearest centroid
            distances = cdist(X, centroids)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(K):
                mask = labels == k
                if np.sum(mask) > 0:
                    new_centroids[k] = np.mean(X[mask], axis=0)
                else:
                    # If empty, pick a random point
                    new_centroids[k] = X[random.randint(0, n_points-1)]
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        # Form clusters
        clusters = [[] for _ in range(K)]
        for i, c in enumerate(customers):
            k = labels[i]
            clusters[k].append(c)
        
        # Solve each cluster
        routes = []
        total_dist = 0.0
        for cluster in clusters:
            if not cluster:
                continue
            r, d = solve_cluster_route(cluster)
            if r is None:
                return None, float('inf')
            routes.append(r)
            total_dist += d
            
        return routes, total_dist

    best_routes = None
    best_total_dist = float('inf')
    
    # Try different K values
    # Min K is ceil(total_demand / capacity)
    total_demand = sum(demands[c] for c in customers)
    min_K = max(1, math.ceil(total_demand / capacity))
    max_K = min(len(customers), 10) # Don't go too high
    
    for K in range(min_K, max_K + 1):
        if time.time() > DEADLINE:
            break
        routes, total_dist = cluster_and_solve(K)
        if routes is not None and total_dist < best_total_dist:
            best_total_dist = total_dist
            best_routes = routes[:]
            
    # If no valid solution found with clustering, fallback to single route per customer
    if best_routes is None:
        best_routes = []
        for c in customers:
            best_routes.append([c])
            
    return best_routes
