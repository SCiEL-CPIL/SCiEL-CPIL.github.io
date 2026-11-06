"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # 10s safety margin before the 120s kill

def _time_exceeded():
    return time.time() > _DEADLINE

# Cache for global instance properties to avoid recomputation on every call
_instance_cache = {}

def _get_instance_properties(depot, unvisited_nodes, demands, distance_matrix):
    """
    Compute global properties of the instance that remain constant during the construction.
    We use the first call to initialize these.
    """
    if 'depot' in _instance_cache:
        # Check if it's the same instance (simple check: depot and matrix shape)
        # In CVRP, the set of nodes is fixed. The 'unvisited' changes, but 'all_nodes' is fixed.
        # We assume the evaluator re-uses the same instance for the 50 customers or similar.
        # To be safe, we key by depot and matrix shape.
        if _instance_cache['depot'] == depot and _instance_cache['shape'] == distance_matrix.shape:
            return _instance_cache
        
    # If cache miss or different instance, recompute
    # We need to infer all nodes. Usually 0..N-1.
    # The distance matrix size gives N.
    n = distance_matrix.shape[0]
    all_nodes = list(range(1, n)) # Exclude depot (0) from customer list for angle calc
    
    # Estimate coordinates? No, we only have distances.
    # We can compute a "pseudo-angle" relative to the depot using distances?
    # Actually, without coordinates, "angle" is hard.
    # However, we can use a **Spectral Embedding** or **MDS** approximation?
    # That might be too slow or complex.
    
    # Alternative Structural Change: **Cluster-Based Routing via K-Means on Distance Matrix**
    # Instead of angles, we cluster the customers into K groups (where K is estimated from capacity).
    # Then, we prefer nodes in the "current" cluster until it's full or empty, then switch.
    # This creates distinct routes.
    
    # How to determine K?
    # Total Demand / Capacity.
    # We don't know total demand easily without summing all demands.
    # But we have 'demands' for all nodes? Yes, 'demands' is passed.
    # Wait, 'demands' is an array of length N.
    total_demand = np.sum(demands[1:])
    capacity = 50 # From prompt: capacity 50
    est_num_routes = max(1, int(np.ceil(total_demand / capacity)))
    
    # Perform K-Means clustering on the distance matrix rows (excluding depot)
    # K-Means on high-dim data (distances to all other nodes) works well for spatial clustering.
    
    # Use a simple deterministic K-Means to avoid randomness issues or seed dependency?
    # Or just use the first `est_num_routes` nodes as initial centroids?
    
    customers = np.array(all_nodes)
    if len(customers) == 0:
        return {'clusters': {}, 'est_num_routes': 0}
    
    # Distance matrix for customers only
    # dm is N x N. We want customer-to-customer distances.
    cust_dm = distance_matrix[1:, 1:] # (N-1) x (N-1)
    
    # Simple K-Means implementation
    k = min(est_num_routes, len(customers))
    if k <= 1:
        clusters = {0: list(customers)}
    else:
        # Initialize centroids: Pick k evenly spaced nodes from the list
        # Or use a random seed? Deterministic is better for reproducibility in evaluation.
        # Let's pick nodes that are far apart from each other (Max-Min initialization).
        centroids = []
        # Start with node 0 (customer 1)
        centroids.append(0)
        
        while len(centroids) < k:
            # Find the node with max min-distance to current centroids
            # dist_to_centroids[i] = min(dist(i, c) for c in centroids)
            # Vectorized:
            # cust_dm is (N-1, N-1). 
            # centroids are indices in 0..N-2
            min_dists = np.min(cust_dm[:, centroids], axis=1)
            next_centroid = np.argmax(min_dists)
            if next_centroid not in centroids:
                centroids.append(next_centroid)
        
        # Assign nodes to nearest centroid
        # Iterate a few times
        labels = np.zeros(len(customers), dtype=int)
        for _ in range(10): # 10 iterations should be enough for 50 points
            # Compute distances to centroids
            # cust_dm[:, centroids] -> (N-1, k)
            dists = cust_dm[:, centroids]
            new_labels = np.argmin(dists, axis=1)
            
            # Update centroids
            # For each cluster, find the point closest to the mean? 
            # Or just pick the point with min sum of dists to others in cluster?
            # Simplest: Keep the label, recompute centroid as the "center" point.
            # In metric spaces, the "center" is often the point with minimum eccentricity.
            # Let's just use the current centroids for assignment, and update centroids 
            # to the point in the cluster with the smallest average distance to other points in the cluster.
            
            changed = False
            for i in range(k):
                mask = new_labels == i
                if np.sum(mask) == 0:
                    continue
                cluster_points = np.where(mask)[0]
                # Calculate average distance to other points in the cluster
                # Sum of dists to all in cluster
                # cust_dm[cluster_points][:, cluster_points].sum(axis=1)
                # This is O(N^2) per cluster, total O(N^2 * K). For N=50, K~5, this is fast.
                sub_dm = cust_dm[np.ix_(cluster_points, cluster_points)]
                sums = np.sum(sub_dm, axis=1)
                best_idx_in_cluster = cluster_points[np.argmin(sums)]
                
                if centroids[i] != best_idx_in_cluster:
                    centroids[i] = best_idx_in_cluster
                    changed = True
            
            if not changed:
                break
            
            labels = new_labels
        
        # Build cluster mapping
        clusters = {}
        for i in range(k):
            members = customers[labels == i]
            clusters[i] = list(members)
            
    _instance_cache['depot'] = depot
    _instance_cache['shape'] = distance_matrix.shape
    _instance_cache['clusters'] = clusters
    _instance_cache['est_num_routes'] = k
    _instance_cache['total_demand'] = total_demand
    
    return _instance_cache

def _select_next_node_clustered(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Strategy:
    1. Determine which cluster the current_node belongs to.
    2. Prefer nodes in the same cluster that are feasible and close.
    3. If no feasible nodes in same cluster, or cluster is "done" (no unvisited in it), 
       switch to the "best" next cluster (closest to current node or depot).
    
    This forces the route to stay within a geographic/structural cluster,
    reducing the number of route changes and crossing routes.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Filter by capacity
    feasible = [i for i in unvisited_nodes if demands[i] <= rest_capacity]
    if len(feasible) == 0:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])

    if _time_exceeded():
        # Fallback to nearest neighbor
        idx = np.argmin(distance_matrix[current_node, feasible])
        return int(feasible[idx])

    # Get instance properties
    props = _get_instance_properties(depot, unvisited_nodes, demands, distance_matrix)
    clusters = props['clusters']
    
    # Map node -> cluster_id
    node_to_cluster = {}
    for cid, members in clusters.items():
        for m in members:
            node_to_cluster[m] = cid
            
    current_cluster = node_to_cluster.get(current_node, -1)
    
    # 1. Find feasible nodes in the current cluster
    same_cluster_feasible = [u for u in feasible if node_to_cluster.get(u, -1) == current_cluster]
    
    if len(same_cluster_feasible) > 0:
        # Choose the nearest one in the same cluster
        # This encourages completing the current cluster's route
        u_arr = np.array(same_cluster_feasible)
        dists = distance_matrix[current_node, u_arr]
        min_idx = np.argmin(dists)
        return int(u_arr[min_idx])
    
    # 2. If no feasible nodes in current cluster, we must switch clusters.
    # Which cluster to switch to?
    # Heuristic: The cluster that has a node closest to the current_node.
    # This minimizes the "travel" to the next cluster.
    
    # Group feasible nodes by cluster
    cluster_candidates = {}
    for u in feasible:
        cid = node_to_cluster.get(u, -1)
        if cid not in cluster_candidates:
            cluster_candidates[cid] = []
        cluster_candidates[cid].append(u)
    
    # For each candidate cluster, find the node closest to current_node
    best_cluster_node = None
    best_dist = float('inf')
    
    for cid, nodes in cluster_candidates.items():
        u_arr = np.array(nodes)
        dists = distance_matrix[current_node, u_arr]
        min_d = np.min(dists)
        if min_d < best_dist:
            best_dist = min_d
            best_cluster_node = u_arr[np.argmin(dists)]
            
    if best_cluster_node is not None:
        return int(best_cluster_node)
    
    # Fallback (should not happen if feasible is not empty)
    idx = np.argmin(distance_matrix[current_node, feasible])
    return int(feasible[idx])

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: Cluster-Based Node Selection.
    
    Uses a structural clustering of the instance (based on distance matrix) to
    guide the construction. It prefers to stay within the current cluster until
    it is empty or infeasible, then moves to the nearest cluster. This reduces
    route crossings and improves the mean total route length.
    """
    return _select_next_node_clustered(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)

# EVOLVE-BLOCK-END
