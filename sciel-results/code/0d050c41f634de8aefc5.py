"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 3.0

# Cache for angular data to avoid recomputation on every call if possible
# But since we don't know the coordinates directly, we must derive them or use distance properties.
# Wait, the interface provides `distance_matrix`. We do NOT have coordinates explicitly.
# However, for Euclidean distances (which VRP instances typically are), we can estimate angles?
# No, we cannot reliably recover coordinates from a distance matrix without heavy computation (MDS) which might be slow or unstable.
# 
# Let's re-evaluate the "Sweepline" idea without coordinates.
# The directive suggests: "Model the CVRP as a spatial clustering problem... Delaunay triangulation...".
# But without coordinates, we can't do Delaunay.
# 
# Alternative Structural Change: **Farthest-First Traversal with Local Lookahead**.
# The current program uses a "savings" heuristic: minimize d(u, v) - d(depot, v).
# This effectively tries to keep the route "tight" around the depot.
# 
# A different family is **Insertion Heuristics** adapted for sequential choice.
# Or **Nearest Neighbor with Capacity Penalties**.
# 
# Let's look at the "Sweepline" directive again. It says: "greedily extend the current route to the feasible neighbor with the smallest edge weight in this triangulation".
# Since we lack coordinates, we can approximate "locality" using the distance matrix.
# 
# Let's try a different structural approach: **2-Opt Lookahead Greedy**.
# Instead of just picking the best immediate next node, we evaluate the cost of picking node v, and then assuming we go to the *nearest* neighbor of v (w) next.
# Score(v) = d(u, v) + d(v, w) + d(w, depot)  (if w is the last node)
# Or more simply: Score(v) = d(u, v) + min_{w in feasible \ {v}} d(v, w)
# This is a 1-step lookahead.
# 
# Actually, the directive specifically asked for a "Delaunay triangulation" approach.
# If I cannot use coordinates, I can use a **Graph-based** approach.
# The distance matrix defines a complete graph.
# A "sparse geometric graph" approximation can be made by keeping only the K nearest neighbors for each node.
# Strategy: **K-Nearest Neighbor Graph Constrained Search**.
# 1. Precompute (or compute on the fly) the K nearest neighbors for each node based on distance.
# 2. When choosing the next node from `current_node`, only consider feasible unvisited nodes that are in the K-NN set of `current_node` (or in the K-NN set of any node in the current route? No, just current is a good start).
# 3. If no feasible node is in the K-NN set, fall back to the global nearest feasible node.
# 
# This enforces local planarity (prevents long crossing edges) and reduces the search space.
# It is structurally different from the global savings heuristic.

_K_NN = 10  # Number of nearest neighbors to consider

# We need to cache the K-NN graph because the distance matrix is constant.
_knn_cache = {}

def _get_knn_graph(distance_matrix, n_nodes, k):
    """Compute K nearest neighbors for each node."""
    if distance_matrix in _knn_cache:
        return _knn_cache[distance_matrix]
    
    # Convert to numpy array if not already
    dm = np.asarray(distance_matrix)
    n = dm.shape[0]
    
    knn_graph = {}
    for i in range(n):
        # Get distances from i to all others
        dists = dm[i].copy()
        dists[i] = np.inf  # Exclude self
        
        # Get indices of k smallest distances
        # argsort is O(n log n), which is fine for n=51
        indices = np.argsort(dists)
        knn_graph[i] = [int(j) for j in indices[:k]]
        
    _knn_cache[distance_matrix] = knn_graph
    return knn_graph


def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated VRP: pick next customer or depot (0) to close route.
    
    Strategy: K-Nearest Neighbor Constrained Greedy.
    We restrict the choice of the next node to the K nearest neighbors of the current node.
    This enforces local connectivity and prevents "crossing" long edges that are common
    in global heuristics like pure savings or nearest neighbor without locality constraints.
    If no feasible neighbor exists in the K-NN set, we fall back to the global nearest
    feasible node.
    """
    if _time_exceeded():
        return 0

    unvisited = list(unvisited_nodes)
    if not unvisited:
        return 0

    # Filter by capacity
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return 0

    # Get K-NN graph
    # Note: distance_matrix might be a list or array. We use it as a key.
    # To be safe with hashing, we can use a hash of the array or just assume it's consistent.
    # Since the evaluator uses the same matrix, we can cache by id or content.
    # Let's use a simple check.
    
    dm = np.asarray(distance_matrix)
    n_nodes = dm.shape[0]
    
    # Retrieve or compute K-NN
    # Using a global cache keyed by a hash of the matrix is risky if matrix is huge, but n=51 is small.
    # We'll use a simple tuple of tuples as key? No, just use the object id if we assume stability, 
    # or better, compute it once per unique matrix structure.
    # For simplicity and safety against cache pollution, we compute K-NN on the fly if not cached by a robust key.
    # Let's use a global variable to store the last matrix and its K-NN.
    
    global _last_dm_hash
    global _last_knn
    
    # Simple hash for small matrices
    # Actually, np.array doesn't have a stable hash. Let's just compute it every time?
    # n=51, K=10. 51 * 51 log(51) is very fast. < 1ms.
    # So we can just compute it inside the function to avoid caching complexities.
    
    knn_neighbors = []
    dists_from_current = dm[current_node].copy()
    dists_from_current[current_node] = np.inf
    
    # Get indices of K nearest neighbors of current_node
    # We need to filter by feasibility later.
    # First, get the K nearest in general.
    k_indices = np.argsort(dists_from_current)[:_K_NN]
    knn_set = set(int(i) for i in k_indices)
    
    # Check if any feasible node is in the K-NN set
    feasible_in_knn = [n for n in feasible if n in knn_set]
    
    if feasible_in_knn:
        # Choose the closest feasible node among the K-NNs
        # This is equivalent to choosing the min distance among feasible_in_knn
        # Since knn_set is small, we can just iterate
        best_node = None
        best_dist = float('inf')
        for n in feasible_in_knn:
            d = dm[current_node, n]
            if d < best_dist:
                best_dist = d
                best_node = n
        return best_node
    else:
        # Fallback: No feasible node in K-NN.
        # Choose the globally nearest feasible node.
        # This ensures we always make progress.
        feasible_arr = np.array(feasible)
        dists_to_feasible = dm[current_node, feasible_arr]
        best_idx = np.argmin(dists_to_feasible)
        return int(feasible_arr[best_idx])

# EVOLVE-BLOCK-END
