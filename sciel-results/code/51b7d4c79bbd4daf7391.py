"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed data per distance_matrix
_cache = {}

def _get_cache_key(dm):
    # Use a hash of the first few elements and shape to identify the matrix
    # Since dm is passed each time, we assume it's the same instance for a given run
    # We can use id() if we assume the object persists, but safer to use a content-based key
    # For 50 cities, hashing the whole matrix is fast enough (<1ms)
    try:
        return hash((dm.shape, dm.tobytes()))
    except:
        return None

def _precompute(dm):
    """Precompute useful data for the distance matrix."""
    n = dm.shape[0]
    # Sort nodes by distance from each node? No, that's n^2 log n.
    # Instead, let's just keep the matrix.
    # Maybe precompute sorted indices for each row?
    # sorted_indices[i] = indices of all nodes sorted by distance from i
    sorted_indices = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        sorted_indices[i] = np.argsort(dm[i])
    
    return {
        'n': n,
        'sorted_indices': sorted_indices
    }

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Deadline guard
    start_time = time.time()
    
    # Try to get cached data
    key = _get_cache_key(distance_matrix)
    if key is not None and key in _cache:
        data = _cache[key]
    else:
        data = _precompute(distance_matrix)
        if key is not None:
            _cache[key] = data
    
    n = data['n']
    sorted_indices = data['sorted_indices']
    
    if len(unvisited_nodes) == 0:
        return -1 # Should not happen based on contract, but safe
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
        
    # Heuristic: 
    # 1. Immediate distance from current
    # 2. Distance from candidate to destination (to avoid stranding)
    # 3. Maybe "regret" or "density" of unvisited nodes near candidate
    
    # Let's use a simple weighted sum:
    # score = d(current, cand) + alpha * d(cand, destination)
    # This is a 2-opt style lookahead or just a penalty for being far from the end.
    
    # We can optimize the lookup using the sorted_indices if needed, 
    # but for 50 nodes, direct indexing is fast.
    
    # Vectorized scoring
    # d_curr = distance_matrix[current_node, unvisited_nodes]
    # d_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Let's add a small lookahead: 
    # For each candidate, estimate the cost of the next step.
    # min_next = min_{u in unvisited, u != cand} d(cand, u)
    # This is O(N^2) per step if done naively. 
    # With N=50, total steps 50, total ops 50*50*50 = 125k, which is fine.
    
    # Let's try: score = d(current, cand) + 0.5 * min(d(cand, unvisited\cand)) + 0.5 * d(cand, destination)
    # Or simpler: score = d(current, cand) + d(cand, destination)
    
    # Let's test the "insertion" heuristic idea:
    # Pick the node that minimizes the increase in tour length if inserted?
    # No, we are building sequentially.
    
    # Let's try a "nearest neighbor with destination bias".
    # score = d(current, cand) + beta * d(cand, destination)
    # beta = 0.5
    
    # To be safe and fast:
    d_curr = distance_matrix[current_node, unvisited_nodes]
    d_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Weighted sum
    alpha = 0.5
    scores = d_curr + alpha * d_dest
    
    # Add a small tie-breaker or noise? No, deterministic is better for reproducibility.
    
    # Check time
    if time.time() - start_time > 0.5:
         # Fallback to simple nearest neighbor
         return int(unvisited_nodes[np.argmin(d_curr)])
    
    min_idx = np.argmin(scores)
    return int(unvisited_nodes[min_idx])
# EVOLVE-BLOCK-END
