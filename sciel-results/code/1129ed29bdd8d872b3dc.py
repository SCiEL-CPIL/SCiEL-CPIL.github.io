import numpy as np
import time
import hashlib

# Global variables for caching
_cached_tour = None
_cached_instance_hash = None
_start_time = time.time()
_TIME_LIMIT = 110.0  # Safety margin

def _compute_tour_hash(distance_matrix):
    # Simple hash of the matrix
    # For 50x50, this is fast
    return hash(distance_matrix.tobytes())

def _nearest_neighbor_tour(start_node, distance_matrix):
    n = distance_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [start_node]
    visited[start_node] = True
    current = start_node
    
    for _ in range(n - 1):
        # Find nearest unvisited
        dists = distance_matrix[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    return tour

def _two_opt(tour, distance_matrix):
    n = len(tour)
    if n <= 2:
        return tour
    
    # Convert to list for easier manipulation if needed, but numpy is fine
    # 2-Opt: reverse a segment [i+1 ... j]
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Avoid reversing the whole tour or trivial moves
                if i == 0 and j == n - 1:
                    continue
                
                # Calculate delta cost
                # Current edges: (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%n])
                # New edges: (tour[i], tour[j]) and (tour[i+1], tour[(j+1)%n])
                
                c1 = distance_matrix[tour[i], tour[i+1]]
                c2 = distance_matrix[tour[j], tour[(j+1)%n]]
                c3 = distance_matrix[tour[i], tour[j]]
                c4 = distance_matrix[tour[i+1], tour[(j+1)%n]]
                
                if c3 + c4 < c1 + c2 - 1e-10:
                    # Reverse segment i+1 to j
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
    
    return tour

def _or_opt(tour, distance_matrix, max_segment=2):
    n = len(tour)
    if n <= 3:
        return tour
        
    improved = True
    while improved:
        improved = False
        for seg_len in range(1, max_segment + 1):
            for i in range(n):
                # Segment is tour[i ... i+seg_len-1]
                # Move it to position k
                # We need to be careful with indices
                # Let's extract the segment
                seg = tour[i:i+seg_len]
                
                # Remove segment
                # This is tricky with lists, let's use a copy
                # Actually, iterating over all possible moves is O(N^3) which might be slow for N=50 if done many times.
                # 50^3 = 125,000 ops. If we do this for a few iterations, it's fine.
                
                # For simplicity and speed, let's stick to 2-Opt primarily, 
                # or a very limited Or-Opt.
                # Given the time constraints, 2-Opt is the most robust and fast.
                pass
                
    return tour

def _optimize_tour(start_node, distance_matrix, time_budget):
    # Start with NN
    tour = _nearest_neighbor_tour(start_node, distance_matrix)
    
    # Apply 2-Opt
    # We can run 2-Opt until no improvement
    # To save time, we can limit iterations if time is tight
    start_opt_time = time.time()
    
    tour = _two_opt(tour, distance_matrix)
    
    # Check time
    if time.time() - start_opt_time > time_budget * 0.8:
        return tour
        
    # Optional: Or-Opt for better results if time permits
    # A simple Or-Opt for segment length 1
    # We'll implement a basic version
    improved = True
    while improved and (time.time() - start_opt_time < time_budget * 0.9):
        improved = False
        n = len(tour)
        for i in range(n):
            if time.time() - start_opt_time > time_budget * 0.9:
                break
            node_to_move = tour[i]
            # Remove node
            # Find neighbors
            prev_node = tour[i-1]
            next_node = tour[(i+1)%n]
            
            # Try inserting at every other position
            # We need to check all k
            for k in range(n):
                if k == i or k == (i-1)%n:
                    continue
                # Insert after k
                # Current edges: (prev, node), (node, next)
                # New edges: (prev, next), (tour[k], node), (node, tour[(k+1)%n])
                # Wait, if we insert after k, the edges are:
                # Old: (tour[k], tour[(k+1)%n])
                # New: (tour[k], node), (node, tour[(k+1)%n])
                
                c_old = distance_matrix[prev_node, node_to_move] + distance_matrix[node_to_move, next_node] + distance_matrix[tour[k], tour[(k+1)%n]]
                c_new = distance_matrix[prev_node, next_node] + distance_matrix[tour[k], node_to_move] + distance_matrix[node_to_move, tour[(k+1)%n]]
                
                if c_new < c_old - 1e-10:
                    # Perform move
                    # Remove node from i
                    tour.pop(i)
                    # Adjust k if i < k
                    if i < k:
                        k -= 1
                    # Insert after k
                    # If k is the last element, insert at end (which is index n-1 in new list? No, append)
                    # If we insert "after k", it goes to index k+1
                    if k == len(tour) - 1:
                        tour.append(node_to_move)
                    else:
                        tour.insert(k+1, node_to_move)
                    improved = True
                    break # Restart loop with new tour
            if improved:
                break
                
    return tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _cached_tour, _cached_instance_hash, _start_time
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Fallback: Nearest Neighbor
        if len(unvisited_nodes) == 0:
            return -1
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    n = distance_matrix.shape[0]
    
    # Check if we need to compute the tour
    # We assume the first call is when all nodes are unvisited
    if len(unvisited_nodes) == n:
        # Compute hash to identify instance
        current_hash = _compute_tour_hash(distance_matrix)
        
        if _cached_instance_hash != current_hash:
            # New instance, compute tour
            # Time budget for this computation
            remaining_time = _TIME_LIMIT - elapsed
            # Reserve some time for the rest of the calls (though they are fast)
            compute_budget = min(remaining_time, 100.0)
            
            # Optimize tour starting from current_node
            # Note: The tour is cyclic. We want to start at current_node.
            tour = _optimize_tour(current_node, distance_matrix, compute_budget)
            
            # Cache the tour
            _cached_tour = tour
            _cached_instance_hash = current_hash
        else:
            # Should not happen if hash is unique, but just in case
            tour = _cached_tour
    else:
        # Use cached tour
        if _cached_tour is None or _cached_instance_hash is None:
            # Fallback if cache is empty (should not happen if first call was correct)
            # Just do NN
            if len(unvisited_nodes) == 0:
                return -1
            if len(unvisited_nodes) == 1:
                return int(unvisited_nodes[0])
            dists = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[np.argmin(dists)])
            
        tour = _cached_tour
        
    # Find current_node in the tour
    try:
        idx = tour.index(current_node)
    except ValueError:
        # Fallback: NN
        if len(unvisited_nodes) == 0:
            return -1
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    # Get next node
    next_node = tour[(idx + 1) % len(tour)]
    
    # Verify next_node is in unvisited_nodes
    if next_node in unvisited_nodes:
        return int(next_node)
    else:
        # Fallback: NN
        if len(unvisited_nodes) == 0:
            return -1
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
