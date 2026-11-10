"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _nearest_neighbor_tour(d, start):
    """Construct a tour using the Nearest Neighbor heuristic starting from 'start'."""
    n = len(d)
    visited = np.zeros(n, dtype=bool)
    tour = np.zeros(n, dtype=int)
    tour[0] = start
    visited[start] = True
    current = start
    for i in range(1, n):
        dists = d[current].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour[i] = next_node
        visited[next_node] = True
        current = next_node
    return tour

def _two_opt(d, tour, max_passes=15):
    """Improve a tour using 2-opt local search."""
    n = len(tour)
    tour = tour.copy()
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            a = tour[i]
            b = tour[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c = tour[j]
                d_node = tour[j + 1] if j + 1 < n else tour[0]
                delta = d[a, c] + d[b, d_node] - d[a, b] - d[c, d_node]
                if delta < -1e-10:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
        if not improved:
            break
    return tour

def _or_opt(d, tour, max_passes=10):
    """Improve a tour using Or-Opt (move 1 or 2 nodes) local search."""
    n = len(tour)
    tour = tour.copy()
    for _ in range(max_passes):
        improved = False
        # Move 1 node
        for i in range(n):
            if i == 0:
                continue # Don't move start node in standard TSP tour representation if we fix it, 
                         # but since it's a cycle, we can move any. 
                         # However, ACO starts at 0. It's safer to allow moves but ensure 
                         # we don't break the cycle logic. 
                         # Let's just allow moving any segment of length 1 or 2.
            
            node = tour[i]
            prev = tour[i-1]
            next_ = tour[(i+1)%n]
            
            # Calculate cost of removing node
            cost_remove = d[prev, next_] - d[prev, node] - d[node, next_]
            
            # Try inserting at every other position
            best_gain = cost_remove
            best_pos = -1
            
            for j in range(n):
                if j == i or j == (i-1)%n or j == (i+1)%n:
                    continue
                p_prev = tour[(j-1)%n]
                p_next = tour[j]
                
                # Cost of inserting node between p_prev and p_next
                cost_insert = d[p_prev, node] + d[node, p_next] - d[p_prev, p_next]
                
                # Net change = cost_insert - (cost of current state which is already accounted 
                # in the 'remove' logic if we think about it as a swap)
                # Actually, simpler: 
                # Current segment: prev - node - next_
                # New segment: prev - next_ (node removed)
                # Insertion: p_prev - node - p_next (node inserted)
                # Old: d[prev, node] + d[node, next_] + d[p_prev, p_next]
                # New: d[prev, next_] + d[p_prev, node] + d[node, p_next]
                # Delta = New - Old
                delta = d[prev, next_] + d[p_prev, node] + d[node, p_next] - (d[prev, node] + d[node, next_] + d[p_prev, p_next])
                
                if delta < best_gain - 1e-10:
                    best_gain = delta
                    best_pos = j
            
            if best_pos != -1:
                # Perform move
                tour = np.delete(tour, i)
                # Insert at best_pos in the new array
                # Note: indices shift after delete
                tour = np.insert(tour, best_pos, node)
                improved = True
        
        # Move 2 nodes (segment)
        for i in range(n):
            if i == 0:
                continue
            n1 = tour[i]
            n2 = tour[(i+1)%n]
            prev = tour[i-1]
            next_ = tour[(i+2)%n]
            
            # Remove segment n1-n2
            # Old: prev - n1 - n2 - next_
            # New: prev - next_
            # Insert n1-n2 between p_prev and p_next
            best_gain = np.inf
            best_pos = -1
            
            for j in range(n):
                if j == i or j == (i+1)%n or j == (i-1)%n or j == (i+2)%n:
                    continue
                p_prev = tour[(j-1)%n]
                p_next = tour[j]
                
                # Old: d[prev, n1] + d[n1, n2] + d[n2, next_] + d[p_prev, p_next]
                # New: d[prev, next_] + d[p_prev, n1] + d[n1, n2] + d[n2, p_next]
                delta = d[prev, next_] + d[p_prev, n1] + d[n1, n2] + d[n2, p_next] - \
                        (d[prev, n1] + d[n1, n2] + d[n2, next_] + d[p_prev, p_next])
                
                if delta < best_gain - 1e-10:
                    best_gain = delta
                    best_pos = j
            
            if best_pos != -1:
                # Remove n1, n2
                tour = np.delete(tour, [i, (i+1)%n])
                # Insert n1, n2 at best_pos
                tour = np.insert(tour, best_pos, [n1, n2])
                improved = True
        
        if not improved:
            break
    return tour

def _compute_consensus_heuristic(d, deadline):
    """
    Compute a heuristic matrix based on the frequency of edges appearing in 
    multiple local optima found via multi-start NN + 2-opt + Or-Opt.
    """
    n = len(d)
    
    # Edge frequency counter
    edge_freq = np.zeros((n, n), dtype=np.float64)
    
    # Number of diverse starts to sample
    # We want to sample enough to get a good consensus but fast enough to finish.
    # n=50, 2-opt is O(n^2) per pass. 
    # Let's aim for ~20-30 diverse tours if time permits.
    num_starts = 30
    starts = np.random.choice(n, size=num_starts, replace=False)
    # Ensure 0 is included for consistency, though symmetry makes it less critical
    if 0 not in starts:
        starts[0] = 0
        
    start_time = time.time()
    
    for start in starts:
        if time.time() > deadline:
            break
            
        # 1. NN Tour
        tour = _nearest_neighbor_tour(d, start)
        
        # 2. 2-Opt
        tour = _two_opt(d, tour, max_passes=10)
        
        # 3. Or-Opt
        tour = _or_opt(d, tour, max_passes=5)
        
        # 4. Record edges
        for i in range(n):
            u = tour[i]
            v = tour[(i+1) % n]
            edge_freq[u, v] += 1.0
            edge_freq[v, u] += 1.0

    # If no tours were computed (timeout immediately?), fallback
    if edge_freq.sum() == 0:
        return 1.0 / (d + 1e-9)

    # Normalize frequencies
    # Max frequency is num_starts * 2 (since we add to both directions)
    # But different tours might share edges.
    # We want to map frequency to attractiveness.
    # High frequency -> High attractiveness.
    # Low frequency -> Low attractiveness (but keep a base inverse distance).
    
    # Strategy: 
    # H = alpha * (freq / max_freq) + beta * (1/d)
    # This ensures that edges that appear in many good tours are favored, 
    # but very short edges that are rare are still somewhat attractive.
    
    max_freq = np.max(edge_freq)
    if max_freq == 0:
        max_freq = 1.0
        
    freq_norm = edge_freq / max_freq
    
    # Inverse distance component
    inv_dist = 1.0 / (d + 1e-9)
    # Normalize inv_dist to [0, 1] roughly for mixing
    max_inv = np.max(inv_dist)
    if max_inv > 0:
        inv_dist_norm = inv_dist / max_inv
    else:
        inv_dist_norm = inv_dist

    # Blend: 70% consensus, 30% distance
    # This prevents the heuristic from being purely binary (in tour / not in tour)
    # which can cause premature convergence.
    H = 0.7 * freq_norm + 0.3 * inv_dist_norm
    
    # Scale up to make it dominant over initial pheromone (1.0)
    # If H is around 0.1-1.0, and pheromone is 1.0, the ratio is small.
    # We want edges with high freq to have H >> 1.
    # Let's scale so max H is around 10-20.
    H = H * 10.0
    
    # Ensure positive
    H[H <= 0] = 1e-9
    H[~np.isfinite(H)] = 1e-9
    
    return H

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO of TSP.
    Larger values indicate more attractive edges.
    """
    start_time = time.time()
    deadline = start_time + 280  # Leave 50s margin for the 330s limit
    
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = len(d)
    
    # Validate input
    if d.shape != (n, n) or not np.all(np.isfinite(d)):
        return np.ones((n, n)) * 1e-9
    
    try:
        H = _compute_consensus_heuristic(d, deadline)
    except Exception:
        # Fallback to simple inverse distance
        H = 1.0 / (d + 1e-9)
        H[~np.isfinite(H)] = 1e-9
        H[H <= 0] = 1e-9
    
    # Final validation
    if H.shape != (n, n) or not np.all(np.isfinite(H)):
        H = 1.0 / (d + 1e-9)
        H[~np.isfinite(H)] = 1e-9
        H[H <= 0] = 1e-9
    
    return H
# EVOLVE-BLOCK-END
