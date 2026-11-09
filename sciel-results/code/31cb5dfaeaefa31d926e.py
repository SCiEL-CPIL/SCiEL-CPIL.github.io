"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Explicit deadline guard
_DEADLINE_FRACTION = 0.9
_WALL_LIMIT = 330.0
_start = time.time()

def _deadline_left():
    return _WALL_LIMIT * _DEADLINE_FRACTION - (time.time() - _start)

def _nearest_neighbor_tour(d, start=0):
    """
    Construct a tour using the Nearest Neighbor heuristic starting from a specific node.
    """
    n = d.shape[0]
    visited = np.zeros(n, dtype=bool)
    order = [start]
    visited[start] = True
    cur = start
    for _ in range(n - 1):
        dists = d[cur].copy()
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        order.append(nxt)
        visited[nxt] = True
        cur = nxt
    return order

def _two_opt(d, order, max_iters=50):
    """
    Improve a tour using the 2-opt heuristic.
    Returns the improved order and its length.
    """
    n = len(order)
    if n < 4:
        return order, float(sum(d[order[i], order[(i + 1) % n]] for i in range(n)))
    
    # Build a distance lookup via the matrix using node indices
    # order is list of node indices; closed tour order -> order -> order[0]
    def tour_len(o):
        return float(sum(d[o[i], o[(i + 1) % n]] for i in range(n)))

    best = list(order)
    best_len = tour_len(best)
    
    # Optimization: Precompute delta improvements? No, simple loop is fine for n=50.
    # Standard 2-opt: reverse segment between i and j.
    # Edges removed: (best[i-1], best[i]) and (best[j], best[(j+1)%n])
    # Edges added: (best[i-1], best[j]) and (best[i], best[(j+1)%n])
    
    for _ in range(max_iters):
        if _deadline_left() <= 0:
            break
        improved = False
        # i is the start of the segment to reverse (index in order)
        # We iterate i from 1 to n-2 (0-based indices in the list)
        # and j from i+1 to n-1.
        # Note: In a cyclic tour, reversing [i..j] changes edges (i-1, i) and (j, j+1) to (i-1, j) and (i, j+1).
        # Indices are modulo n.
        
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Skip if adjacent (which would be a no-op or 3-opt)
                if j == i + 1:
                    continue
                if i == 1 and j == n - 1:
                    continue
                
                a = best[i - 1]
                b = best[i]
                c = best[j]
                d_idx = best[(j + 1) % n]
                
                # Current cost: d[a,b] + d[c,d]
                # New cost: d[a,c] + d[b,d]
                delta = (d[a, c] + d[b, d_idx]) - (d[a, b] + d[c, d_idx])
                
                if delta < -1e-9:
                    # Reverse segment i to j
                    best[i:j + 1] = best[i:j + 1][::-1]
                    best_len += delta
                    improved = True
        
        if not improved:
            break
            
    return best, best_len

def _good_directed_tour(distance_matrix):
    """
    Find a good directed tour by trying multiple starting nodes for NN + 2-opt.
    """
    n = distance_matrix.shape[0]
    best_tour = None
    best_len = np.inf
    
    # Try a few starting nodes (0, 1, 2, and maybe a random one if time permits)
    starts = [0, 1, 2]
    if n > 3:
        starts.append(n // 2)
    if n > 5:
        starts.append(n - 1)
        
    for start in starts:
        if _deadline_left() <= 0:
            break
        nn_tour = _nearest_neighbor_tour(distance_matrix, start)
        opt_tour, opt_len = _two_opt(distance_matrix, nn_tour, max_iters=50)
        if opt_len < best_len:
            best_len = opt_len
            best_tour = opt_tour
            
    return best_tour

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for the ACO solver.
    Uses a combination of inverse distance and a boost for edges
    present in a near-optimal tour found via NN + 2-opt.
    """
    dist = np.array(distance_matrix, dtype=np.float64)
    n = dist.shape[0]
    
    # Ensure positive distances
    dist = np.maximum(dist, 1e-9)
    
    # Base heuristic: inverse distance
    # Add epsilon to avoid division by zero if any diagonal is 0
    d_safe = np.where(dist > 1e-9, dist, 1.0)
    base_heuristic = 1.0 / d_safe
    
    # Ensure diagonal is 0 or very small, as ants never pick themselves
    np.fill_diagonal(base_heuristic, 0.0)
    
    # Find a reference tour
    ref_tour = _good_directed_tour(dist)
    
    if ref_tour is None:
        ref_tour = _nearest_neighbor_tour(dist)
        
    # Create a mask for edges in the reference tour
    # Note: The tour is a sequence of nodes. Edges are (tour[i], tour[i+1])
    # Since the tour is closed, edges are (tour[i], tour[(i+1)%n])
    in_tour = np.zeros((n, n), dtype=float)
    for i in range(n):
        u = ref_tour[i]
        v = ref_tour[(i + 1) % n]
        in_tour[u, v] = 1.0
        # Distance is symmetric, but heuristic is used in directed probability?
        # The task says "probability proportional to pheromone[i, j] * heuristic[i, j]"
        # and "pheromone[i, j] ... for each step i to j".
        # Usually ACO is directed, but TSP is symmetric.
        # The distance matrix is likely symmetric.
        # If the heuristic is symmetric, it's fine.
        in_tour[v, u] = 1.0
        
    # Boost edges in the reference tour
    # A multiplier of 5-10 is typically effective to guide without completely restricting
    # The inspiration used 5.0, let's stick with that or slightly higher for stronger guidance
    boost_factor = 5.0
    heuristic = base_heuristic * (1.0 + boost_factor * in_tour)
    
    # The task states: "Heuristic entries that are not positive become 1e-9"
    # Our heuristic is positive everywhere except diagonal (0).
    # We should set diagonal to 1e-9 or just leave it? 
    # "Heuristic entries that are not positive become 1e-9"
    # 0 is not positive. So diagonal should be 1e-9.
    heuristic[heuristic <= 0] = 1e-9
    
    # Normalize to prevent numerical issues? 
    # The inspiration did not normalize, but the current program did.
    # Normalizing might be good to keep values in a reasonable range.
    # However, if we normalize, we might lose the relative scaling between boosted and non-boosted edges if not careful.
    # Let's check the magnitude. 1/dist is around 1-10. Boosted is 6-60.
    # Max is around 60. 1/60 is small. 
    # The ACO formula uses pheromone * heuristic. Pheromone starts at 1.
    # If heuristic is ~1, probabilities are proportional to 1.
    # If heuristic is ~10, probabilities are proportional to 10.
    # This seems fine. Normalizing to max=1 would make non-boosted edges very small.
    # Let's try without normalization first, or with a softer normalization.
    # Actually, the current program (score 0.39) normalized.
    # The inspiration (score 1.43) did NOT normalize.
    # This suggests normalization might be harmful or the boost factor needs tuning.
    # Let's follow the inspiration's lead: no normalization, just ensure positivity.
    
    return heuristic
# EVOLVE-BLOCK-END
