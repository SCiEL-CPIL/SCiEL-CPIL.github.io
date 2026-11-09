import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np
import time

def _two_opt(tour, d):
    """
    Optimizes a tour using 2-opt.
    tour: list of node indices, length n (no repeat of start at end).
    d: distance matrix.
    Returns: optimized tour list.
    """
    n = len(tour)
    if n <= 2:
        return tour
    
    t = tour[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                u, v = t[i-1], t[i]
                x, y = t[j], t[(j+1) % n]
                
                if u == x or u == y or v == x or v == y:
                    continue
                
                old_cost = d[u, v] + d[x, y]
                new_cost = d[u, x] + d[v, y]
                
                if new_cost < old_cost:
                    t[i:j+1] = t[i:j+1][::-1]
                    improved = True
                    
    return t

def _nearest_neighbor_tour(d, start=0):
    n = d.shape[0]
    visited = [False] * n
    visited[start] = True
    tour = [start]
    current = start
    
    for _ in range(n - 1):
        # Find nearest unvisited
        min_dist = float('inf')
        next_node = -1
        for j in range(n):
            if not visited[j] and d[current, j] < min_dist:
                min_dist = d[current, j]
                next_node = j
        
        if next_node == -1:
            break
        
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    return tour

def _evaluate_heuristic(d, h, deadline):
    """
    Evaluates a heuristic by simulating the ACO process briefly
    and comparing the best tour found (after 2-opt) to the
    quality of the heuristic itself.
    
    Actually, for speed and reliability, we use a proxy:
    Run a modified nearest-neighbor where the 'cost' is 1/h[i][j]
    (since ACO picks prob prop to pheromone * heuristic, and pheromone starts at 1,
    initial bias is strongly heuristic). Then run 2-opt on that tour using original d.
    If this yields a short tour, the heuristic is good.
    """
    n = d.shape[0]
    if time.time() > deadline:
        return float('inf')
    
    # Effective distance for construction: inverse heuristic
    # h might have zeros or negatives, clamp to small positive
    h_safe = np.maximum(h, 1e-9)
    
    # We want to pick next node with high h.
    # Let's do a greedy construction based on h.
    # Start at 0.
    visited = [False] * n
    visited[0] = True
    tour = [0]
    current = 0
    
    for _ in range(n - 1):
        # Pick unvisited j with max h[current, j]
        # But we also want to avoid bad edges? 
        # The ACO uses pheromone * heuristic. Initially pheromone is 1.
        # So it's purely heuristic driven initially.
        # Let's try max h[current, j] among unvisited.
        
        max_h = -1
        next_node = -1
        for j in range(n):
            if not visited[j] and h_safe[current, j] > max_h:
                max_h = h_safe[current, j]
                next_node = j
        
        if next_node == -1:
            break
        
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    # Run 2-opt on the resulting tour using original distance d
    if len(tour) == n:
        optimized_tour = _two_opt(tour, d)
        # Calculate length
        length = 0.0
        for i in range(n):
            length += d[optimized_tour[i], optimized_tour[(i+1) % n]]
        return length
    else:
        return float('inf')

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin per instance
    
    d = distance_matrix.copy()
    n = d.shape[0]
    
    # Ensure diagonal is not used for heuristic selection logic, 
    # though the matrix passed to ACO will have diagonal entries.
    # In ACO, diagonal is ignored as nodes are unvisited.
    
    candidates = []
    
    # 1. Inverse distance variants
    candidates.append(1.0 / (d + 1e-9))
    candidates.append(1.0 / (d**1.5 + 1e-9))
    candidates.append(1.0 / (d**2 + 1e-9))
    candidates.append(1.0 / (d**4 + 1e-9))
    
    # 2. Offset variants
    candidates.append(1.0 / (d + 0.05))
    candidates.append(1.0 / (d + 0.1))
    candidates.append(1.0 / (d + 0.2))
    
    # 3. Exponential decay
    candidates.append(np.exp(-d))
    
    # 4. Neighbor-based boosts
    h_base = 1.0 / (d + 1e-9)
    
    # Top 3
    h_top3 = h_base.copy()
    for i in range(n):
        row = d[i].copy()
        row[i] = np.inf
        top3_idx = np.argpartition(row, 3)[:3]
        h_top3[i, top3_idx] *= 10.0
    candidates.append(h_top3)
    
    # Top 5
    h_top5 = h_base.copy()
    for i in range(n):
        row = d[i].copy()
        row[i] = np.inf
        top5_idx = np.argpartition(row, 5)[:5]
        h_top5[i, top5_idx] *= 5.0
    candidates.append(h_top5)
    
    # Top 1 (Strict NN)
    h_top1 = h_base.copy()
    for i in range(n):
        row = d[i].copy()
        row[i] = np.inf
        top1_idx = np.argmin(row)
        h_top1[i, top1_idx] *= 20.0
    candidates.append(h_top1)
    
    # 5. Angle-like proxy or geometric considerations?
    # Try a mix: high weight for short edges, but penalize "sharp turns"?
    # Without coords, we can't do angles easily.
    
    # 6. Randomized variants? No, deterministic is better for selection.
    
    best_heuristic = candidates[0]
    best_score = float('inf')
    
    for idx, h in enumerate(candidates):
        # Check time
        if time.time() > deadline:
            break
        
        # Evaluate candidate
        score = _evaluate_heuristic(d, h, deadline)
        
        if score < best_score:
            best_score = score
            best_heuristic = h
    
    return best_heuristic
# EVOLVE-BLOCK-END
