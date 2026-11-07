import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _nearest_neighbor_tour(d, start=0):
    n = d.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    current = start
    tour = [start]
    for _ in range(n - 1):
        d_row = d[current].copy()
        d_row[visited] = np.inf
        d_row[current] = np.inf
        next_node = np.argmin(d_row)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour

def _two_opt(tour, d):
    n = len(tour)
    if n <= 2:
        return tour
    
    def tour_length(t):
        return sum(d[t[i], t[(i + 1) % n]] for i in range(n))

    best_tour = tour[:]
    best_len = tour_length(best_tour)
    
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                u1 = best_tour[i - 1]
                u2 = best_tour[i]
                u3 = best_tour[j]
                u4 = best_tour[(j + 1) % n]
                
                old_len = d[u1, u2] + d[u3, u4]
                new_len = d[u1, u3] + d[u2, u4]
                
                if new_len < old_len - 1e-10:
                    best_tour = best_tour[:]
                    best_tour[i:j + 1] = best_tour[i:j + 1][::-1]
                    best_len = tour_length(best_tour)
                    improved = True
    return best_tour

def _mst_edges(d):
    """Compute MST edges using Prim's algorithm."""
    n = d.shape[0]
    in_mst = [False] * n
    min_dist = [float('inf')] * n
    min_dist[0] = 0
    parent = [-1] * n
    edges = []
    
    for _ in range(n):
        # Find unvisited node with minimum distance
        u = -1
        for i in range(n):
            if not in_mst[i] and (u == -1 or min_dist[i] < min_dist[u]):
                u = i
        in_mst[u] = True
        if parent[u] != -1:
            edges.append((parent[u], u))
        for v in range(n):
            if not in_mst[v] and d[u, v] < min_dist[v]:
                min_dist[v] = d[u, v]
                parent[v] = u
    return edges

def _mwpm_edges(d):
    """Compute Minimum Weight Perfect Matching edges."""
    n = d.shape[0]
    # For even n, find perfect matching
    if n % 2 != 0:
        return []
    
    # Create cost matrix for matching
    # We need to find a perfect matching in the complete graph
    # This is equivalent to finding minimum weight perfect matching
    # Use scipy's linear_sum_assignment on a modified matrix
    
    # For TSP, we want each node to have degree 2 in the tour
    # MWPM gives us a 1-factor (each node degree 1)
    # We'll compute it on the distance matrix
    
    row_ind, col_ind = linear_sum_assignment(d)
    edges = []
    for r, c in zip(row_ind, col_ind):
        if r != c:
            edges.append((r, c))
    return edges

def _quick_aco_eval(d, eta, n_ants=15, n_iter=8, seed=123):
    """Fast ACO simulation to score heuristic quality with proper pheromone updates."""
    rng = np.random.RandomState(seed)
    n = d.shape[0]
    tau = np.ones((n, n))
    best_len = float('inf')
    
    for _ in range(n_iter):
        all_tours = []
        all_lens = []
        for _ in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            current = 0
            visited[0] = True
            tour_len = 0.0
            tour = [0]
            
            for _ in range(n - 1):
                probs = tau[current, :] * eta[current, :]
                probs[visited] = 0.0
                probs[current] = 0.0
                total = np.sum(probs)
                if total <= 1e-15:
                    unvisited = np.where(~visited)[0]
                    next_node = unvisited[rng.randint(len(unvisited))]
                else:
                    cum_probs = np.cumsum(probs) / total
                    r = rng.rand()
                    next_node = np.searchsorted(cum_probs, r)
                    if next_node >= n: next_node = n - 1
                tour_len += d[current, next_node]
                tour.append(next_node)
                current = next_node
                visited[current] = True
            tour_len += d[current, 0]
            all_tours.append(tour)
            all_lens.append(tour_len)
            if tour_len < best_len:
                best_len = tour_len
        
        # Pheromone update
        tau *= 0.9
        for tour, L in zip(all_tours, all_lens):
            if L < 1e-15: continue
            delta = 1.0 / L
            for k in range(n):
                u = tour[k]
                v = tour[(k + 1) % n]
                tau[u, v] += delta
                tau[v, u] += delta
    
    return best_len

def heuristics(distance_matrix):
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    mask = np.eye(n, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    # 1. Find a high quality tour
    best_tour = None
    best_tour_len = float('inf')
    
    # Use multiple restarts for NN + 2-Opt
    for i in range(30):
        if time.time() > deadline - 60.0: break
        start_node = i % n
        tour = _nearest_neighbor_tour(d, start=start_node)
        tour = _two_opt(tour, d)
        length = sum(d[tour[k], tour[(k + 1) % n]] for k in range(n))
        if length < best_tour_len:
            best_tour_len = length
            best_tour = tour[:]
            
    if best_tour is None:
        best_tour = _nearest_neighbor_tour(d, start=0)
        best_tour = _two_opt(best_tour, d)
        best_tour_len = sum(d[best_tour[k], best_tour[(k + 1) % n]] for k in range(n))

    # 2. Compute structural edges
    mst_e = _mst_edges(d)
    mwpm_e = _mwpm_edges(d)
    
    # Create edge indicator arrays
    tour_edges = np.zeros((n, n), dtype=bool)
    mst_edge_arr = np.zeros((n, n), dtype=bool)
    mwpm_edge_arr = np.zeros((n, n), dtype=bool)
    
    for k in range(n):
        u = best_tour[k]
        v = best_tour[(k + 1) % n]
        tour_edges[u, v] = True
        tour_edges[v, u] = True
    
    for u, v in mst_e:
        mst_edge_arr[u, v] = True
        mst_edge_arr[v, u] = True
    
    for u, v in mwpm_e:
        mwpm_edge_arr[u, v] = True
        mwpm_edge_arr[v, u] = True
    
    # 3. Generate candidate heuristics
    # Base: eta = 1 / d^p
    # Boost tour edges, MST edges, MWPM edges with smooth weights
    
    d_safe = d.copy()
    d_safe[mask] = 1.0
    
    candidates = []
    
    # Parameters: (p, tour_boost, mst_boost, mwpm_boost)
    params = [
        (1.5, 5.0, 2.0, 3.0),
        (2.0, 5.0, 2.0, 3.0),
        (2.0, 8.0, 3.0, 4.0),
        (2.5, 5.0, 2.0, 3.0),
        (2.5, 8.0, 3.0, 4.0),
        (3.0, 5.0, 2.0, 3.0),
        (3.0, 8.0, 3.0, 4.0),
        (2.0, 10.0, 4.0, 5.0),
        (2.5, 10.0, 4.0, 5.0),
        (2.0, 1.0, 1.0, 1.0),  # Weak boost
        (3.0, 1.0, 1.0, 1.0),
    ]
    
    for p, tb, mb, wb in params:
        if time.time() > deadline - 30.0: break
        
        eta = 1.0 / (d_safe ** p)
        
        # Apply boosts using smooth surrogates
        # Tour edges get strongest boost
        eta[tour_edges] *= tb
        # MST edges get moderate boost  
        eta[mst_edge_arr & ~tour_edges] *= mb
        # MWPM edges get additional boost
        eta[mwpm_edge_arr & ~tour_edges & ~mst_edge_arr] *= wb
        
        eta[mask] = 1e-9
        candidates.append(eta)
        
    # 4. Evaluate candidates with quick ACO
    best_eta = candidates[0]
    best_score = float('inf')
    
    seeds = [42, 123, 456, 789]
    
    for idx, eta in enumerate(candidates):
        if time.time() > deadline - 5.0: break
        
        scores = []
        for s in seeds:
            score = _quick_aco_eval(d, eta, seed=s)
            scores.append(score)
        avg_score = np.mean(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_eta = eta.copy()
            
    return best_eta
