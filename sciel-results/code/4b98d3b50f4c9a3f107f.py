"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _nearest_neighbor_tour(dm, start=0):
    """Greedy nearest-neighbor tour from start."""
    n = dm.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    cur = start
    for _ in range(n - 1):
        # Find nearest unvisited node
        row = dm[cur].copy()
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        visited[nxt] = True
        tour.append(nxt)
        cur = nxt
    return tour


def _two_opt(tour, dm, max_passes=100):
    """Apply 2-opt local search to improve tour."""
    n = len(tour)
    best = list(tour)
    best_len = (dm[best, np.roll(best, -1)]).sum()
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            if improved:
                break
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Edge (i, i+1) and (j, j+1)
                a, b = best[i], best[i + 1]
                c, d = best[j], best[(j + 1) % n]
                # Delta: remove (a,b) and (c,d), add (a,c) and (b,d)
                delta = dm[a, c] + dm[b, d] - dm[a, b] - dm[c, d]
                if delta < -1e-10:
                    # Reverse segment i+1..j
                    best[i + 1:j + 1] = best[i + 1:j + 1][::-1]
                    best_len += delta
                    improved = True
    return best, best_len


def _compute_directed_bias(dm):
    """Compute a directed heuristic bias from multiple NN+2opt tours."""
    n = dm.shape[0]
    deadline = time.time() + 120.0  # Safety margin
    
    # Multi-start NN + 2-opt
    tours = []
    start_nodes = list(range(n))
    np.random.shuffle(start_nodes)
    
    for start in start_nodes:
        if time.time() > deadline:
            break
        tour = _nearest_neighbor_tour(dm, start)
        tour, _ = _two_opt(tour, dm)
        tours.append(tour)
        
        # Early stop if we have enough good tours
        if len(tours) >= 20:
            break
    
    # Build directed edge frequency/quality matrix
    # For each directed edge (i,j) in good tours, boost heuristic
    bias = np.ones((n, n), dtype=np.float64)
    
    for tour in tours:
        for k in range(n):
            i = tour[k]
            j = tour[(k + 1) % n]
            bias[i, j] += 1.0
    
    # Normalize: larger bias = more attractive
    # Use log to compress dynamic range
    bias = np.log(bias + 1e-6)
    
    return bias


def heuristics(distance_matrix):
    """Return edge-attractiveness matrix (larger = more attractive)."""
    dm = np.asarray(distance_matrix, dtype=np.float64)
    n = dm.shape[0]
    
    # Base heuristic: inverse distance
    base = 1.0 / (dm + 1e-9)
    
    # Directed bias from local search
    try:
        bias = _compute_directed_bias(dm)
        # Combine: base * bias, then normalize
        combined = base * bias
        # Ensure positive
        combined = np.maximum(combined, 1e-9)
        return combined
    except Exception:
        # Fallback to base heuristic
        return base
# EVOLVE-BLOCK-END
