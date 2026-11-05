"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _evaluate_fast(adjacency, nbrs, scores):
    """
    Fast evaluation of the greedy independent set size.
    adjacency: bool matrix (not used if nbrs is provided)
    nbrs: list of lists of neighbor indices
    scores: array of scores
    """
    n = len(nbrs)
    if n == 0:
        return 0
    
    # Sort vertices by descending score
    # kind='stable' ensures consistent ordering for equal scores
    indices = np.argsort(-scores, kind='stable')
    
    blocked = np.zeros(n, dtype=bool)
    count = 0
    
    for v in indices:
        if not blocked[v]:
            count += 1
            blocked[v] = True
            # Block neighbors
            if len(nbrs[v]) > 0:
                blocked[nbrs[v]] = True
                
    return count


def priority(adjacency):
    """
    Compute vertex scores for the Maximum Independent Set greedy decoder.
    Strategy: Prefer low-degree vertices that are connected to high-degree vertices.
    This helps in blocking hubs early by selecting their low-degree neighbors.
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]

    if n == 0:
        return np.array([])

    # Precompute neighbor lists
    nbrs = [np.where(adjacency[i])[0] for i in range(n)]
    degree = np.array([len(nbrs[i]) for i in range(n)], dtype=float)

    # Helper to normalize features to [0, 1]
    def norm(x):
        x = np.asarray(x, dtype=float)
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    # 1. Degree
    D_n = norm(degree)

    # 2. Mean Neighbor Degree
    mean_nd = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            mean_nd[i] = np.mean(degree[nbrs[i]])
        else:
            mean_nd[i] = 0
    MeanND_n = norm(mean_nd)

    # 3. Sum of Neighbor Degrees
    snd = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            snd[i] = np.sum(degree[nbrs[i]])
    SND_n = norm(snd)

    # 4. Clustering Coefficient
    # Use matrix multiplication for triangles: tri[i] = 0.5 * (A^2)[i,i]
    A = adjacency.astype(float)
    A_sq = A @ A
    tri = np.diag(A_sq) / 2.0
    
    clus = np.zeros(n)
    mask = degree > 1
    # Avoid division by zero
    denom = degree[mask] * (degree[mask] - 1.0)
    clus[mask] = 2.0 * tri[mask] / denom
    Clus_n = norm(clus)

    # 5. Core Number (Simplified k-core approximation)
    # We use the degeneracy ordering or a simple k-core peeling.
    # For n=200, a simple iterative peeling is fast enough.
    core = np.zeros(n, dtype=int)
    deg_copy = degree.astype(int).copy()
    remaining = np.ones(n, dtype=bool)
    
    # To make it faster, we can just use the degree as a proxy if core is expensive,
    # but core is a good feature. Let's do a simple version.
    # Sort by degree ascending for peeling?
    # Standard k-core: iteratively remove vertices with degree < k.
    # Here we want the core number of each vertex.
    # We can compute this by repeatedly removing the min-degree vertex.
    # This is O(n^2) which is fine for n=200.
    
    order = np.argsort(degree)
    current_deg = degree.astype(int).copy()
    removed = np.zeros(n, dtype=bool)
    
    # This simple loop assigns core number based on the min-degree at time of removal
    # which is equivalent to the k-core number for the graph.
    # Actually, the standard algorithm for core numbers is:
    # 1. Sort vertices by degree.
    # 2. Iterate, removing the vertex with current min degree, updating neighbors.
    # The core number of v is the min degree it had when it was removed? 
    # No, it's the max k such that v is in the k-core.
    # A simpler approximation: use the degree in the graph after removing all lower-degree vertices.
    
    # Let's just use a simple heuristic: 
    # Core_n will be the normalized degree, as computing exact core numbers 
    # with a naive loop might be slightly more complex to get exactly right 
    # in a short time, but degree is a strong proxy. 
    # However, let's try a quick exact calculation.
    
    # Exact Core Number Calculation
    deg_curr = degree.astype(int).copy()
    alive = np.ones(n, dtype=bool)
    core_nums = np.zeros(n, dtype=int)
    
    # Use a bucket sort approach or just simple loops since n=200
    # We will repeatedly find the vertex with min degree among alive vertices,
    # assign its core number, remove it, and update neighbors.
    
    for _ in range(n):
        if not np.any(alive):
            break
        # Find min degree among alive
        alive_degrees = np.where(alive, deg_curr, np.inf)
        v = np.argmin(alive_degrees)
        if not alive[v]:
            continue
        core_nums[v] = deg_curr[v]
        alive[v] = False
        for u in nbrs[v]:
            if alive[u]:
                deg_curr[u] -= 1
                
    Core_n = norm(core_nums.astype(float))

    # --- Search for Best Weights ---
    # Hypothesis: Score = -w_D * D + w_MeanND * MeanND + w_Clus * Clus + w_Core * Core
    # We expect w_D > 0 (penalize high degree), w_MeanND > 0 (reward high neighbor degree).
    
    best_count = -1
    best_scores = -D_n.copy()
    
    # Candidate weights: (w_D, w_MeanND, w_SND, w_Clus, w_Core)
    # Note: We apply -w_D * D_n, so if w_D is positive, it penalizes high D_n.
    candidates = [
        (1.0, 0.0, 0.0, 0.0, 0.0), # Baseline: Low Degree
        (1.0, 0.5, 0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, 0.0, 0.0),
        (1.0, 1.5, 0.0, 0.0, 0.0),
        (1.0, 2.0, 0.0, 0.0, 0.0),
        (1.0, 2.5, 0.0, 0.0, 0.0),
        (1.0, 3.0, 0.0, 0.0, 0.0),
        
        (1.0, 1.0, 0.0, 0.5, 0.0),
        (1.0, 1.0, 0.0, -0.5, 0.0),
        (1.0, 1.5, 0.0, 0.5, 0.0),
        (1.0, 1.5, 0.0, -0.5, 0.0),
        
        (1.0, 1.0, 0.5, 0.0, 0.0),
        (1.0, 1.0, -0.5, 0.0, 0.0),
        
        (1.0, 1.0, 0.0, 0.0, 0.5),
        (1.0, 1.0, 0.0, 0.0, -0.5),
        
        (1.0, 1.5, 0.5, 0.5, -0.5),
        (1.0, 2.0, 0.0, 0.5, -0.5),
        (1.0, 1.0, 0.5, 0.5, 0.5),
    ]
    
    for w_D, w_MeanND, w_SND, w_Clus, w_Core in candidates:
        if time.time() > deadline:
            break
            
        scores = (-w_D * D_n + 
                  w_MeanND * MeanND_n + 
                  w_SND * SND_n + 
                  w_Clus * Clus_n + 
                  w_Core * Core_n)
        
        count = _evaluate_fast(adjacency, nbrs, scores)
        
        if count > best_count:
            best_count = count
            best_scores = scores.copy()

    # If we have time, do a small local refinement around the best weights
    # We don't store the weights, so we'd need to re-derive or just skip.
    # Given the grid is small and covers the likely space, we will return the best found.
    
    return best_scores

# EVOLVE-BLOCK-END
