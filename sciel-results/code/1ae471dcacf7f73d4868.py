"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _nearest_neighbor_tour(dist):
    """Greedy nearest-neighbor tour from node 0."""
    n = dist.shape[0]
    visited = [False] * n
    visited[0] = True
    tour = [0]
    current = 0
    for _ in range(n - 1):
        # Find nearest unvisited node
        best_j = -1
        best_d = float('inf')
        row = dist[current]
        for j in range(n):
            if not visited[j] and row[j] < best_d:
                best_d = row[j]
                best_j = j
        visited[best_j] = True
        tour.append(best_j)
        current = best_j
    return tour


def _two_opt_improve(dist, tour):
    """Apply 2-opt improvement to the tour."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            a = tour[i]
            b = tour[i + 1]
            for j in range(i + 2, n):
                if j == i + 1:
                    continue
                c = tour[j]
                d = tour[(j + 1) % n]
                # Check if swapping segments reduces total distance
                # Current: ...a-b...c-d...
                # After:   ...a-c...b-d...
                current_cost = dist[a][b] + dist[c][d]
                new_cost = dist[a][c] + dist[b][d]
                if new_cost + 1e-12 < current_cost:
                    # Reverse segment between i+1 and j
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def _run_aco_sim(dist, n_ants=20, n_iter=30, seed=42):
    """Run ACO simulation with the given distance matrix."""
    n = dist.shape[0]
    rng = np.random.RandomState(seed)
    
    # Pheromone matrix, starts at 1
    pheromone = np.ones((n, n))
    
    # Heuristic is computed once (this is what we're optimizing)
    # But for simulation, we need a heuristic. We'll use 1/dist as baseline
    # and then the actual heuristics() function will be tested separately.
    # For now, just return the best tour length found.
    
    # Actually, the evaluator calls heuristics(dist) to get the heuristic matrix,
    # then runs ACO with that. So _run_aco_sim is just for internal testing.
    
    # For our purposes, we'll use a simple heuristic to test the ACO framework
    heuristic = 1.0 / (dist + 1e-9)
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    best_tour_len = float('inf')
    
    for iteration in range(n_iter):
        tours = []
        tour_lengths = []
        
        for ant in range(n_ants):
            visited = [False] * n
            visited[0] = True
            tour = [0]
            current = 0
            tour_len = 0.0
            
            for _ in range(n - 1):
                # Compute probabilities for unvisited nodes
                probs = np.zeros(n)
                for j in range(n):
                    if not visited[j]:
                        probs[j] = pheromone[current, j] * heuristic[current, j]
                
                total = probs.sum()
                if total < 1e-30:
                    # Fallback: pick random unvisited
                    unvisited = [j for j in range(n) if not visited[j]]
                    next_node = unvisited[rng.randint(len(unvisited))]
                else:
                    probs = probs / total
                    next_node = rng.choice(n, p=probs)
                
                tour_len += dist[current, next_node]
                tour.append(next_node)
                visited[next_node] = True
                current = next_node
            
            # Close the tour
            tour_len += dist[current, 0]
            tours.append(tour)
            tour_lengths.append(tour_len)
            
            if tour_len < best_tour_len:
                best_tour_len = tour_len
        
        # Update pheromone
        pheromone *= 0.9
        for tour, length in zip(tours, tour_lengths):
            deposit = 1.0 / length
            for i in range(len(tour) - 1):
                i_node = tour[i]
                j_node = tour[i + 1]
                pheromone[i_node, j_node] += deposit
            # Close the tour
            pheromone[tour[-1], tour[0]] += deposit
    
    return best_tour_len


def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix for ACO.
    
    Strategy: Use a combination of:
    1. Inverse distance (base heuristic)
    2. Degree-based correction: penalize edges that lead to high-degree "hubs"
       that are hard to complete, or reward edges that maintain a smooth tour
    3. Local smoothness: for each edge (i,j), consider the angles formed by
       neighboring edges to prefer straighter paths
    
    The key insight is that ACO with pure 1/dist heuristic can get trapped in
    local optima. By adding a structural component that favors edges which
    are part of "natural" tour structures (e.g., edges that don't create
    sharp turns or isolated clusters), we can guide ants better.
    """
    dist = np.asarray(distance_matrix, dtype=np.float64)
    n = dist.shape[0]
    
    # Base: inverse distance
    base = 1.0 / (dist + 1e-9)
    
    # Ensure finite and positive
    base[~np.isfinite(base)] = 1e-9
    base[base <= 0] = 1e-9
    
    # Compute for each node, the k nearest neighbors (k=3)
    k = min(3, n - 1)
    nn = np.zeros((n, k), dtype=int)
    for i in range(n):
        # Get sorted distances from node i
        sorted_indices = np.argsort(dist[i])
        # Skip self
        mask = sorted_indices != i
        sorted_indices = sorted_indices[mask]
        nn[i, :k] = sorted_indices[:k]
    
    # Build a "structural score" matrix
    # For each edge (i,j), score based on whether j is in i's kNN and i is in j's kNN
    struct_score = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            score = 1.0
            if j in nn[i]:
                score *= 1.5
            if i in nn[j]:
                score *= 1.5
            # Penalize long edges more aggressively
            # Use a power law: shorter edges get disproportionately higher score
            ratio = dist[i, j] / dist[i].mean()
            score *= 1.0 / (1.0 + 0.5 * max(0, ratio - 1.0))
            struct_score[i, j] = score
    
    # Combine base and structural scores
    # Normalize both to [0, 1] range for combination
    base_min, base_max = base.min(), base.max()
    struct_min, struct_max = struct_score.min(), struct_score.max()
    
    if base_max > base_min:
        base_norm = (base - base_min) / (base_max - base_min)
    else:
        base_norm = np.ones_like(base)
    
    if struct_max > struct_min:
        struct_norm = (struct_score - struct_min) / (struct_max - struct_min)
    else:
        struct_norm = np.ones_like(struct_score)
    
    # Weighted combination: 70% base, 30% structural
    combined = 0.7 * base_norm + 0.3 * struct_norm
    
    # Scale back to a reasonable range for ACO (pheromone starts at 1,
    # so heuristic values should be O(1))
    combined = combined * (base_max - base_min) + base_min
    
    # Final cleanup
    combined[~np.isfinite(combined)] = 1e-9
    combined[combined <= 0] = 1e-9
    
    # Make diagonal 0 (no self-loops)
    np.fill_diagonal(combined, 0.0)
    
    return combined
# EVOLVE-BLOCK-END
