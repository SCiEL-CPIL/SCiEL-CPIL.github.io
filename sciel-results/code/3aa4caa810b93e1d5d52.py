"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _construct_nn_tour(distance_matrix, demands, capacity):
    """
    Construct a tour using a simple Nearest Neighbor heuristic with capacity constraints.
    Returns a list of edges (i, j) that form the routes.
    """
    n = distance_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    remaining_cap = capacity
    edges = []
    
    while not visited[1:].all():
        # Find eligible customers
        eligible_mask = (~visited) & (np.arange(n) > 0)
        eligible_mask &= (demands[:n] <= remaining_cap)
        eligible_indices = np.where(eligible_mask)[0]
        
        if len(eligible_indices) == 0:
            # Return to depot
            edges.append((current, 0))
            current = 0
            remaining_cap = capacity
            if visited[1:].all():
                break
            continue
        
        # Pick nearest eligible customer
        dists = distance_matrix[current, eligible_indices]
        nearest_idx = np.argmin(dists)
        next_node = eligible_indices[nearest_idx]
        
        edges.append((current, next_node))
        remaining_cap -= demands[next_node]
        visited[next_node] = True
        current = next_node
        
        if visited[1:].all():
            edges.append((current, 0))
            break
            
    return edges

def _fit_heuristic(distance_matrix, edges):
    """
    Create a heuristic matrix that strongly favors the edges in the provided tour.
    Non-tour edges get a low but positive value to allow exploration.
    """
    n = distance_matrix.shape[0]
    # Base heuristic: inverse distance (standard ACO heuristic)
    # We use a small epsilon to avoid division by zero
    h = 1.0 / (distance_matrix + 1e-6)
    
    # Boost the edges found in the NN tour significantly
    # This "seeds" the pheromone-like attraction towards a good structure
    for i, j in edges:
        h[i, j] *= 10.0
        
    # Ensure symmetry isn't broken in a way that hurts, though ACO usually uses directed
    # The task says "edge-attractiveness matrix", ACO uses pheromone[i,j] * heuristic[i,j]
    # Usually heuristic is symmetric in CVRP, but let's keep it as is.
    
    # Normalize to prevent extreme values? 
    # The task says "larger is more attractive". 
    # If values are too large, probability becomes 1 for those edges, reducing exploration.
    # Let's keep the boost moderate.
    
    return h

def _run_aco(distance_matrix, coordinates, demands, capacity, heuristic, n_ants=20, n_iterations=30, seed=42):
    """Run ACO and return the best (lowest) total route distance found by any ant."""
    rng = np.random.RandomState(seed)
    n = distance_matrix.shape[0]
    pheromone = np.ones_like(heuristic, dtype=float)
    # Ensure heuristic is positive
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_overall = float('inf')
    
    for it in range(n_iterations):
        best_iter_cost = float('inf')
        
        for ant in range(n_ants):
            # Ant state
            current = 0
            remaining_cap = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            total_dist = 0.0
            
            while True:
                # Find eligible customers
                eligible_mask = (~visited) & (np.arange(n) > 0)
                eligible_mask &= (demands[:n] <= remaining_cap)
                eligible_indices = np.where(eligible_mask)[0]
                
                if len(eligible_indices) == 0:
                    # Return to depot
                    total_dist += distance_matrix[current, 0]
                    current = 0
                    remaining_cap = capacity
                    # Check if all customers visited
                    if visited[1:].all():
                        break
                    continue
                
                # Compute probabilities
                # P(i,j) = (tau_ij * eta_ij) / sum(...)
                probs = pheromone[current, eligible_indices] * heuristic[current, eligible_indices]
                probs = np.where(probs > 0, probs, 1e-9)
                probs /= probs.sum()
                
                # Pick next
                idx = rng.choice(len(eligible_indices), p=probs)
                next_node = eligible_indices[idx]
                
                total_dist += distance_matrix[current, next_node]
                remaining_cap -= demands[next_node]
                visited[next_node] = True
                current = next_node
                
                # If all visited, go back to depot
                if visited[1:].all():
                    total_dist += distance_matrix[current, 0]
                    break
        
        # Track best per iteration
        if total_dist < best_iter_cost:
            best_iter_cost = total_dist
        
        # Track overall best
        if total_dist < best_overall:
            best_overall = total_dist
        
        # Pheromone update
        pheromone *= 0.9
        pheromone += 1.0 / best_iter_cost
    
    return best_overall

def heuristics(distance_matrix, coordinates, demands, capacity):
    deadline = time.time() + 290  # Safety margin: 330s kill, leave 40s buffer
    
    # 1. Generate a strong initial tour using Nearest Neighbor
    # This is deterministic and very fast for N=50
    nn_edges = _construct_nn_tour(distance_matrix, demands, capacity)
    
    # 2. Fit the heuristic to this tour
    # This creates a "landscape" where the NN tour is a deep valley.
    # ACO will start by following this valley but can escape if it finds something better.
    heuristic = _fit_heuristic(distance_matrix, nn_edges)
    
    # 3. Optional: Quick validation run to ensure it's not worse than a baseline
    # We can run a single ACO iteration or just return the fitted heuristic.
    # Given the tight time budget and that "fitting" is a strong prior, 
    # we trust the structure. However, to be safe against pathological cases,
    # let's do a very short ACO run (e.g., 1 iteration) to see if it crashes or behaves badly.
    # Actually, the task says "Heuristic is computed once... before the colony runs".
    # The evaluator runs the colony. We just need to return the matrix.
    
    # Ensure validity: positive, finite, correct shape
    if heuristic.shape != distance_matrix.shape:
        # Fallback to standard inverse distance
        heuristic = 1.0 / (distance_matrix + 1e-9)
        
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
