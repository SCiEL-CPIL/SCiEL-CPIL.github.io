"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO.
    Uses a tuned gamma parameter to balance distance penalty.
    """
    start_time = time.time()
    n = distance_matrix.shape[0]
    
    # Convert inputs to numpy arrays for consistent handling
    dm = np.asarray(distance_matrix, dtype=np.float64)
    
    # The heuristic is typically eta_ij = 1 / d_ij
    # To tune it, we can use eta_ij = 1 / (d_ij)^gamma
    # A higher gamma makes the heuristic more sensitive to distance differences,
    # effectively making ants prefer shorter edges more strongly.
    # A lower gamma makes the heuristic flatter, relying more on pheromones.
    
    # We will test a few values of gamma to see which one yields better results
    # in a quick proxy ACO run.
    
    def run_quick_aco(gamma, n_ants=10, n_iters=10):
        """
        Run a quick ACO with the given gamma to estimate quality.
        Returns the best cost found.
        """
        # Compute heuristic for this gamma
        with np.errstate(divide='ignore', invalid='ignore'):
            eta = 1.0 / np.power(dm + 1e-9, gamma)
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        
        # Initialize pheromones
        tau = np.ones_like(dm)
        
        best_cost = float('inf')
        
        demands_arr = np.asarray(demands, dtype=np.float64)
        
        for _ in range(n_iters):
            iter_best_cost = float('inf')
            
            for _ in range(n_ants):
                # Ant starts at depot (0)
                current = 0
                unvisited = set(range(1, n))
                load = 0.0
                cost = 0.0
                
                while unvisited:
                    # Find feasible neighbors
                    feasible = []
                    for j in unvisited:
                        if demands_arr[j] <= capacity - load:
                            feasible.append(j)
                    
                    if not feasible:
                        # Return to depot
                        cost += dm[current, 0]
                        current = 0
                        load = 0.0
                        continue
                    
                    # Compute probabilities
                    probs = np.zeros(len(feasible))
                    for k, j in enumerate(feasible):
                        probs[k] = tau[current, j] * eta[current, j]
                    
                    total_prob = np.sum(probs)
                    if total_prob == 0:
                        # Fallback: pick randomly
                        next_node = random.choice(feasible)
                    else:
                        probs = probs / total_prob
                        r = random.random()
                        cumsum = np.cumsum(probs)
                        idx = np.searchsorted(cumsum, r)
                        if idx >= len(feasible):
                            idx = len(feasible) - 1
                        next_node = feasible[idx]
                    
                    # Move to next node
                    cost += dm[current, next_node]
                    load += demands_arr[next_node]
                    unvisited.remove(next_node)
                    current = next_node
                
                # Return to depot
                cost += dm[current, 0]
                
                if cost < iter_best_cost:
                    iter_best_cost = cost
            
            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
            
            # Update pheromones
            tau *= 0.9
            if iter_best_cost > 0:
                tau += 1.0 / iter_best_cost
        
        return best_cost
    
    # Candidate gammas
    candidate_gammas = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    best_gamma = 1.0
    best_score = float('inf')
    
    for gamma in candidate_gammas:
        # Check time budget
        if time.time() - start_time > 10:
            break
        
        score = run_quick_aco(gamma)
        if score < best_score:
            best_score = score
            best_gamma = gamma
    
    # Compute final heuristic with best gamma
    with np.errstate(divide='ignore', invalid='ignore'):
        heuristic = 1.0 / np.power(dm + 1e-9, best_gamma)
    
    # Ensure positive and finite
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END
