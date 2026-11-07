import numpy as np
import time

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=42):
    """
    Fast simulation of the ACO process.
    Returns the best prize collected by any ant in any iteration.
    """
    n = len(prizes)
    # Ensure heuristic is valid (positive)
    H = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_prize = 0.0
    rng = np.random.default_rng(seed)
    
    # Pre-compute distance to start for all nodes
    d_to_start = distance_matrix[:, 0]
    
    for it in range(num_iters):
        best_iter_prize = 0.0
        
        for _ in range(num_ants):
            current_node = 0
            visited_mask = np.zeros(n, dtype=bool)
            visited_mask[0] = True
            path_len = 0.0
            total_prize = prizes[0]
            
            while True:
                remaining_budget = max_len - path_len
                # Condition: path_len + d[i, j] + d[j, 0] <= max_len
                # => d[i, j] + d[j, 0] <= max_len - path_len
                d_from_current = distance_matrix[current_node, :]
                
                # Vectorized check for valid moves
                valid_indices = np.where(
                    (~visited_mask) & 
                    (d_from_current + d_to_start <= remaining_budget)
                )[0]
                
                if len(valid_indices) == 0:
                    break
                
                # Get heuristic weights for valid nodes
                h_vals = H[current_node, valid_indices]
                
                # Normalize to probabilities
                total_h = np.sum(h_vals)
                if total_h <= 0:
                    break
                    
                probs = h_vals / total_h
                
                # Sample next node
                next_node = valid_indices[rng.choice(len(valid_indices), p=probs)]
                
                # Update state
                path_len += distance_matrix[current_node, next_node]
                total_prize += prizes[next_node]
                visited_mask[next_node] = True
                current_node = next_node
            
        best_iter_prize = max(best_iter_prize, total_prize)
        best_prize = max(best_prize, best_iter_prize)

    return best_prize

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit
    
    n = len(prizes)
    eps = 1e-9
    
    # Pre-compute distance to start
    d_to_start = distance_matrix[:, 0]
    
    # Define feasibility: An edge (i, j) is "feasible" if starting at 0,
    # going to i, then j, then back to 0 fits within max_len?
    # Actually, the constraint is dynamic: path_len + d[i,j] + d[j,0] <= max_len.
    # Since path_len >= d[0,i] (triangle inequality roughly, but not strictly),
    # a necessary condition for j to ever be reachable from i is d[i,j] + d[j,0] < max_len.
    # If d[i,j] + d[j,0] >= max_len, then even if we arrived at i with 0 cost (impossible unless i=0),
    # we couldn't return.
    # More strictly, if we are at i, we must have path_len >= 0.
    # So if d[i,j] + d[j,0] > max_len, it's NEVER possible to move i->j and return.
    # We can set H[i,j] to a very small value for such edges to prevent wasting probability mass.
    
    feasible = (distance_matrix + d_to_start[None, :]) <= max_len
    
    # Initial Heuristic:
    # High prize, low distance.
    # h[i, j] = (prizes[j] + 0.1) / (distance[i, j] + 0.1)
    # But mask infeasible edges with a tiny value.
    
    base_prize = prizes[None, :] + 0.1
    base_dist = distance_matrix + 0.1
    H = base_prize / base_dist
    
    # Apply feasibility mask
    H = np.where(feasible, H, eps)
    
    # Ensure no zeros/negatives
    H = np.maximum(H, eps)
    
    best_score = _simulate_aco(prizes, distance_matrix, max_len, H)
    
    # Coordinate Ascent
    # We treat H as a vector of independent variables.
    # For each cell (i, j), we try to find a value that maximizes the score.
    # Since the landscape is discrete (due to RNG sampling), we use a "hill climbing" approach
    # by testing a set of candidate multipliers for each cell.
    
    # Optimization: Only optimize cells that are feasible and have non-zero prize potential.
    # Cells with prize 0 (node 0) don't matter much for attraction TO them (we start there),
    # but attraction FROM them matters.
    
    # Pre-compute indices to iterate over
    # We will cycle through (i, j) pairs.
    # To save time, we can prioritize cells that are likely important:
    # 1. Edges leaving node 0 (first move).
    # 2. Edges between high-prize nodes.
    
    # Let's just do a few full passes over the matrix.
    # Number of cells: 50*50 = 2500.
    # Simulation cost: ~3000-5000 ops.
    # 2500 cells * 3 candidates * 1 sim = 7500 sims per pass.
    # 7500 sims * ~2ms/sim = 15s per pass.
    # We have ~280s. So ~15-20 passes are feasible.
    
    # Candidate values for each cell:
    # We keep the current value, and try to scale it up or down.
    # Scaling factors: 0.1, 0.5, 2.0, 10.0
    
    # To make it more effective, we can also try setting it to a "baseline" value
    # based on the global prize/distance ratio, in case local optima trap us.
    
    # However, a simpler and often effective strategy for discrete stochastic optimization:
    # Try a grid of values for each cell? No, too expensive.
    # Let's use a randomized coordinate ascent with multiple restarts or larger steps.
    
    # Actually, since the ACO is stochastic, a single simulation is noisy.
    # But we are forced to use the same seed. So the function is deterministic given H.
    # This makes coordinate ascent valid.
    
    # Strategy:
    # For each cell (i, j):
    #   Try current H[i, j]
    #   Try H[i, j] * 0.5
    #   Try H[i, j] * 2.0
    #   Try H[i, j] * 0.1
    #   Try H[i, j] * 10.0
    #   Pick the one with the highest score.
    
    # Optimization:
    # Many cells are irrelevant (infeasible). We already set them to eps.
    # We should not optimize infeasible cells.
    
    # Let's create a list of (i, j) to optimize.
    cells_to_optimize = np.argwhere(feasible)
    
    # Shuffle the order to avoid bias? Or keep it row-major?
    # Row-major is fine.
    
    num_passes = 10 # Conservative estimate
    
    for p in range(num_passes):
        if time.time() > deadline - 20:
            break
            
        # Optional: Shuffle cells in later passes?
        # For now, let's just iterate.
        
        for idx, (i, j) in enumerate(cells_to_optimize):
            if time.time() > deadline - 20:
                break
                
            current_val = H[i, j]
            current_score = best_score # This is the score with the CURRENT H matrix
            
            # Candidates
            # We want to explore values around the current one.
            # Also, a "reset" to a standard value might help escape local minima.
            standard_val = (prizes[j] + 0.1) / (distance_matrix[i, j] + 0.1)
            
            candidates = [
                current_val,
                current_val * 0.5,
                current_val * 2.0,
                current_val * 0.1,
                current_val * 10.0,
                standard_val
            ]
            
            best_cand_val = current_val
            best_cand_score = current_score
            
            # Remove duplicates to save time
            unique_cands = []
            seen = set()
            for c in candidates:
                # Round to avoid floating point issues in set
                key = round(c, 12)
                if key not in seen:
                    seen.add(key)
                    unique_cands.append(c)
            
            for cand_val in unique_cands:
                if cand_val <= 0:
                    cand_val = eps
                
                # Temporarily update H
                H[i, j] = cand_val
                score = _simulate_aco(prizes, distance_matrix, max_len, H)
                
                if score > best_cand_score:
                    best_cand_score = score
                    best_cand_val = cand_val
            
            # Restore best value
            H[i, j] = best_cand_val
            best_score = best_cand_score

    # Final check: ensure all values are positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, eps)
    
    return H
