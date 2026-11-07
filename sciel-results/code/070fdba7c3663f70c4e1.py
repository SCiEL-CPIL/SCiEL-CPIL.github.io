"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute a heuristic matrix for Ant Colony Optimization using a Lookahead Greedy Strategy.
    
    Strategy:
    Instead of a static local ratio, we compute a heuristic h[i, j] that estimates the 
    total prize achievable if the ant moves from i to j.
    
    h[i, j] is defined as:
    1. prize[j] (the immediate reward)
    2. Plus the estimated future prize from a greedy tour starting at j, 
       constrained by the remaining budget.
    
    The greedy future tour is constructed by repeatedly selecting the unvisited node k 
    that maximizes prize[k] / (d[current, k] + d[k, 0]) such that the move is feasible.
    This provides a global, path-dependent attractiveness signal.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Ensure non-negative and finite
    prizes = np.clip(prizes, 0, None)
    distance_matrix = np.clip(distance_matrix, 0, None)
    
    n = len(prizes)
    eps = 1e-9
    
    # Pre-compute return costs to 0
    return_costs = distance_matrix[:, 0]
    
    # h[i, j] will be the estimated total prize if we go i -> j -> ... -> 0
    h = np.zeros((n, n), dtype=float)
    
    # We need to calculate the "future value" for each node j given a specific remaining budget.
    # The remaining budget when moving i -> j is: max_len - d[i, j] - d[j, 0]
    # Let R[i, j] = max_len - d[i, j] - d[j, 0]
    
    # To make this efficient, we can pre-compute a "greedy potential" for each node j
    # assuming it starts with a full budget, but that's not quite right because the budget
    # depends on i.
    
    # However, n=50 is small. We can afford O(n^3) or even O(n^4) operations if optimized.
    # Let's compute the greedy future prize for each node j, starting from j, 
    # with a variable budget. But the budget is continuous.
    
    # Alternative: Since max_len is small (3.0) and distances are Euclidean in unit square,
    # the number of nodes visitable is small.
    
    # Let's define a function to compute the greedy prize from node 'start' with 'budget' left.
    # But calling this for every (i, j) pair (2500 pairs) with an inner loop might be slow in Python.
    # We need a vectorized or highly optimized approach.
    
    # Observation: The greedy choice depends on the current node and the remaining budget.
    # Let's approximate the future value by pre-computing the "best greedy tour" for each starting node j
    # assuming the *maximum possible* remaining budget (which is max_len - d[j,0]).
    # This is an upper bound. 
    # Actually, a better approximation for h[i,j] is:
    # h[i,j] = prize[j] + sum of prizes of nodes in a greedy tour starting from j, 
    # where the tour is constrained by the *actual* remaining budget R[i,j].
    
    # Given n=50, we can iterate over all pairs (i, j).
    # For each pair, we run a greedy simulation.
    # 50 * 50 = 2500 simulations.
    # Each simulation takes at most O(n^2) steps (finding best next node).
    # Total O(n^4) = 50^4 = 6.25 million operations. This is very fast in Python/NumPy.
    
    # Let's implement the greedy simulation for each (i, j).
    
    for i in range(n):
        for j in range(n):
            if i == j:
                h[i, j] = 0
                continue
            
            d_ij = distance_matrix[i, j]
            d_j0 = return_costs[j]
            
            # Feasibility check: Can we go i -> j -> 0?
            if d_ij + d_j0 > max_len:
                h[i, j] = 1e-9
                continue
            
            # Immediate prize
            total_prize = prizes[j]
            
            # Remaining budget for future nodes after visiting j
            # The path so far is 0 -> ... -> i -> j.
            # Wait, the ant starts at 0. The heuristic h[i,j] is used when the ant is at i.
            # The length so far is NOT known to the heuristic function (it's static).
            # The heuristic is computed ONCE. It does not know the path taken.
            # Therefore, we must assume the "worst case" or "average case" or simply 
            # that the ant arrives at i with the *minimum* possible distance from 0 to i?
            # No, the heuristic is static. It cannot depend on the path length.
            
            # The standard ACO heuristic eta[i,j] is usually static, e.g., 1/d[i,j].
            # My proposed strategy depends on the *remaining budget*, which depends on the path.
            # Since the heuristic is computed once, it cannot know the current path length.
            
            # REVISION OF STRATEGY:
            # The heuristic h[i,j] must be static. 
            # However, we can model the *potential* of node j.
            # If we assume the ant is at i, the *minimum* distance from 0 to i is d[0,i] (triangle inequality).
            # So the *maximum* remaining budget when at i is max_len - d[0,i].
            # When moving to j, the *maximum* remaining budget for future nodes is:
            # max_len - d[0,i] - d[i,j] - d[j,0]
            # This is an optimistic estimate of the slack.
            
            # Let's use this optimistic slack to compute the future greedy prize.
            # Slack = max_len - d[0,i] - d[i,j] - d[j,0]
            
            d_0i = distance_matrix[0, i]
            slack = max_len - d_0i - d_ij - d_j0
            
            if slack < 0:
                # Even with the most efficient path to i, we can't go to j and return
                # with any budget left? Or just can't go to j and return?
                # If d_0i + d_ij + d_j0 > max_len, then even the direct path 0->i->j->0 is too long.
                # But the ant might have taken a longer path to i? No, if the direct path is too long, 
                # any path to i is >= d_0i, so it's definitely infeasible.
                h[i, j] = 1e-9
                continue
            
            # Now, estimate the future prize from j with 'slack' budget.
            # We simulate a greedy tour starting at j, with current length 0 (relative to j),
            # but constrained by the total additional length 'slack'.
            # Note: The return to 0 is already accounted for in the slack calculation?
            # No, slack is the budget left for *other* nodes between j and the return to 0?
            # Let's clarify the budget:
            # Total Budget: max_len
            # Path: 0 -> ... -> i -> j -> k1 -> k2 -> ... -> 0
            # Length: d(0...i) + d(i,j) + d(j,k1) + ... + d(km,0) <= max_len
            # We approximated d(0...i) by d(0,i).
            # So remaining budget for (j -> k1 -> ... -> 0) is max_len - d(0,i) - d(i,j).
            # Let B = max_len - d(0,i) - d(i,j).
            # We must visit j and return to 0. Cost: d(j,0).
            # Remaining for intermediate nodes: B - d(j,0) = slack.
            
            # So we have 'slack' length to spend on nodes between j and 0.
            # We start at j. We want to visit nodes k such that d(j,k) + ... + d(k,0) <= slack.
            # Actually, the greedy step is: from current node c, pick k such that d(c,k) + d(k,0) <= remaining_budget.
            # Then update remaining_budget -= d(c,k).
            
            current_node = j
            remaining_budget = slack
            visited = set([j])
            
            while remaining_budget > 0:
                best_prize_ratio = -1
                best_next = -1
                
                # Find best next node k
                # k must be unvisited
                # d(current_node, k) + d(k, 0) <= remaining_budget
                # Maximize prize[k] / d(current_node, k) ? Or prize[k]?
                # Let's maximize prize[k] / (d(current_node, k) + eps) to get high value per cost.
                
                # Vectorized search for best k
                # dists = distance_matrix[current_node, :]
                # rets = return_costs
                # feasible = (dists + rets) <= remaining_budget
                # feasible[list(visited)] = False
                # ratios = prizes / (dists + eps)
                # ratios[~feasible] = -1
                # best_idx = argmax(ratios)
                
                dists = distance_matrix[current_node, :]
                rets = return_costs
                feasible_mask = (dists + rets) <= remaining_budget
                
                # Exclude visited
                for v in visited:
                    feasible_mask[v] = False
                
                if not np.any(feasible_mask):
                    break
                
                # Calculate ratios
                # Avoid division by zero
                safe_dists = np.where(dists > 0, dists, 1.0)
                ratios = prizes / safe_dists
                ratios[~feasible_mask] = -1
                
                best_idx = np.argmax(ratios)
                
                if ratios[best_idx] <= 0:
                    break
                
                # Move to best_idx
                total_prize += prizes[best_idx]
                remaining_budget -= distance_matrix[current_node, best_idx]
                current_node = best_idx
                visited.add(best_idx)
                
                # Check time
                if time.time() > deadline:
                    # Emergency exit: return current h
                    # Fill rest with 1e-9
                    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
                    return h
            
            h[i, j] = total_prize
            
    # Ensure all values are positive and finite
    h = np.where(np.isfinite(h), h, 1e-9)
    h = np.maximum(h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
