"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an edge-attractiveness matrix for ant colony optimization.
    
    Strategy: 
    Uses a "Greedy Path Simulation" heuristic. Instead of a static local ratio, 
    for each potential move (i, j), we simulate a greedy tour starting from j 
    with the remaining budget. The heuristic value H[i, j] is the total prize 
    collected in that simulated tour. This guides ants towards moves that lead 
    to high-value *complete* paths, not just high-value *next* nodes.
    """
    start_time = time.time()
    deadline = start_time + 290  # 300s limit, keep 10s margin
    
    prizes = np.asarray(prizes)
    distance_matrix = np.asarray(distance_matrix)
    n = len(prizes)
    
    # Precompute: 
    # 1. Return cost for each node (distance to 0)
    return_cost = distance_matrix[:, 0]
    
    # 2. For each node j, precompute the "greedy next step" logic? 
    # No, the greedy simulation depends on the *current* visited set and remaining budget.
    # However, we can approximate the "value of being at j" by simulating a greedy 
    # path from j assuming we haven't visited anything else (except 0 and j).
    # This is an approximation because the actual ant will have visited other nodes.
    # But it captures the "potential" of j much better than local ratio.
    
    # To make this robust and fast, we will compute a "potential" for each node j:
    # V[j] = max prize collectable in a greedy path starting at j, 
    #        assuming current position is 0, visited={0}, budget=max_len.
    # Wait, the move is i->j. The state after moving to j is:
    # current=j, visited={0, ..., i, j}, current_len = len_so_far + d[i,j].
    # The remaining budget is max_len - current_len.
    # The heuristic H[i,j] should reflect the value of j *given* the context of i.
    
    # A full simulation for every (i,j) pair (50*50=2500 sims) might be too slow 
    # if each sim is O(n^2). 2500 * 2500 = 6.25M ops. This is likely fast enough in NumPy/Python 
    # if optimized, but let's check.
    # Actually, a simpler and very effective structural change is:
    # H[i,j] = (prize_j + GreedyPotential(j, remaining_budget)) / (d[i,j] + return_cost[j])
    # But we don't know remaining_budget without knowing the path to i.
    
    # Alternative Structural Change: 
    # Use the "Total Prize of Greedy Tour from j" as the numerator.
    # Let G[j] be the total prize of a greedy tour starting at 0, going to j first, 
    # then greedily picking next nodes until budget runs out.
    # This G[j] is independent of i. 
    # H[i,j] = G[j] / (d[i,j] + return_cost[j])
    # This encourages moves to j that lead to good *overall* tours.
    
    # Let's implement G[j] calculation.
    # For each j in 1..n-1:
    #   current = j
    #   visited = {0, j}
    #   current_len = d[0,j]
    #   total_prize = prizes[j]
    #   while True:
    #       best_next = -1
    #       best_val = -1
    #       for k in unvisited:
    #           if current_len + d[current,k] + d[k,0] <= max_len:
    #               # Greedy choice: max prize / distance? Or just max prize?
    #               # Let's use prize / (d[current,k] + d[k,0]) to be consistent with orienteering logic
    #               val = prizes[k] / (d[current,k] + d[k,0] + 1e-9)
    #               if val > best_val:
    #                   best_val = val
    #                   best_next = k
    #       if best_next == -1: break
    #       total_prize += prizes[best_next]
    #       current_len += d[current, best_next]
    #       current = best_next
    #       visited.add(best_next)
    #   G[j] = total_prize
    
    # Then H[i,j] = G[j] / (d[i,j] + return_cost[j])
    # This is O(n^2 * n) = O(n^3) for the precomputation. 50^3 = 125,000 ops. Very fast.
    
    G = np.zeros(n)
    # Node 0 has prize 0, but we don't move TO 0 in the heuristic context usually (or do we?)
    # The ant moves from i to j. j can be 0? "moves to an unvisited node j". 
    # Node 0 is visited at start. So j != 0.
    
    for j in range(1, n):
        if time.time() > deadline:
            # Fallback to simple heuristic if timeout
            return (prizes[None, :] + 1e-9) / (distance_matrix + return_cost[None, :] + 1e-9)
        
        current = j
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        visited[j] = True
        current_len = distance_matrix[0, j]
        total_prize = prizes[j]
        
        # If we can't even start, G[j] is just prizes[j] (or 0 if infeasible, but j is feasible by definition of being a candidate)
        # Actually, if d[0,j] > max_len, it's infeasible. But the ant logic handles feasibility.
        # We assume j is a valid first step for the simulation.
        
        while True:
            best_next = -1
            best_val = -1
            
            # Find best next node k
            # We can vectorize this inner loop slightly, but n=50 is small.
            for k in range(1, n):
                if not visited[k]:
                    new_len = current_len + distance_matrix[current, k]
                    if new_len + distance_matrix[k, 0] <= max_len:
                        # Greedy metric: Prize / Cost
                        cost = distance_matrix[current, k] + distance_matrix[k, 0]
                        val = prizes[k] / (cost + 1e-9)
                        if val > best_val:
                            best_val = val
                            best_next = k
            
            if best_next == -1:
                break
            
            total_prize += prizes[best_next]
            current_len += distance_matrix[current, best_next]
            current = best_next
            visited[best_next] = True
            
    # Now construct the heuristic matrix
    # H[i, j] = G[j] / (d[i,j] + return_cost[j])
    # G is shape (n,)
    # return_cost is shape (n,)
    # distance_matrix is (n,n)
    
    # Denominator: d[i,j] + return_cost[j]
    # return_cost[j] is the cost to return from j to 0.
    # This term represents the "commitment" of going to j.
    
    denom = distance_matrix + return_cost[None, :]
    
    # Numerator: G[j] broadcasted to rows
    numer = G[None, :]
    
    # Avoid division by zero
    heuristic = numer / (denom + 1e-9)
    
    # Handle node 0 column? 
    # The ant never moves to 0 (it's visited). But the matrix must be full shape.
    # If j=0, G[0]=0, so H[i,0]=0. This is fine.
    
    # Ensure positive values
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
