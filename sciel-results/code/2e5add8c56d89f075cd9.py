import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    n = len(prizes)
    d = distance_matrix
    
    # Ensure inputs are numpy arrays for vectorization
    prizes = np.asarray(prizes, dtype=float)
    d = np.asarray(distance_matrix, dtype=float)
    
    # Node-centric "greedy ratio" heuristic:
    # Attractiveness of moving to j from i is proportional to:
    # prizes[j] / (dist[i,j] + dist[j,0])
    # This represents the prize gained per unit of "effective cost" 
    # (distance to j plus distance back to start, which is the constraint tightener).
    
    # Vectorized computation:
    # dist[i, j] is the matrix d.
    # dist[j, 0] is the column d[:, 0], reshaped to (1, n) to broadcast against rows.
    # prizes[j] is the vector prizes, reshaped to (1, n) to broadcast against rows.
    
    # Cost matrix: cost[i, j] = d[i, j] + d[j, 0]
    # We want heuristic[i, j] = prizes[j] / cost[i, j]
    
    # d is (n, n). d[:, 0] is (n,). We need to add d[j, 0] to each row i of d.
    # So we broadcast d[:, 0] as (1, n).
    dist_to_start = d[:, 0].reshape(1, -1)
    
    effective_cost = d + dist_to_start
    
    # Avoid division by zero or very small numbers
    # The problem states max_len is 3, and points are in unit square, so distances are small.
    # d[i,i] = 0, so cost[i,i] = d[i,0] + d[i,0] = 2*d[i,0].
    # If d[i,0] is 0 (node 0), cost is 0. But we never move to node 0 (it's visited).
    # However, for safety, we add a small epsilon to the denominator.
    
    eps = 1e-9
    effective_cost = np.maximum(effective_cost, eps)
    
    # prizes is (n,). We need prizes[j] for each column j.
    # So we broadcast prizes as (1, n).
    prize_vector = prizes.reshape(1, -1)
    
    # heuristic[i, j] = prizes[j] / (d[i, j] + d[j, 0])
    heuristic = prize_vector / effective_cost
    
    # The diagonal (i == j) should be 0 or very small because we don't move to self.
    # In the ACO, j is unvisited, so i != j. But the matrix shape must be n x n.
    # Setting diagonal to 0 is safe because pheromone[i, i] * heuristic[i, i] will be 0.
    # However, the problem says "Heuristic entries that are not positive become 1e-9".
    # So if we set diagonal to 0, it becomes 1e-9. That's fine.
    # But we can also just let the formula handle it. If i==j, cost is 2*d[i,0].
    # If i=0, cost is 0, so we have division by eps. prizes[0] is 0, so 0/eps = 0.
    # So diagonal entries for node 0 will be 0.
    
    # Ensure all values are positive as per requirement: "Heuristic entries that are not positive become 1e-9"
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Ensure finite values
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Optional: Scale the heuristic. 
    # The ACO probability is proportional to pheromone[i,j] * heuristic[i,j].
    # Pheromone starts at 1. 
    # The scale of heuristic matters relative to pheromone updates.
    # Pheromone update: pheromone = 0.9 * pheromone + best_prize/10.
    # Best prize is sum of prizes, typically around 25 (50 nodes, avg 0.5, visit ~10 nodes? No, max_len=3 is tight).
    # With max_len=3, you can probably visit 5-10 nodes. Prize sum ~ 2-5.
    # So pheromone grows to ~ 0.5.
    # If heuristic values are ~ 1-10, the product is ~ 0.5-5.
    # If heuristic values are ~ 100-1000, the product is ~ 50-500.
    # The relative ratios matter more than absolute scale, as long as they don't overflow/underflow.
    # Let's normalize the heuristic to have a mean of 1 or similar, or just leave it.
    # The "greedy ratio" naturally has a scale. Let's see the typical values.
    # prizes ~ 0.5. dist ~ 0.1-0.5. cost ~ 0.2-1.0. ratio ~ 0.5-2.5.
    # This seems reasonable.
    
    # Check time
    if time.time() > deadline:
         # Fallback to simple prize-based if somehow slow (unlikely for vectorized op)
         return np.maximum(prizes.reshape(1, -1) / (d + 1e-9), 1e-9)
    
    return heuristic
