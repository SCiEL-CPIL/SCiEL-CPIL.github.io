"""
Rewritten heuristic for Multiple-Constraint Knapsack Problem (MCKP)
using Pareto dominance filtering to guide Ant Colony Optimization.
"""
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes heuristic attractiveness scores for items.
    
    Args:
        prizes: np.array. Shape (100,) or (5, 100). If (5, 100), sums are taken.
        weights: np.array. Shape (5, 100).
        
    Returns:
        np.array of shape (100,) with heuristic scores.
    """
    start_time = time.time()
    
    prizes = np.asarray(prizes)
    weights = np.asarray(weights)
    
    # Ensure prizes is 1D
    if prizes.ndim == 2:
        total_prizes = np.sum(prizes, axis=0)
    else:
        total_prizes = prizes.flatten()
        
    # Ensure weights is 2D (5, N)
    if weights.ndim == 1:
        # Fallback if shape is weird, but prompt guarantees (5,100)
        weights = weights.reshape(1, -1)
        
    n_items = len(total_prizes)
    
    # Calculate sum of weights per item
    # weights shape: (5, 100) -> sum(axis=0) -> (100,)
    sum_weights = np.sum(weights, axis=0)
    
    # Base heuristic: Prize-to-Weight ratio
    epsilon = 1e-9
    base_heuristic = total_prizes / (sum_weights + epsilon)
    
    # Pareto Dominance Check
    # Item i is dominated if there exists j such that:
    # 1. total_prizes[j] >= total_prizes[i]
    # 2. weights[:, j] <= weights[:, i] (element-wise)
    # 3. At least one strict inequality in the above.
    
    is_dominated = np.zeros(n_items, dtype=bool)
    
    # O(N^2) loop is acceptable for N=100
    for i in range(n_items):
        # If item i is already dominated, skip
        if is_dominated[i]:
            continue
            
        pi = total_prizes[i]
        wi = weights[:, i]
        
        for j in range(n_items):
            if i == j:
                continue
                
            pj = total_prizes[j]
            
            # If j's prize is strictly less, it cannot dominate i
            if pj < pi:
                continue
                
            wj = weights[:, j]
            
            # Check if j has <= weights in all dimensions
            # Using np.all for element-wise comparison
            if np.all(wj <= wi):
                # Check for strict dominance
                # If pj > pi, it's strictly better in prize
                # If any wj < wi, it's strictly better in weight
                if pj > pi or np.any(wj < wi):
                    is_dominated[i] = True
                    break
        # Time guard (unlikely to trigger)
        if time.time() - start_time > 5.0:
            break
            
    # Apply Boost to Non-Dominated Items
    # These items represent better trade-offs and should be preferred
    boost_factor = 2.0
    final_heuristic = base_heuristic.copy()
    final_heuristic[~is_dominated] *= boost_factor
    
    # Sanitize output
    # Ensure all values are positive and finite
    final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
    final_heuristic[final_heuristic <= 0] = 1e-9
    
    return final_heuristic
