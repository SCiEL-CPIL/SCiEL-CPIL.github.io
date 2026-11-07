import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    
    # Input validation and conversion
    # prizes and weights are expected to be shape (5, 100)
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Basic sanity check for shape
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        # Fallback to simple ones if shape is unexpected, though contract guarantees shape
        return np.ones(100)
    
    # Calculate total prize per item (sum across constraints)
    total_prize = np.sum(prizes, axis=0)
    
    # Calculate the maximum weight per item across all constraints.
    # In MCKP, an item is constrained by its most expensive dimension.
    # Using max instead of sum captures the "bottleneck" resource.
    max_weight = np.max(weights, axis=0)
    
    # Avoid division by zero
    max_weight_safe = np.where(max_weight > 1e-9, max_weight, 1e-9)
    
    # Efficiency: Prize per unit of bottleneck weight
    efficiency = total_prize / max_weight_safe
    
    # Normalize efficiency to [0, 1] range for this instance
    # This makes the heuristic scale-invariant
    min_eff = np.min(efficiency)
    max_eff = np.max(efficiency)
    
    if max_eff > min_eff:
        norm_efficiency = (efficiency - min_eff) / (max_eff - min_eff)
    else:
        norm_efficiency = np.ones_like(efficiency)
        
    # Map normalized efficiency to heuristic range [1, 10]
    # ACO heuristics typically work well in a bounded range to prevent 
    # pheromone domination or vice-versa.
    # We use a power scaling to emphasize the difference between good and bad items.
    # alpha = 1.0 gives linear mapping. Higher alpha makes top items much more attractive.
    # Let's try alpha = 2.0 to strongly guide ants towards high-efficiency items.
    alpha = 2.0
    eta = 1.0 + 9.0 * (norm_efficiency ** alpha)
    
    # Ensure positivity and finite values
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return eta
        
    return eta
