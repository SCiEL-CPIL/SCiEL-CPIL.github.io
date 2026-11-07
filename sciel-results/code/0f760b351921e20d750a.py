import numpy as np

def heuristics(prizes, weights):
    # prizes, weights: (5, 100)
    # capacity per constraint = 25
    cap = 25.0
    C, N = weights.shape
    
    total_prize = prizes.sum(axis=0)  # (N,)
    
    # Resource scarcity per constraint: total weight / capacity
    # Higher means that constraint is more over-subscribed
    scarcity = weights.sum(axis=1) / cap  # (C,)
    
    # Lagrangian reduced cost: prize - sum_c scarcity_c * w_ic
    # This is a different modeling family: dual-based marginal value
    reduced = total_prize - (scarcity[:, None] * weights).sum(axis=0)
    
    # We want larger = more attractive, but reduced cost can be negative
    # Shift and scale so all positive
    # Use a soft positive transform
    heuristic = reduced
    
    # Ensure positive & finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
