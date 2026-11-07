import numpy as np

def heuristics(prizes, weights):
    # prizes: (5, 100)
    # weights: (5, 100)
    
    # Calculate total prize per item
    total_prize = np.sum(prizes, axis=0)
    
    # Calculate max constraint utilization per item
    # Capacity is 25 for each constraint
    capacity = 25.0
    utilizations = weights / capacity  # (5, 100)
    max_util = np.max(utilizations, axis=0)  # (100,)
    
    # Heuristic: reward high prize, penalize high utilization
    # Items that use a lot of any single constraint's capacity are less flexible
    heuristic = total_prize / (1.0 + max_util)
    
    # Ensure positivity
    epsilon = 1e-9
    heuristic = np.where(heuristic > 0, heuristic, epsilon)
    
    # Normalize to mean 1 to balance with initial pheromone (which starts at 1)
    mean_h = np.mean(heuristic)
    if mean_h > 0:
        heuristic = heuristic / mean_h
        
    return heuristic
