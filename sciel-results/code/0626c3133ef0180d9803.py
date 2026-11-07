import numpy as np

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Strategy:
    Prioritize items with high total prize relative to their maximum resource consumption
    across any single constraint. This penalizes items that are 'heavy' in any one dimension,
    which helps avoid creating tight bottlenecks in specific constraints that prevent packing
    other items.
    """
    # Calculate total prize for each item (sum across the 5 prize components)
    # prizes shape: (5, 100) -> prize_sum shape: (100,)
    prize_sum = np.sum(prizes, axis=0)
    
    # Calculate the maximum weight for each item across the 5 constraints
    # weights shape: (5, 100) -> max_weight shape: (100,)
    max_weight = np.max(weights, axis=0)
    
    # Compute heuristic: Prize / MaxWeight
    # Add a small epsilon to denominator to avoid division by zero
    # If an item has 0 weight in all constraints, it's free, so it should have very high heuristic.
    # However, the problem states weights are uniform in [0,1], so 0 is possible but unlikely for all 5.
    # If max_weight is 0, the item is free. We should give it a high score.
    # Using 1e-9 as epsilon.
    eps = 1e-9
    heuristic = prize_sum / (max_weight + eps)
    
    # Ensure all values are positive as required by the ACO logic 
    # (The task says "Heuristic entries that are not positive become 1e-9", 
    # but it's safer to return positive values directly if we can control it, 
    # although the ACO code might override. We'll clamp to 1e-9 to be safe.)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
