"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items using a Marginal Scarcity Heuristic.
    
    Strategy:
    1. Calculate the total prize for each item (sum across constraints).
    2. Estimate the "scarcity" of each constraint resource by looking at the 
       efficiency of the best items for that resource.
    3. Calculate the "opportunity cost" of each item: how much total prize 
       would be lost if we selected this item instead of the most efficient 
       items for the resources it consumes.
    4. Score items based on their prize relative to this opportunity cost.
    
    This moves away from simple prize/weight ratios to a model that accounts 
    for the competition between items for limited resources.
    """
    # prizes: (5, 100), weights: (5, 100)
    # Ensure inputs are numpy arrays for safety, though the contract implies they are
    prizes = np.asarray(prizes)
    weights = np.asarray(weights)
    
    # 1. Total prize per item (scalar per item)
    item_prizes = np.sum(prizes, axis=0) # shape (100,)
    
    # 2. Estimate Scarcity / Opportunity Cost
    # For each constraint c, we want to know the "value density" of the best items.
    # If an item uses weight w_ij in constraint j, it displaces other items.
    # The opportunity cost is roughly the sum over constraints of 
    # (weight_ij * max_efficiency_j).
    
    # Calculate efficiency for each item in each constraint: prize_ij / weight_ij
    # We need to be careful with division by zero.
    # Since weights are uniform [0,1], zeros are possible but unlikely to be exactly 0 
    # in a continuous uniform distribution, but let's be safe.
    # Actually, the problem says uniform [0,1], so 0 is possible.
    
    # To estimate the "scarcity" or "shadow price" of each constraint, 
    # we can look at the top performers for that specific constraint.
    # A simple proxy: For constraint j, the best items have high prizes[:,j] / weights[:,j].
    # Let's compute the efficiency matrix: (5, 100)
    # efficiency[i, j] = prizes[i, j] / (weights[i, j] + eps)
    
    eps = 1e-9
    efficiency = prizes / (weights + eps)
    
    # For each constraint j, find the maximum efficiency among all items.
    # This represents the "best alternative" available for that resource.
    # If an item consumes weight in a resource that has very high max efficiency,
    # it is "expensive" in terms of opportunity cost.
    
    max_eff_per_constraint = np.max(efficiency, axis=1) # shape (5,)
    
    # Now, calculate the opportunity cost for each item.
    # Opportunity Cost for item j = sum over constraints i of (weights[i, j] * max_eff_per_constraint[i])
    # This is a dot product: weights.T (100, 5) dot max_eff_per_constraint (5,) -> (100,)
    
    opportunity_cost = np.dot(weights.T, max_eff_per_constraint)
    
    # 3. Heuristic Score
    # We want to reward high prize and low opportunity cost.
    # Score = item_prizes / (opportunity_cost + eps)
    
    scores = item_prizes / (opportunity_cost + eps)
    
    # Ensure all scores are positive to avoid issues with the ACO probability calculation
    # The contract says: "Heuristic entries that are not positive become 1e-9"
    # But it's better to ensure they are positive if possible, or just return raw and let evaluator handle it?
    # "Heuristic entries that are not positive become 1e-9" is done by the evaluator.
    # However, if our calculation produces 0 or negative, they become 1e-9.
    # Our calculation: item_prizes >= 0, opportunity_cost >= 0. So scores >= 0.
    # If item_prize is 0, score is 0 -> becomes 1e-9.
    # If opportunity_cost is 0 (item has 0 weight in all constraints where max_eff is high? Unlikely), score is inf?
    # No, if opportunity_cost is 0, it means weights are 0. Then prize/0 is inf.
    # We should cap or handle this.
    
    # If an item has 0 weight in all constraints, it's free. It should be very attractive.
    # But inf is not finite. The contract says "non-finite values... make the whole result invalid".
    # So we must ensure finite values.
    
    # Let's check for potential infinities.
    # If opportunity_cost is very small, score is large.
    # We can clip the scores or add a larger epsilon to the denominator.
    
    # A safer denominator: opportunity_cost + 1.0 ?
    # Or just clip the final scores?
    # The ACO uses pheromone * heuristic. If heuristic is huge, it dominates.
    # Let's normalize or clip to a reasonable range?
    # The seed program returns raw ratios.
    # Let's just ensure no division by zero causes inf.
    
    # If opportunity_cost is 0, we add eps. But if prize is > 0, result is prize/eps which is huge.
    # Is that bad? It makes that item almost certainly picked.
    # If an item is free (0 weight), it should be picked.
    # But "non-finite" is invalid. 1e9 is finite.
    
    # Let's replace any non-finite values with a large finite number, e.g., 1e6.
    # Or better, use np.where to handle the division.
    
    # Actually, if weights are uniform [0,1], P(weight=0) is 0 in continuous dist, 
    # but in floating point, it's possible.
    
    # Let's use a more robust denominator.
    # denom = opportunity_cost + 1e-6
    # scores = item_prizes / denom
    
    # This avoids inf.
    
    scores = item_prizes / (opportunity_cost + 1e-6)
    
    # Ensure shape is (100,)
    return scores
# EVOLVE-BLOCK-END
