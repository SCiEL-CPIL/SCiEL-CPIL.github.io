import numpy as np

def heuristics(prizes, weights):
    # Calculate total weight for each item across all constraints
    total_weights = np.sum(weights, axis=0)
    
    # Add small epsilon to avoid division by zero
    eps = 1e-9
    
    # Sum-Weight Heuristic: Prize divided by total weight burden
    # This penalizes items that are heavy in any combination of constraints
    eta = prizes[0] / (total_weights + eps) # Wait, prizes is (5, 100). Which prize?
    
    # The problem says "prizes and weights, of shape (5, 100)".
    # Usually in MCKP, there is ONE prize per item, and multiple weights.
    # But here prizes is (5, 100). This implies 5 prize values per item?
    # Or is it 5 instances? No, "3 fixed instances with 100 items".
    # The function takes `prizes` and `weights`.
    # "prizes and weights, of shape (5, 100), are uniform in [0, 1]."
    # This is ambiguous. Does it mean 5 constraints, and for each constraint, there is a prize?
    # No, Knapsack has 1 prize per item.
    # Re-reading: "Item scores for ant colony optimization of the multiple-constraint knapsack".
    # "prizes and weights, of shape (5, 100)".
    # If there are 5 constraints, weights is (5, 100).
    # What are the 5 rows of prizes?
    # Maybe it's 5 different prize vectors? Or maybe the "prize" is also multi-dimensional?
    # "Every constraint has capacity 25".
    # "best prize of that iteration".
    # If prizes is (5, 100), maybe the total prize is the sum? Or the max?
    # Let's assume the prize for an item is the sum of its prize components, or the first row, or the average.
    # Given the symmetry of the problem statement "prizes and weights... shape (5,100)", it's highly likely that the prize is also a vector, or we should aggregate it.
    # However, "best prize of that iteration" suggests a single scalar score for the ant's solution.
    # If an ant picks items $I$, the total prize is likely $\sum_{i \in I} \text{prize}_i$.
    # If $\text{prize}_i$ is a vector, how is it summed?
    # Most likely, the "prize" is a scalar per item, and the shape (5, 100) for prizes is a mistake in my interpretation, OR the prize is the sum across the 5 dimensions.
    # Let's look at the current code: `ratios = prizes / (weights + eps)`.
    # This divides element-wise. So `ratios` is (5, 100).
    # Then it takes the harmonic mean across axis 0 (the 5 constraints).
    # This implies that for each item, we have 5 ratios, one per constraint.
    # This requires `prizes` to be aligned with `weights`.
    # So, for constraint $j$, we have prize $p_{ij}$ and weight $w_{ij}$.
    # This is unusual for standard MCKP (usually 1 prize, N weights).
    # But if the problem defines it this way, we follow the data shape.
    # The "total prize" of an ant's solution is likely the sum of the prizes of the selected items.
    # If prizes are vectors, maybe the score is the sum of all components?
    # Let's assume the heuristic should be based on the "efficiency" across all dimensions.
    
    # If we use Sum-Weight:
    # We need a single prize value per item to divide by total weight.
    # Let's assume the "prize" of an item is the sum of its prize components: `item_prize = np.sum(prizes, axis=0)`.
    # Then `eta = item_prize / (np.sum(weights, axis=0) + eps)`.
    
    # Alternatively, if the current code's element-wise division is correct, it treats each constraint independently.
    # The current code returns a scalar per item (after mean across axis 0).
    # So I must return a scalar per item.
    
    # Let's stick to the element-wise ratio approach but change the aggregation.
    
    eps = 1e-9
    ratios = prizes / (weights + eps)
    
    # Sum-Weight Heuristic Interpretation 1:
    # Aggregate prizes and weights separately?
    # If we sum prizes across constraints and sum weights across constraints:
    total_prize = np.sum(prizes, axis=0)
    total_weight = np.sum(weights, axis=0)
    eta = total_prize / (total_weight + eps)
    
    # Ensure positive finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
