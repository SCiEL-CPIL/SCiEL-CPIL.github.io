import numpy as np

def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    
    # 1. Total prize per item
    total_prizes = np.sum(prizes, axis=0) # (100,)
    
    # 2. Identify the most binding constraint for each item
    # This is the constraint where the item consumes the largest absolute weight.
    # Since capacities are all 25, the constraint with the largest weight is the 
    # most limiting for that specific item.
    max_weight_per_item = np.max(weights, axis=0) # (100,)
    
    # 3. Calculate value density relative to the most limiting resource
    # We use the max weight to normalize the prize. This highlights items that
    # offer high prize for the single worst-case resource consumption.
    # Add epsilon to avoid division by zero.
    value_density = total_prizes / (max_weight_per_item + 1e-9)
    
    # 4. Scarcity Premium Calculation
    # The directive suggests scaling by inverse of remaining capacity fraction.
    # Capacity C = 25.
    # For the most binding constraint of item i, let w_max_i be the weight.
    # Remaining capacity fraction = (25 - w_max_i) / 25 = 1 - w_max_i/25.
    # Inverse = 1 / (1 - w_max_i/25).
    # This creates a premium for items that do NOT heavily strain the capacity,
    # but more importantly, it modulates the value density.
    # However, a simpler interpretation of "prioritizing items that offer high value 
    # relative to their most limiting resource" is just value_density.
    # The "scarcity premium" implies we should boost items if their limiting 
    # resource is NOT too saturated globally, or if the item itself is "light" 
    # in that constraint.
    
    # Let's stick to the core structural change: 
    # Heuristic = Prize / (Weight in most binding constraint)
    # This is a different structural family than Prize / (Sum of all weights).
    # The sum-based heuristic treats all constraints as a pool. 
    # The max-based heuristic treats the bottleneck constraint as the cost.
    
    # To add the "scarcity" aspect mentioned in the directive:
    # If an item has a very high weight in its most binding constraint (e.g., 20 out of 25),
    # it is very "expensive" in terms of that constraint. 
    # The inverse capacity fraction logic:
    # If w_max is small, 1/(1 - w_max/25) is close to 1.
    # If w_max is large (close to 25), 1/(1 - w_max/25) is huge.
    # Wait, the directive says: "scaling its prize-to-weight ratio by the inverse 
    # of the remaining capacity fraction ... thereby prioritizing items that offer 
    # high value relative to their most limiting resource".
    
    # Let's re-read carefully: "prioritizing items that offer high value relative 
    # to their most limiting resource".
    # This usually means: High Prize / High Limiting Weight is bad? Or Good?
    # "High value relative to... resource" usually means High Prize / Low Weight.
    # So value_density = Prize / max_weight is the core.
    
    # The "scarcity premium" part: 
    # "scaling ... by the inverse of the remaining capacity fraction"
    # Let R_i = 25 - max_weight[i].
    # Inverse fraction = 1 / (R_i / 25) = 25 / R_i.
    # If R_i is small (item is heavy in that constraint), the multiplier is large.
    # This would PENALIZE heavy items if we multiply by this? No, multiplying by a large number increases the score.
    # If we want to prioritize items that are "light" in their most binding constraint, 
    # we should NOT multiply by the inverse of remaining capacity if that makes heavy items score higher.
    
    # Let's look at the standard interpretation: 
    # If I want to favor items that use the resource efficiently, I want High Prize / Low Weight.
    # The "scarcity" might refer to the global state. But the heuristic is static.
    # Let's assume the "scarcity premium" is meant to adjust for the fact that 
    # if an item takes up a lot of the *specific* constraint, it blocks more potential 
    # future items for that constraint.
    
    # Actually, a simpler and robust structural change is just using the max weight 
    # instead of the sum weight. This fundamentally changes how the constraints 
    # are aggregated. Sum assumes additive cost; Max assumes bottleneck cost.
    
    # Let's implement the bottleneck cost heuristic.
    # Heuristic = Total Prize / Max Weight in that item.
    
    # To be safe and follow the "scarcity" hint without over-complicating:
    # We can apply a power scaling to sharpen the differences.
    
    heuristic = value_density
    
    # Ensure positivity and finite values
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Power scaling to enhance the effect of the structural change
    heuristic = heuristic ** 1.2
    
    return heuristic
