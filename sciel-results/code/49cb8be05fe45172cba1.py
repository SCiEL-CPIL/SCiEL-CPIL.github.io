import numpy as np
def heuristics(distance_matrix, coordinates, demands, capacity):
    # demands is a float array of length N
    # distance_matrix is N x N
    # We want to penalize high demand nodes
    # heuristic[i, j] = 1 / (dist[i, j] * (1 + demand[j]))
    
    # Ensure demands is numpy array
    demands = np.asarray(demands, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    # Calculate denominator
    # (1 + demands) has shape (N,)
    # distance_matrix * (1 + demands) has shape (N, N) with column-wise scaling
    denom = distance_matrix * (1.0 + demands)
    
    # Avoid division by zero
    # The task says "Heuristic entries that are not positive become 1e-9"
    # Distance is usually positive. Demand is 1-9. So denom is positive.
    # But just in case, handle zeros or negatives.
    safe_denom = np.where(denom > 0, denom, 1e9) # If denom <= 0, make it large so heuristic is small? 
    # Actually, the prompt says "Heuristic entries that are not positive become 1e-9".
    # This implies we calculate the heuristic, and then clip.
    
    heuristic = 1.0 / safe_denom
    
    # Clip non-positive or non-finite values to 1e-9
    # The prompt says: "Heuristic entries that are not positive become 1e-9"
    # It doesn't explicitly say what happens to negative distances, but distances should be >= 0.
    # If distance is 0, denom is 0, 1/0 is inf. inf is not positive? No, inf is positive.
    # But 1/0 is problematic.
    # Let's follow the instruction strictly: calculate, then check positivity.
    
    # Calculate raw heuristic
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_heuristic = 1.0 / (distance_matrix * (1.0 + demands))
        
    # Replace non-positive or non-finite with 1e-9
    # "not positive" means <= 0.
    # "non-finite" means inf or nan.
    # The prompt says "Heuristic entries that are not positive become 1e-9".
    # It also says "An exception, non-finite values ... make the whole result invalid".
    # Wait, "Heuristic entries that are not positive become 1e-9" is a rule for the heuristic function.
    # "An exception, non-finite values or a wrong shape make the whole result invalid" refers to the final result of the evaluation? Or the heuristic?
    # "The heuristic is computed once per instance... An exception, non-finite values or a wrong shape make the whole result invalid."
    # This implies if `heuristics` returns non-finite values, the whole run is invalid.
    # So we MUST ensure the returned matrix has no non-finite values.
    # And "Heuristic entries that are not positive become 1e-9" suggests we should enforce this.
    
    # So:
    # 1. Calculate 1 / (dist * (1+dem))
    # 2. If value is not positive (<= 0) or not finite, set to 1e-9.
    
    mask = (raw_heuristic > 0) & np.isfinite(raw_heuristic)
    heuristic = np.where(mask, raw_heuristic, 1e-9)
    
    return heuristic
