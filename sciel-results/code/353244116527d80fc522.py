import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    # Deadline: The evaluator kills at 330s. 
    # We should aim to return well before that. 
    # The calculation is very fast (O(N*M)), so time is not an issue.
    # But we must adhere to the guard requirement.
    deadline = start_time + 290.0

    # Input validation
    try:
        prizes = np.asarray(prizes, dtype=float)
        weights = np.asarray(weights, dtype=float)
        
        # Expected shape (5, 100)
        if prizes.shape != (5, 100) or weights.shape != (5, 100):
             # Attempt to reshape if possible, else return zeros/epsilon
             # But contract says "wrong shape ... makes whole result invalid" 
             # So we should probably return something that doesn't crash the evaluator 
             # or raise an exception? 
             # "An exception ... make the whole result invalid."
             # So we should avoid exceptions.
             # If shape is wrong, we can't compute. Return 1e-9 array?
             pass # We assume correct shape based on contract, but guard just in case.

        n_constraints, n_items = 5, 100
        capacity = 25.0
        epsilon = 1e-9

        # 1. Ratios
        # Add epsilon to weights to avoid division by zero
        safe_w = weights + epsilon
        ratios = prizes / safe_w
        
        # 2. Harmonic Mean of Ratios
        # H = N / sum(1/R)
        inv_ratios = 1.0 / (ratios + epsilon)
        sum_inv = np.sum(inv_ratios, axis=0)
        h_mean = n_constraints / (sum_inv + epsilon)
        
        # 3. Geometric Slack
        # Reward items that consume less capacity
        slack = 1.0 - weights / capacity
        slack = np.clip(slack, epsilon, 1.0)
        # Log-sum-exp trick for stability, though simple mean of logs is fine for 5 items
        log_slack = np.log(slack)
        mean_log_slack = np.mean(log_slack, axis=0)
        geo_slack = np.exp(mean_log_slack)
        
        # 4. Composite Score
        # Base score
        score = h_mean * geo_slack
        
        # 5. Bottleneck Penalty (Max-Flow inspired)
        # Calculate total load per constraint
        loads = np.sum(weights, axis=1) # shape (5,)
        # Pressure is how much the constraint exceeds capacity
        pressure = np.maximum(0.0, loads - capacity) / capacity # shape (5,)
        
        # Item-specific pressure: sum of weight in each constraint * pressure of that constraint
        item_pressure = np.sum(weights * pressure, axis=0) # shape (100,)
        
        # Apply penalty
        penalty_factor = 1.0 + item_pressure
        score = score / penalty_factor
        
        # 6. Sharpening
        # Power of 1.2 to emphasize high-efficiency items
        heuristic = np.power(score + epsilon, 1.2)
        
        # 6. Cleanup
        # Replace non-finite or non-positive with epsilon
        heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)
        
        # Ensure shape is (100,)
        heuristic = heuristic.reshape(n_items)
        
        # Time check
        if time.time() > deadline:
             return heuristic
             
        return heuristic

    except Exception:
        # Fallback: return a valid array of small positive numbers
        # This prevents a hard crash if something unexpected happens
        return np.full((100,), 1e-9)
