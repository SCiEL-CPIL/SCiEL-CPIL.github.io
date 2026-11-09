import numpy as np
import time

def priority(adjacency):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin from 120s limit
    
    try:
        # Ensure we have a numeric array
        adj = np.asarray(adjacency, dtype=float)
        
        # Compute degrees
        degrees = adj.sum(axis=1)
        
        # Inverse degrees with smoothing
        inv_degrees = 1.0 / (degrees + 1.0)
        
        # Sum of inverse degrees of neighbors
        neighbor_inv_sum = adj @ inv_degrees
        
        # Penalty: High degree and blocking good neighbors is bad
        penalty = degrees + neighbor_inv_sum
        
        # Higher score is better
        scores = -penalty
        
        # Check time (should be instant)
        if time.time() > deadline:
             # Fallback to simple LDF if something went wrong or was slow
             return -degrees
             
        return scores
        
    except Exception:
        # Fallback to simple LDF if any error occurs
        try:
            adj = np.asarray(adjacency, dtype=float)
            degrees = adj.sum(axis=1)
            return -degrees
        except:
            # Absolute fallback
            return np.zeros(adjacency.shape[0])
