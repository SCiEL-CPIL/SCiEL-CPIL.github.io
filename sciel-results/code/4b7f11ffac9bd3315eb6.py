import numpy as np
import time

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for the ACO solver.
    Uses a geometric heuristic based on inverse squared distance to suppress long-range edges.
    """
    start_time = time.time()
    deadline = start_time + 10.0 # Safety margin well below 300s
    
    n = distance_matrix.shape[0]
    
    # Compute inverse squared distance
    # Add small epsilon to avoid division by zero (diagonal)
    # We want to penalize long distances heavily.
    # 1/d^2 decays faster than 1/d.
    
    d_safe = np.where(distance_matrix > 1e-9, distance_matrix, 1.0)
    
    # Structural change: Use inverse squared distance
    heuristic = 1.0 / (d_safe * d_safe)
    
    # Ensure diagonal is handled. The task says "Heuristic entries that are not positive become 1e-9".
    # Diagonal is 0 distance -> 1/0 is inf. We set d_safe to 1.0 for 0s, so 1/1 = 1.
    # But usually diagonal is 0. If distance is 0, d_safe is 1.0. Heuristic is 1.0.
    # This is positive.
    # However, ants don't pick themselves.
    
    # The task says: "Heuristic entries that are not positive become 1e-9"
    # Our heuristic is positive everywhere.
    
    # Check deadline
    if time.time() > deadline:
         # Return a safe default if we somehow took too long (unlikely for this simple op)
         return np.ones_like(distance_matrix)
    
    # Ensure no non-positive values
    heuristic[heuristic <= 0] = 1e-9
    
    # Normalize? The task doesn't require normalization, just relative attractiveness.
    # Larger is more attractive.
    
    return heuristic
