import numpy as np
import time

def heuristics(demands, capacity):
        start_time = time.time()
        n = len(demands)
        # Compute all pair sums
        pair_sums = demands[:, None] + demands[None, :]
        
        # Calculate residual space for valid pairs
        # If pair_sums > capacity, it's invalid.
        # We want high heuristic for low residual (high pair sum).
        
        # Create a mask for valid pairs
        valid_mask = pair_sums <= capacity
        
        # Initialize heuristic matrix with 1e-9
        H = np.full((n, n), 1e-9, dtype=float)
        
        # Calculate residual for valid pairs
        # residual = capacity - pair_sums
        # For invalid pairs, residual is negative, but we don't use them.
        residual = capacity - pair_sums
        
        # Apply heuristic formula for valid pairs
        # H = 1 / (1 + residual)
        # We only apply where valid_mask is True
        H[valid_mask] = 1.0 / (1.0 + residual[valid_mask])
        
        # Ensure no non-positive values (shouldn't happen with this formula for valid pairs)
        H[H <= 0] = 1e-9
        
        # Deadline check (trivial but required)
        if time.time() - start_time > 290:
             # Fallback to a simple heuristic if we somehow take too long
             # This branch is effectively unreachable for N=120
             return np.where(pair_sums <= capacity, pair_sums / capacity, 1e-9)
             
        return H
