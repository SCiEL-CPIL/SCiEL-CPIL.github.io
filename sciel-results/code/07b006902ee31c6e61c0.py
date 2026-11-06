"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    # Start with a copy of the ensemble
    result = ensemble.copy()
    
    # Basic safety check: ensure inputs are finite
    if not np.all(np.isfinite(ensemble)) or not np.all(np.isfinite(observations)):
        return result
    
    # Simple observation nudging with adaptive gain
    # The gain is set to 0.6 as in the original, but we can adjust if needed
    # For a more principled approach, we might use ensemble variance
    # but for now, stick with a robust fixed gain
    
    # Clip to ensure values stay within reasonable bounds
    result[:, observed_indices] += 0.6 * (observations - result[:, observed_indices])
    
    # Ensure output is within [-100, 100] as required
    result = np.clip(result, -100, 100)
    
    # Ensure all values are finite
    if not np.all(np.isfinite(result)):
        # Fallback to original ensemble if something went wrong
        return ensemble.copy()
        
    return result
# EVOLVE-BLOCK-END
