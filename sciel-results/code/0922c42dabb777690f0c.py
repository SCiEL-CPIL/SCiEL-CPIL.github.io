# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = _T0 + 320.0

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    fb = float(f_best)

    # Pure minimum-variance acquisition:
    # Score = posterior standard deviation (higher sigma = more uncertain = explore there)
    # Add a tiny exploitation term to break ties and ensure finite, well-behaved scores.
    # This forces the GP to resolve uncertainty in unexplored regions,
    # preventing premature convergence to local noise minima in the 1D landscape.
    
    s = np.maximum(sigma, 1e-8)
    
    # Primary signal: explore where uncertainty is highest
    score = s.copy()
    
    # Tiny LCB tie-breaker: -mu * 1e-6 to ensure deterministic ordering when sigma is flat
    # This keeps scores finite and adds minimal exploitation bias
    score += -mu * 1e-6
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
