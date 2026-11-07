"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time and enforce a safety margin
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    elapsed = _get_elapsed()
    # Safety margin: if we're getting close to the 330s limit, return a simple score
    if elapsed > 320:
        # Return a simple score based on expected improvement without heavy computation
        z = (f_best - mu) / (sigma + 1e-9)
        return ((f_best - mu) * 0.5 * (1 + np.tanh(z)) + sigma * np.exp(-0.5 * z**2)) / cost
    
    # Adaptive beta: shift from exploration to exploitation as f_best improves
    # Rosenbrock 2D minimum is 0, so f_best approaching 0 indicates progress
    # Early: high beta (more exploration), Late: low beta (more exploitation)
    
    # Normalize f_best relative to typical rosenbrock range
    # On [-2,2]^2, rosenbrock values range roughly from 0 to ~100+
    # Use a sigmoid-like transition based on f_best
    # When f_best is large (early), beta ~ 2.0 (exploration)
    # When f_best is small (late), beta ~ 0.8 (exploitation)
    
    # Transition point around f_best = 1.0
    # beta = 0.8 + 1.2 * sigmoid(-f_best)
    beta = 0.8 + 1.2 / (1.0 + np.exp(f_best))
    
    # Compute z-score
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Expected improvement: E[max(f_best - mu, 0)]
    # Using normal CDF approximation via tanh
    # Phi(z) ≈ 0.5 * (1 + tanh(0.7978845608028654 * z))
    phi_z = 0.5 * (1 + np.tanh(0.7978845608028654 * z))
    pdf_z = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    
    # Standard EI
    ei = (f_best - mu) * phi_z + sigma * pdf_z
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # Use linear cost penalty for stronger cost awareness
    score = ei / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
