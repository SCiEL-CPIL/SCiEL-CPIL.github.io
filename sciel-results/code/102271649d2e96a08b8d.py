import numpy as np
from scipy.stats import norm
import time

# Module-level variable to track start time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next).
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: If we are close to the 300s limit, return a safe fallback
    # The problem states "Each evaluation must finish within 300 s" and evaluator kills at 330s.
    # Since we don't control the loop, we can't stop it, but we can ensure we don't hang.
    # However, a simple check here is good practice. If elapsed > 290s, we could return 
    # a deterministic best guess to fail fast or just proceed quickly.
    # Given the vectorized nature, the computation is fast (<1ms). 
    # The risk is if the caller does heavy work. We just ensure our function is O(N) and fast.
    elapsed = time.time() - _start_time
    if elapsed > 295.0:
        # Fallback: Return something that prioritizes low cost and low mu
        # This is a safety net to avoid any potential complex branching if we were doing iterative methods.
        # For now, just proceed with the fast calculation.
        pass

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Sanitize inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
    
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # Assuming f(x) ~ N(mu(x), sigma(x)^2)
    # Let z = (f_best - mu(x)) / sigma(x)
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    
    # Calculate z
    # Avoid division by zero
    z = (f_best - mu) / sigma
    
    # Calculate EI
    # Phi is CDF, phi is PDF
    # Use scipy.stats.norm for accuracy
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-awareness
    # We want to maximize EI per unit cost.
    # score = EI / cost
    score = ei / cost
    
    # Adaptive Exploration via Temperature/Scaling?
    # Pure EI can be greedy. 
    # A common variation is to add a small UCB term or scale EI by a factor.
    # However, the prompt asks for a structural change. 
    # The current program uses LCB. EI is structurally different (probabilistic improvement vs confidence bound).
    
    # To make it more robust against local optima in early stages, 
    # we can blend EI with a pure exploration term based on sigma, 
    # but weighted by cost.
    # Let's stick to Cost-Weighted EI as the primary structural change.
    # If EI is 0 everywhere (converged), the scores will be 0. 
    # To avoid selecting arbitrary points when EI is 0, we might add a tiny epsilon or rely on the cost term.
    # If EI is 0, score is 0. The evaluator picks the "best" (highest) score. 
    # If all are 0, it might pick index 0. 
    # To break ties towards cheaper points, we already divide by cost.
    # But if EI is 0 for all, score is 0 for all. 
    # Let's add a small exploration bonus that decays as f_best improves? 
    # No, let's keep it pure EI/cost. It is a standard, strong baseline.
    
    # One refinement: If f_best is very high (early), EI is large. 
    # If f_best is close to optimum, EI is small.
    # This naturally shifts from exploration (high EI due to high uncertainty and distance) 
    # to exploitation (high EI only where local improvement is likely).
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
