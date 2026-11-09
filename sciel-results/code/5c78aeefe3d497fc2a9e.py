import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Deterministic Expected Improvement (EI) with dynamically annealed exploration parameter xi.
    
    The exploration parameter xi starts at 0.1 and decays linearly to 0.0 over the optimization steps.
    We estimate the current step from f_best: initially f_best is high (around 10-20 for Goldstein-Price),
    and it decreases as we find better points. We use a heuristic to map f_best to a step number.
    
    Actually, a simpler approach: we don't have access to the step count directly.
    But we can use the fact that f_best decreases. Let's use a different heuristic.
    
    Alternative: Use a fixed schedule based on the magnitude of sigma relative to the problem scale.
    Or better: since we can't track steps, use a simple decay based on how "small" f_best is.
    
    Let me think about this differently. The task says 30 evaluations total, 5 initial, so 25 optimization steps.
    We can't directly access the step counter. However, we can use a heuristic:
    - In early steps, f_best is large (e.g., > 5)
    - In late steps, f_best is small (e.g., < 2)
    
    Let's use: xi = 0.1 * max(0, 1 - f_best / 10)
    This gives xi=0.1 when f_best=0, and xi=0 when f_best>=10.
    But this is inverted. We want high exploration early (when f_best is high) and low exploration late (when f_best is low).
    
    So: xi = 0.1 * min(1, max(0, (f_best - 3) / 7))
    When f_best=10: xi = 0.1 * min(1, 7/7) = 0.1
    When f_best=3: xi = 0.1 * min(1, 0/7) = 0
    This gives a linear decay from 0.1 to 0 as f_best goes from 10 to 3.
    
    But this is fragile. Let's use a more robust approach.
    
    Actually, let's just use a small fixed xi that balances exploration and exploitation.
    The current program scores 1.39 with MC-EI + 0.1*sigma exploration bonus.
    
    Let's try deterministic EI with a small xi. The standard EI formula is:
    EI(x) = (f_best - mu - xi) * Phi((f_best - mu - xi)/sigma) + sigma * phi((f_best - mu - xi)/sigma)
    where Phi is the CDF and phi is the PDF of the standard normal.
    
    For minimization, we want to maximize improvement, so:
    EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    With xi:
    EI_xi(x) = (f_best - mu - xi) * Phi((f_best - mu - xi)/sigma) + sigma * phi((f_best - mu - xi)/sigma)
    
    Let's use a small xi like 0.01 to encourage slight exploration.
    """
    start_time = time.time()
    
    sigma = np.maximum(sigma, 1e-8)
    
    # Use deterministic EI with a small exploration parameter
    # We'll use xi = 0.01 as a fixed small value to balance exploration and exploitation
    xi = 0.01
    
    # Compute z = (f_best - mu - xi) / sigma
    z = (f_best - mu - xi) / sigma
    
    # Handle numerical issues: clip z to avoid overflow in exp
    z_clipped = np.clip(z, -10, 10)
    
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = exp(-z^2/2) / sqrt(2*pi)
    
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = (f_best - mu - xi) * Phi_z + sigma * phi_z
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Safety deadline guard
    elapsed = time.time() - start_time
    if elapsed > 250:
        # Return the best result found so far
        return ei
    
    return ei
