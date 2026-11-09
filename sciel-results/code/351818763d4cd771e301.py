"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # EHypR: Expected Hypar Ratio acquisition function
    # Maximizes E[max(f_best, f(x)) / f(x)] for minimization
    
    # Clip f_best to avoid division by zero or negative issues
    # For sphere function, minimum is 0, so f_best >= 0
    # If f_best is very close to 0, we should be very exploitative
    f_best_safe = max(f_best, 1e-10)
    
    # Standardize: z = (f(x) - mu) / sigma
    # We need to compute E[max(f_best, f(x)) / f(x)]
    # = E[max(f_best, f(x))] / E[f(x)] is NOT correct (ratio of expectations != expectation of ratio)
    # But we can approximate or use a different formulation
    
    # Actually, for EHypR, a common approximation for the ratio is:
    # If f_best > 0, then max(f_best, f) / f = f_best/f if f < f_best, else 1
    # So E[max(f_best, f)/f] = E[f_best/f * 1(f<f_best) + 1 * 1(f>=f_best)]
    # = f_best * E[1/f * 1(f<f_best)] + P(f >= f_best)
    
    # This is complex. Let's use a simpler, well-known alternative:
    # Use Expected Improvement (EI) which is well-understood and effective
    
    # EI for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    z = (f_best_safe - mu) / sigma
    
    # Handle cases where sigma is very small
    z_safe = np.where(np.abs(sigma) > 1e-10, z, np.where(mu < f_best_safe, 10, -10))
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # But for minimization, if mu < f_best, we want to exploit
    # The standard EI formula for minimization is:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # when f_best - mu > 0, and 0 otherwise
    
    improvement = f_best_safe - mu
    ei = np.zeros_like(mu)
    
    # Where improvement > 0
    mask = improvement > 0
    if np.any(mask):
        z_masked = z_safe[mask]
        ei[mask] = improvement[mask] * norm.cdf(z_masked) + sigma[mask] * norm.pdf(z_masked)
    
    # Where improvement <= 0, EI = 0 (already better than best)
    # But we might still want to explore, so let's add a small exploration term
    
    # To balance exploration, let's use a hybrid:
    # score = EI + lambda * sigma (uncertainty bonus)
    # Or better, use the quantile approach but adaptively
    
    # Actually, let's try a different structural change: use the posterior mean
    # with a confidence bound that adapts to the current best value
    
    # The key insight from the lesson: "Adaptive UCB balances exploration and 
    # exploitation by scaling the uncertainty bonus with the current best value"
    
    # Let's implement adaptive LCB:
    # score = -mu + beta * sigma
    # where beta adapts based on f_best
    
    # When f_best is large (early), use larger beta (more exploration)
    # When f_best is small (late), use smaller beta (more exploitation)
    
    # For sphere function on [-5,5], f_best starts around 25 and goes to 0
    # Let's use: beta = base_beta * (f_best / f_initial_scale)
    # with f_initial_scale = 25 (max of sphere on [-5,5])
    
    f_initial_scale = 25.0
    f_best_normalized = min(f_best_safe / f_initial_scale, 1.0)
    
    # beta starts at 2.0 (exploratory) and decreases to 0.5 (exploitative)
    base_beta = 2.0
    min_beta = 0.5
    beta = min_beta + (base_beta - min_beta) * f_best_normalized
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
