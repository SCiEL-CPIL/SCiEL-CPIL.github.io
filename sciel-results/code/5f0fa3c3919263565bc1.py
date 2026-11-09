"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def _sanitize(scores):
    """Ensure scores are finite and of the correct shape."""
    if scores.shape != (2000,):
        scores = np.resize(scores, (2000,))
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, leave 20s safety margin
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-6)
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    f_best_safe = np.nan_to_num(f_best, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Candidate 1: Standard EI (baseline)
    z_ei = (f_best_safe - mu_safe) / sigma_safe
    ei_scores = (f_best_safe - mu_safe) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Candidate 2: UCB with aggressive exploration (xi=0.05, kappa=2.5)
    # For minimization, UCB is: mu - kappa * sigma + xi
    # But we want higher score to mean "evaluate next", so we invert:
    # score = -(mu - kappa*sigma) = -mu + kappa*sigma
    # Wait, the task says "higher meaning evaluate next".
    # In minimization, we want low mu.
    # Standard EI for minimization: E[f_best - f(x)] = (f_best - mu)Phi(z) + sigma phi(z) where z = (f_best-mu)/sigma
    # This is what EI above is. Higher EI means more expected improvement.
    
    # Let's try UCB variant: score = -mu + kappa * sigma
    # This encourages exploration (high sigma) and exploitation (low mu).
    kappa = 2.5
    ucb_scores = -mu_safe + kappa * sigma_safe
    
    # Candidate 3: PI (Probability of Improvement)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi_scores = norm.cdf((f_best_safe - mu_safe) / sigma_safe)
    
    # Candidate 4: Hybrid EI-UCB
    # Blend EI and UCB
    # Normalize scores to [0, 1] range for blending? Or just average?
    # Let's try a weighted sum.
    # To make them comparable, we can scale them.
    # But simple average might work if scales are similar.
    # EI is in objective units. UCB is in objective units.
    # Let's try: 0.5 * EI + 0.5 * UCB
    hybrid_scores = 0.5 * ei_scores + 0.5 * ucb_scores
    
    # Candidate 5: Tuned EI with small xi (explore more)
    # EI_xi = (f_best - xi - mu)Phi(z) + sigma phi(z), z = (f_best - xi - mu)/sigma
    xi = 0.01
    z_xi = (f_best_safe - xi - mu_safe) / sigma_safe
    ei_xi_scores = (f_best_safe - xi - mu_safe) * norm.cdf(z_xi) + sigma_safe * norm.pdf(z_xi)
    
    # Candidate 6: Lower Confidence Bound (LCB) for minimization
    # LCB = mu + kappa * sigma (we want to minimize this, so score = -LCB)
    # Wait, LCB is a lower bound on f(x). We want f(x) to be small.
    # So we want to minimize mu + kappa*sigma.
    # Score should be high when mu + kappa*sigma is low.
    # So score = -(mu + kappa*sigma) = -mu - kappa*sigma.
    # This is different from UCB above. Let's re-evaluate.
    # UCB for maximization: mu + kappa*sigma.
    # For minimization, we can use LCB: mu - kappa*sigma? No.
    # If we want to find the minimum, we want to explore where the lower bound is low.
    # Lower bound on f(x) is mu - kappa*sigma (if sigma is std dev).
    # So we want to minimize mu - kappa*sigma.
    # Score = -(mu - kappa*sigma) = -mu + kappa*sigma.
    # This is the same as UCB_scores above!
    # So UCB_scores is actually the LCB-based score for minimization.
    
    # Let's try a different exploration constant.
    kappa2 = 1.0
    lcb_scores = -mu_safe + kappa2 * sigma_safe
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        return _sanitize(ei_scores)
    
    # Select the best candidate based on the maximum score
    # We want to return the scores from the acquisition function that seems "best".
    # But "best" is subjective. We can try to pick the one that has the highest peak score,
    # or the one that is most balanced.
    # However, the task says "return 2000 finite scores". It doesn't say we can't pick one.
    # But the loop will pick the argmax of the returned scores.
    # So we want to return a set of scores that leads to a good next point.
    # Let's return the EI scores, but maybe with a slight modification.
    # Actually, let's just return the EI scores, as they are standard.
    # But wait, the directive says "COMPUTE BUDGET... return the best result found".
    # This implies we can do more work. But the acquisition function is called *once* per step.
    # So we can't do a loop *inside* the acquisition function to find the best point.
    # We just return scores.
    # So the "search" is in the choice of acquisition function.
    
    # Let's try to return a weighted combination of EI and LCB.
    # This is a common strategy: mix exploitation (EI) and exploration (LCB).
    # Weights: 0.7 EI, 0.3 LCB
    final_scores = 0.7 * ei_scores + 0.3 * lcb_scores
    
    # Ensure finite
    final_scores = _sanitize(final_scores)
    
    return final_scores
# EVOLVE-BLOCK-END
