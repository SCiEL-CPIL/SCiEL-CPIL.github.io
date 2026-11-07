# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses a hybrid approach: Expected Improvement (EI) with an adaptive
    exploration bonus that decays as f_best improves (approaches 0).
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardize for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration: UCB-like term that shrinks as we get closer to optimum
    # f_best is negative (Weierstrass min is 0, we minimize, so f_best < 0 initially)
    # As f_best -> 0, we want less exploration.
    # Let's use a term that depends on the magnitude of improvement or just a simple decay.
    # A common trick is to use the current best value itself as a proxy for progress.
    # If f_best is very negative (far from 0), explore more. If close to 0, exploit.
    
    # Let's try: beta * sigma * log(1 + |f_best| / scale) or similar?
    # Or simpler: The standard EI already balances this somewhat, but EI can get stuck.
    # Let's add a small UCB term that is scaled by how far we are from the known minimum (0).
    # Since min is 0, and f_best <= 0, let progress = 1 - abs(f_best)/max_abs_f_best? 
    # We don't know max_abs_f_best.
    
    # Alternative: Use a constant beta but tuned for this problem.
    # The previous best was EI (score 1.0). Adaptive UCB improved it to 3.02.
    # Let's implement the "Adaptive UCB" strategy mentioned in the knowledge.
    # "Adaptive UCB shifts from exploration to exploitation as f_best improves"
    
    # Let's define a beta that depends on f_best.
    # If f_best is close to 0, beta should be small.
    # If f_best is far from 0, beta should be larger.
    
    # Weierstrass function values in [-0.5, 0.5] range are typically small.
    # f_best starts around -0.5 to -0.1 and goes to 0.
    
    # Let's use: beta = 0.5 * (1 + abs(f_best))
    # This increases exploration when f_best is far from 0 (large abs value)
    # and decreases it as f_best approaches 0.
    
    beta = 0.5 * (1.0 + abs(f_best))
    
    # UCB term
    ucb = mu + beta * sigma_safe
    
    # Combine EI and UCB? Or just use UCB?
    # The knowledge says "Adaptive UCB ... Measured: 2.155 -> 3.02".
    # This suggests UCB with adaptive beta was better than fixed UCB.
    # Let's try returning the UCB score directly, or a mix.
    # Usually, for minimization, we maximize mu + beta*sigma (if we want to find low values, we want high potential? No.)
    
    # Wait, we are MINIMIZING.
    # mu is the predicted mean. Lower is better.
    # sigma is uncertainty.
    # We want to pick points with low mu (good values) AND high sigma (uncertain).
    # So we want to MINIMIZE mu - beta*sigma? Or MAXIMIZE -mu + beta*sigma?
    # The function returns scores where HIGHER is better for selection.
    # So we want score = -mu + beta*sigma.
    
    # Let's check the EI formula again.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is positive when mu < f_best.
    # If mu is very low (good), EI is high.
    # If sigma is high, EI is higher.
    # So EI is a valid acquisition for minimization when we maximize it.
    
    # Let's stick to a modified EI or a pure Adaptive UCB.
    # Given the hint "Adaptive UCB ... improved", let's try Adaptive UCB.
    
    # Score = -mu + beta * sigma
    # Let's try this.
    
    score = -mu + beta * sigma_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
