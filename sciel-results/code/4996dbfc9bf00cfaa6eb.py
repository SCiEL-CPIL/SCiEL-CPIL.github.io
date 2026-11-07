import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we're close to the 330s limit, return a simple score
    # (this function itself is fast, but we guard just in case of cumulative overhead)
    
    sigma = np.maximum(sigma, 1e-9)
    
    # Use Upper Confidence Bound (UCB) with adaptive exploration
    # UCB is often more robust than EI for early exploration
    # We'll use a combination: UCB for exploration, EI for exploitation
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound with beta parameter
    # beta controls exploration vs exploitation
    # We want to explore more early on, exploit more later
    # Since we don't know the step, we'll use a fixed moderate beta
    # The Beale function has a global minimum at (3, 0.5) with value 0
    # and local minima, so exploration is important
    
    # Use a UCB with beta=1.0 as a good balance
    beta = 1.0
    ucb = mu + beta * sigma  # This is for minimization, so we want low mu and high sigma
    
    # Wait - for minimization, UCB should be:
    # We want to pick points where mu - beta*sigma is low (optimistic)
    # But the return value should be HIGHER for better candidates
    # So we want: score = -(mu - beta*sigma) = -mu + beta*sigma
    
    # Actually, let's use a hybrid: combine EI and UCB
    # EI already balances this, but let's try a different approach
    
    # Log Expected Improvement (LogEI) can be more robust
    # Or use Probability of Improvement (PI)
    
    # Let's use a practical combination:
    # score = EI + lambda * (uncertainty bonus)
    # where uncertainty bonus encourages exploration
    
    # For minimization, a good acquisition is:
    # score = -mu + beta * sigma  (UCB for minimization)
    # or EI which naturally balances
    
    # Let's use EI but with a small exploration boost for high-uncertainty regions
    # Actually, the standard EI for minimization is what we have above
    
    # Let's try UCB which tends to be more robust:
    # For minimization, we want to maximize: -mu + beta*sigma
    # But this can be too exploratory. Let's use a softer version.
    
    # Use a Gaussian Process-based UCB with adaptive beta
    # Based on the number of evaluations (we can infer from f_best changing)
    
    # Simple approach: use EI but add a small uncertainty bonus
    # This is similar to UCB with beta=0.5
    
    # Let's compute both and blend
    ei_score = ei
    ucb_score = -mu + 0.5 * sigma  # For minimization
    
    # Normalize to combine
    # But simpler: just use EI which is well-tested
    
    # Actually, let's use a different strategy:
    # Use EI for exploitation, but add a diversity/exploration term
    # based on how far the candidate is from the current best region
    
    # Since we don't have coordinates, we can't do spatial diversity
    # So let's stick with EI but tune it
    
    # Use EI with a small bonus for high sigma (uncertainty)
    # This makes it more like UCB
    
    # Final: use EI + 0.1 * sigma as a mild exploration boost
    score = ei + 0.1 * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
