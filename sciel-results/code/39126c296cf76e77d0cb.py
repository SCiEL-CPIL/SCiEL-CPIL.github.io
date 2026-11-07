"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Batch-aware Lower Confidence Bound (LCB) with a repulsion term.
    
    Since we don't have coordinates, we approximate spatial diversity by
    penalizing points that are very confident (low sigma) and close to f_best
    in value, encouraging exploration of regions with higher uncertainty
    or regions where the mean is significantly lower.
    
    Actually, the prompt says "No coordinates are passed." So we cannot compute
    distance to the best point. We must rely on the mu and sigma values only.
    
    The directive suggests: "Replace Expected Improvement with a batch-aware 
    Lower Confidence Bound (LCB) that incorporates a simple repulsion term 
    based on the current best point's location, approximated via a distance 
    penalty in the score calculation, to enforce spatial diversity and prevent 
    redundant evaluations in the same basin."
    
    Since we don't have coordinates, we can't compute distance. However, we can
    use a heuristic: if the GP predicts a very low mean and very low variance,
    it might be a "basin" we've already found. But we don't know which points
    correspond to the best observed value.
    
    Let's stick to a robust LCB with a dynamic beta that adapts to the number
    of evaluations if possible, but we don't have that info either.
    
    Let's use a standard LCB: score = mu - beta * sigma.
    To add "diversity" without coordinates, we can use a penalty for points
    that are extremely likely to be the best (low sigma, low mu) to avoid
    re-evaluating the same spot? No, that's risky.
    
    Alternative interpretation: The "repulsion" is conceptual. Without coords,
    we can't implement spatial repulsion. So we must make the LCB itself more
    effective. A common improvement is to use a beta that balances exploration
    and exploitation.
    
    Let's try a slightly more aggressive exploration early on, but since we
    don't know the step, we can't do that.
    
    Let's try a hybrid: EI is known to be good. LCB is also good.
    Maybe a mix? Or a different parameterization.
    
    Let's look at the inspirations. They all scored 1.0.
    The current program is EI.
    
    Let's try LCB with beta = 2.0.
    
    Wait, the directive says "STRUCTURAL change".
    "Replace Expected Improvement with a batch-aware Lower Confidence Bound (LCB)..."
    
    Since I can't do spatial repulsion, I will implement a robust LCB.
    To make it "batch-aware" in a single-step context, I might consider that
    selecting the best point reduces the value of other nearby points.
    Without coordinates, I can't model this.
    
    I will implement a standard LCB with a tuned beta.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB: mu - beta * sigma
    # Higher score means lower predicted value with high uncertainty.
    # For minimization, we want low mu. LCB is mu - beta * sigma.
    # We want to maximize this score?
    # If mu is low, score is low (good for minimization).
    # If sigma is high, score is lower (more negative) due to -beta*sigma.
    # So maximizing (mu - beta * sigma) picks points with low mu and low sigma?
    # No, if sigma is high, -beta*sigma is very negative, so score is low.
    # We want to MAXIMIZE the score.
    # So we want HIGH (mu - beta * sigma).
    # This means we want HIGH mu? No, that's wrong.
    
    # Let's re-evaluate.
    # Objective: Minimize f.
    # GP predicts mu, sigma.
    # LCB for minimization: L(x) = mu(x) - beta * sigma(x).
    # We want to find the point with the LOWEST LCB.
    # But the function returns scores where HIGHER means evaluate next.
    # So we should return -LCB = -mu + beta * sigma.
    # Or simply return beta * sigma - mu.
    
    # Let's check EI.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z).
    # If mu is low, f_best - mu is large (if f_best is higher), EI is high.
    # So EI returns high scores for low mu.
    
    # For LCB:
    # We want to select x that minimizes mu - beta * sigma.
    # So score = -(mu - beta * sigma) = -mu + beta * sigma.
    
    beta = 2.0
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
