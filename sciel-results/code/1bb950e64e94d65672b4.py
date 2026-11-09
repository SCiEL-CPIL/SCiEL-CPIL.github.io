"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290  # 300s limit, 10s safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using GP-LCB (Lower Confidence Bound).
    
    Strategy:
    1. Compute LCB: mu - beta * sigma.
       Since we are minimizing, we want low mu and high sigma (uncertainty).
       Standard LCB for minimization is mu - beta*sigma.
       We want to maximize this score (higher = better candidate).
       So score_raw = mu - beta * sigma.
       
    2. Cost Awareness:
       The cost is 1 + mean(x). We divide the raw score by the cost.
       However, simple division can be unstable if scores are negative or mixed.
       A more robust cost-aware approach for minimization is:
       score = (f_best - (mu - beta*sigma)) / cost  is not quite right because 
       LCB can be very negative.
       
       Let's stick to the standard cost-aware LCB formulation:
       We want to minimize f(x). The "value" of a point is its expected minimum.
       LCB(x) = mu(x) - beta * sigma(x).
       We prefer points with low LCB.
       To make it an acquisition function (higher is better), we can use:
       score = -LCB(x) / cost(x)
       Or, to ensure positive scores and better scaling:
       score = (f_best - LCB(x)) / cost(x) ? 
       
       Let's look at the standard EI again: EI = (f_best - mu)*Phi(z) + sigma*phi(z).
       EI is always >= 0.
       LCB can be negative.
       
       Alternative: Use a "Cost-Adjusted LCB" where we maximize:
       score = (mu - beta * sigma) / cost  <-- This is wrong, we want low mu.
       
       Correct LCB for minimization:
       We want to minimize y.
       Lower bound on y is mu - beta*sigma.
       We want to select the x that minimizes this lower bound.
       So the "desirability" is - (mu - beta*sigma) = -mu + beta*sigma.
       
       Let's use: score = (-mu + beta * sigma) / cost.
       
       Wait, if mu is very low (good), -mu is high (good).
       If sigma is high (uncertain), beta*sigma is high (good).
       Dividing by cost penalizes expensive points.
       
       We need to choose beta. A common heuristic is beta = 2 or 3, or adaptive.
       Since we don't have the number of iterations easily accessible in a stateless function 
       (unless we infer from cost? No, cost is per point), we can use a fixed beta 
       or derive it from the data provided.
       
       Actually, the inputs are mu, sigma, f_best, cost for 2000 candidates.
       We can estimate a reasonable beta based on the scale of sigma.
       Or simply use a fixed beta like 2.0 or 3.0.
       
       Let's try beta = 2.5.
       
       To ensure the scores are finite and well-behaved:
       1. Clip sigma to a small positive value.
       2. Compute raw_lcb = -mu + beta * sigma.
       3. Divide by cost.
       4. Handle non-finite values.
       
       One issue: If mu is large positive and sigma is small, raw_lcb is very negative.
       If mu is large negative, raw_lcb is very positive.
       This seems correct for minimization.
       
       Let's refine the cost normalization.
       Cost is in [1, 2].
       
       To make the acquisition more robust, we can normalize the LCB scores 
       to have a mean of 0 and std of 1 before dividing by cost? 
       No, that loses the absolute scale which might be important for f_best context.
       
       Let's stick to: score = (-mu + beta * sigma) / cost.
       
       However, EI is generally more stable because it is bounded below by 0.
       LCB can have very large positive values if mu is very negative.
       
       Let's try a hybrid or a specific variant: 
       "Cost-Effective LCB"
       
       Another option: Use the "Probability of Improvement" (PI) or "Expected Improvement" (EI) 
       but with a different cost weighting.
       
       The current program uses EI / cost.
       The inspiration suggests LCB might be better.
       
       Let's implement LCB with a dynamic beta.
       A simple dynamic beta: beta = 1 + 2 * log(1 + len(mu)) ? No, len(mu) is always 2000.
       
       Let's use a fixed beta = 2.0.
       
       To prevent extreme values from dominating, we can clip the final score.
       
       Check deadline:
       if time.time() > _DEADLINE:
           return np.zeros_like(mu)
    """
    # Deadline check
    if time.time() > _DEADLINE:
        return np.zeros(2000, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)

    # LCB for minimization: mu - beta * sigma
    # We want to minimize this, so we maximize its negative: -mu + beta * sigma
    beta = 2.5
    
    raw_score = -mu + beta * sigma
    
    # Normalize by cost
    score = raw_score / cost
    
    # Clean up non-finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Optional: Clip extreme scores to prevent numerical issues in the loop
    # The scale of Shekel is around -10 to 10. 
    # -mu can be up to 10. beta*sigma can be large if sigma is large.
    # Let's clip to [-100, 100] to be safe, though normalization by cost (1-2) keeps it similar.
    score = np.clip(score, -100, 100)
    
    return score
# EVOLVE-BLOCK-END
