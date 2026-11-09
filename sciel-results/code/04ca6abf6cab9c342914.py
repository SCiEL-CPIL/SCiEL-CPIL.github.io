"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Strategy:
    Instead of using a deterministic acquisition function like LCB or EI, we use
    Thompson Sampling. We sample a single function value z(x) from the posterior
    distribution N(mu(x), sigma(x)^2) for each candidate point x.
    
    We then select the point that minimizes the sampled value, but we must account
    for the cost. A simple approach is to minimize the sampled value directly,
    but since we want to be cost-aware, we can interpret the "score" as the
    potential for finding a low value relative to the cost.
    
    However, standard Thompson Sampling picks argmin z(x). To make it cost-aware
    in a scoring framework where higher score = better next point, we can define
    the score as the negative of the sampled value, potentially adjusted for cost.
    
    A common cost-aware extension is to assume the "effective" value is z(x) / cost(x)
    or to prioritize points where the expected improvement per cost is high.
    
    But a more direct interpretation of "cost-aware" in the context of a budget
    constraint is that we want to maximize the probability of finding a new best
    within the remaining budget. 
    
    Let's stick to a robust Thompson Sampling variant:
    1. Sample z_i ~ N(mu_i, sigma_i^2) for all i.
    2. We want to find the point that is likely to be the global minimum.
    3. To incorporate cost, we can penalize high-cost points. A simple heuristic
       is to score points by: Score_i = -z_i / cost_i.
       
       Why? If z_i is low (good), -z_i is high (good score). Dividing by cost
       favors cheaper points. This is similar to the "efficiency" ratio.
       
       Alternative: Just use -z_i. This ignores cost in the selection but relies
       on the GP to learn the structure. However, the task explicitly provides
       cost and asks for a cost-aware acquisition. The previous program used
       improvement / cost.
       
       Let's try: Score_i = (f_best - z_i) / cost_i.
       This represents the "potential improvement" of the sampled value over the
       current best, divided by the cost. This is analogous to the expected
       improvement per cost, but using a sample instead of the expectation.
       
       This is a stochastic version of the current LCB/EI logic but with the
       exploration provided by the sampling process.
       
    Steps:
    1. Generate a random standard normal vector r of size N.
    2. Compute z = mu + sigma * r.
    3. Compute potential improvement: imp = f_best - z.
    4. Clip imp to be non-negative? No, if z > f_best, imp is negative.
       If we divide by cost, negative scores are bad. We want to maximize score.
       So we should probably take max(imp, 0) to avoid "rewarding" points that
       are sampled as worse than current best, unless we want to explore them
       for other reasons. But Thompson Sampling naturally explores by sometimes
       sampling low values even if mu is high.
       
       If we use max(imp, 0), we lose the signal from points where z > f_best.
       However, if z is sampled as very low (z << f_best), imp is large positive.
       If z is sampled as high, imp is negative.
       
       Let's use: Score_i = (f_best - z_i) / cost_i.
       Points with z_i < f_best get positive scores. Points with z_i > f_best
       get negative scores. The argmax will naturally pick a point with z_i < f_best
       if one exists in the sample. If all z_i > f_best, it picks the one closest
       to f_best (least negative) with lowest cost. This is reasonable.
       
    5. Handle numerical stability.
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return np.zeros(n, dtype=np.float64)
    
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Sample from the posterior
    # Using a fixed seed per call is not possible as we don't have state,
    # so we use global RNG. This introduces stochasticity which is the point.
    r = np.random.randn(n)
    z = mu + sigma_safe * r
    
    # Calculate score: Potential improvement per cost
    # We want to maximize this.
    # If z < f_best, f_best - z > 0.
    # If z > f_best, f_best - z < 0.
    
    improvement = f_best - z
    scores = improvement / cost_safe
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Optional: Add a small exploration bonus for high uncertainty points
    # that might not have been sampled low enough?
    # No, Thompson Sampling handles exploration via the sampling itself.
    # If sigma is high, z is likely to be far from mu, increasing the chance
    # of a low z (if mu is not extremely high) or a high z.
    # Actually, if sigma is high, the variance of z is high.
    # The probability of z < f_best is Phi((f_best - mu)/sigma).
    # This is exactly the probability of improvement.
    # So this is a valid stochastic acquisition.
    
    if time.time() > deadline:
        pass
        
    return scores
# EVOLVE-BLOCK-END
