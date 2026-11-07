import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Strategy:
    1. Compute the LCB score for minimization: lcb = mu - beta * sigma.
       - Lower lcb values are preferred (indicate potential for lower objective values).
       - beta controls the exploration-exploitation trade-off.
    2. We use a dynamic beta that starts high (exploration) and decays over time.
       Since we don't have direct access to the iteration count, we estimate progress
       from f_best. As f_best decreases (improves), we can reduce beta.
       Alternatively, we can use a fixed beta that works well for the problem scale.
       
       Given the problem: three hump camel 2D on [-5, 5]^2, min is 0.
       The function values are typically in a reasonable range.
       
       A common choice is beta = sqrt(2 * log(n)) where n is the number of evaluations,
       but we don't have n directly. 
       
       However, we can infer a proxy. If f_best is close to 0 (the minimum), we are
       likely in a later stage, so we can reduce exploration.
       If f_best is high, we are early, so we should explore more.
       
       Let's define a target minimum, say 0.
       progress = 1 - (f_best - 0) / (initial_guess - 0)
       This is tricky without initial_guess.
       
       Simpler approach: Use a fixed beta that balances well.
       For GP with Matern 5/2 and length scale 0.2 on [0,1]^2, the uncertainty
       decays relatively quickly.
       
       Let's try beta = 2.0 as a starting point. This is a common heuristic.
       
       To make it adaptive, we can scale beta by a factor that depends on f_best.
       Since the minimum is 0, and the maximum on this domain is around 27 (at corners),
       we can normalize f_best.
       
       Let f_min = 0, f_max = 30 (safe upper bound).
       normalized_f_best = np.clip(f_best / f_max, 0, 1)
       As f_best decreases, normalized_f_best decreases.
       We want beta to decrease as we improve.
       beta = beta_max * (1 - normalized_f_best * 0.5) ?
       Or simply beta = beta_max * (f_best / f_max + 0.5)?
       
       Actually, let's keep it simple and robust. A fixed beta of 1.5 or 2.0 is often
       effective. Let's try beta = 2.0.
       
    3. Cost-awareness: We want to prefer candidates with lower cost.
       The score for selection should be high for good candidates.
       LCB gives low values for good candidates.
       So, we can use: score = -lcb / cost
       Or: score = (f_best - lcb) / cost ? No, lcb can be negative.
       
       Let's stick to: score = -lcb / cost
       This means:
       - If lcb is very negative (very good potential), -lcb is large positive.
       - Dividing by cost penalizes expensive evaluations.
       
       Is this the best way to combine LCB and cost?
       Another option: score = -lcb * (1 / cost)
       Same thing.
       
       Let's consider if we should normalize the LCB first?
       The scale of mu and sigma matters.
       mu and sigma are in objective units.
       For three hump camel, values are O(10).
       So -lcb is O(10). Cost is O(1).
       The score is O(10). This seems reasonable.
       
       Let's refine the dynamic beta.
       If we use a fixed beta, it might be too exploratory early or too exploitative late.
       But since the loop runs until cost > 30, and starts with 5 points,
       we have roughly 25-30 evaluations.
       n is small.
       
       Let's try a beta that decays linearly with f_best.
       f_best starts high (~10-20) and ends near 0.
       beta = 1.0 + 2.0 * (f_best / 10.0)  # Just a guess
       If f_best = 10, beta = 3.0 (high exploration)
       If f_best = 0, beta = 1.0 (low exploration)
       
       Let's use a safer range.
       beta = 1.5 + 1.5 * np.clip(f_best / 10.0, 0, 1)
       
       Actually, simpler is better. Let's use a fixed beta = 2.0 first, as it's a
       standard choice and robust.
       
    4. Implementation details:
       - Ensure sigma is non-negative.
       - Ensure cost is positive.
       - Handle NaNs/Infs.
       
    5. Deadline guard:
       The computation is vectorized and very fast.
       Time should be negligible.
       We can add a simple time check, but it's likely unnecessary.
       However, the directive asks for an explicit deadline guard.
       We'll add a check, but since the operation is O(n), it will almost never trigger.
    """
    
    start_time = time.time()
    deadline = 300.0  # 300s limit, leave 30s margin
    
    # Sanitize inputs
    mu_safe = np.asarray(mu, dtype=np.float64)
    sigma_safe = np.asarray(sigma, dtype=np.float64)
    cost_safe = np.asarray(cost, dtype=np.float64)
    
    # Replace NaNs and Infs
    mu_safe = np.nan_to_num(mu_safe, nan=f_best, posinf=1e6, neginf=-1e6)
    sigma_safe = np.nan_to_num(sigma_safe, nan=0.0, posinf=1e6, neginf=0.0)
    cost_safe = np.nan_to_num(cost_safe, nan=1.0, posinf=1e6, neginf=1e-6)
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma_safe, 1e-8)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost_safe, 1e-8)
    
    # Dynamic beta based on f_best
    # f_best is a scalar
    f_best_val = float(f_best)
    
    # Normalize f_best: assume range [0, 30] for this problem
    # As f_best decreases, we want beta to decrease
    f_norm = np.clip(f_best_val / 30.0, 0.0, 1.0)
    beta = 1.0 + 2.0 * f_norm
    
    # Compute LCB: mu - beta * sigma
    lcb = mu_safe - beta * sigma_safe
    
    # Score: -lcb / cost
    # Higher score is better
    scores = -lcb / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    # Deadline guard (unlikely to trigger, but required)
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return a simple score based on mu
        scores = -mu_safe / cost_safe
        scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
