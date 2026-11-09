"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Strategy:
    1. Use LCB: score = mu - beta * sigma. This is structurally different from PI/EI 
       as it directly estimates the lower bound of the function value, favoring 
       points with high uncertainty (sigma) or low mean (mu).
    2. Cost Awareness: The cost is 1 + mean(x_scaled). Cheaper points are favored.
       We normalize the cost to [0, 1] range relative to the batch to avoid 
       scale issues. Let cost_norm = (cost - min_cost) / (max_cost - min_cost + eps).
       We penalize the score by a cost term. 
       Alternative: score = (mu - beta*sigma) / cost. 
       Since we want to maximize score, and cost is in denominator, lower cost is better.
       However, dividing by cost can be unstable if cost is small. 
       Let's use: score = (mu - beta*sigma) - lambda * cost_norm.
       Or better: Expected Improvement is usually divided by cost in cost-aware BO.
       For LCB, we can subtract a cost penalty.
       
       Let's refine the cost term. The cost is 1 + mean(x). 
       min cost is 1, max cost is 2.
       So cost is in [1, 2].
       We can just subtract a weighted cost: score = (mu - beta*sigma) - lambda * cost.
       
       But we need to balance the magnitude of (mu - beta*sigma) with cost.
       mu and sigma are in objective units. For Rosenbrock, values can be large (up to hundreds).
       Cost is ~1.5. So lambda needs to be small? Or we normalize cost?
       
       Let's try: score = (mu - beta*sigma) / (1 + cost).
       This is a common heuristic. It discounts the potential improvement by the cost.
       
    3. Dynamic Beta: 
       Early in optimization, we want to explore (high beta).
       Later, we want to exploit (low beta).
       We don't know the exact iteration count, but we can infer it from f_best?
       No, f_best is just a value. 
       However, the problem states "While total cost is below 30".
       We don't have the total cost passed in, only the cost of the candidates.
       Wait, the signature is acquisition(mu, sigma, f_best, cost).
       We don't have access to the iteration count or total spent cost directly.
       
       But we can estimate the "stage" of optimization based on f_best.
       If f_best is very high, we are early. If f_best is low, we are late.
       For Rosenbrock 2D, min is 0. Max is ~100+ depending on range.
       Range is [-2, 2]^2. 
       f(-2, -2) = (1 - 4)^2 + 100*(-4)^2 = 9 + 1600 = 1609.
       f(0, 0) = 1.
       f(1, 1) = 0.
       
       So f_best starts around 100-1000 and decreases to 0.
       We can map f_best to a beta value.
       High f_best -> High beta (explore).
       Low f_best -> Low beta (exploit).
       
       Let's define a decay function.
       beta = beta_max * exp(-k * (1 - f_best / f_max_est)) ?
       Or simpler: beta = beta_max * (f_best / f_max_est) ** power.
       
       Let f_max_est = 1000 (rough max for this domain).
       If f_best = 1000, beta = beta_max.
       If f_best = 0, beta = 0.
       
       Let's use beta_max = 3.0 (typical for LCB in 2D).
       beta = 3.0 * (f_best / 1000.0) ** 0.5 
       This decays slowly.
       
       Alternatively, since we don't know the exact start, let's just use a fixed beta 
       but make the cost penalty more significant?
       
       Actually, the current program uses PI. PI is very conservative. It only picks 
       points likely to beat f_best. It ignores the magnitude of improvement.
       LCB is more aggressive. It picks points with low lower bounds.
       
       Let's try a hybrid: 
       score = (mu - beta * sigma) / cost
       
       Why divide by cost? 
       If cost is 1.5, score is ~0.66 * (mu - beta*sigma).
       If cost is 2.0, score is ~0.5 * (mu - beta*sigma).
       This effectively makes the acquisition function 33% more aggressive for cheaper points.
       
       Let's refine the beta. 
       If we use a fixed beta, it might be too aggressive early on or too conservative late.
       Without iteration count, f_best is the best proxy.
       
       Let's try:
       beta = 2.0 + 2.0 * np.exp(-f_best / 50.0)
       If f_best is large (e.g., 100), exp(-2) is small, beta ~ 2.0.
       If f_best is small (e.g., 1), exp(-0.02) ~ 0.98, beta ~ 3.96.
       This is backwards. We want high beta when f_best is high (early).
       
       Let's use:
       beta = 2.0 + 2.0 * np.exp(-f_best / 100.0)
       Early (f_best ~ 1000): exp(-10) ~ 0, beta ~ 2.0.
       Late (f_best ~ 1): exp(-0.01) ~ 0.99, beta ~ 3.98.
       Still backwards.
       
       We want beta to be HIGH when f_best is HIGH.
       So:
       beta = 2.0 + 2.0 * (f_best / 1000.0)
       If f_best = 1000, beta = 4.0.
       If f_best = 0, beta = 2.0.
       
       This seems reasonable. It maintains a baseline exploration of 2.0 sigma, 
       and adds up to 2.0 sigma extra when the current best is poor.
       
       Let's check the cost division.
       score = (mu - beta * sigma) / cost
       
       Is dividing by cost the right move?
       Cost is 1 + mean(x). 
       If we want to minimize cost, we should prefer low cost.
       Dividing by cost achieves this.
       
       However, if mu - beta*sigma is negative (which it usually is for LCB in minimization 
       if we are not at the minimum), the score is negative.
       Higher score is better.
       So we want the least negative score, or the most positive.
       
       Let's verify the sign.
       Minimization.
       LCB = mu - beta*sigma.
       We want to find the minimum.
       So we want the smallest LCB.
       But the acquisition function returns scores where HIGHER = evaluate next.
       So we should maximize -LCB? Or minimize LCB?
       
       If we return LCB, the evaluator picks the MAX.
       So the evaluator would pick the point with the HIGHEST LCB.
       That's the opposite of what we want! We want the LOWEST LCB.
       
       So we must return -LCB or (beta*sigma - mu).
       
       Let's re-evaluate.
       Standard LCB for minimization: minimize mu - beta*sigma.
       Acquisition function for maximization: maximize -(mu - beta*sigma) = beta*sigma - mu.
       
       So score = (beta * sigma - mu) / cost.
       
       Let's check the current program.
       Current: PI = norm.cdf((f_best - mu) / sigma).
       PI is probability f(x) < f_best.
       Higher PI is better.
       This is correct.
       
       My new strategy:
       score = (beta * sigma - mu) / cost.
       
       Let's double check the cost scaling.
       cost is in [1, 2].
       Dividing by cost is a mild penalty.
       
       Let's refine beta.
       beta = 2.0 + 2.0 * np.clip(f_best / 1000.0, 0, 1)
       
       This ensures beta is between 2 and 4.
       
       Let's add a deadline guard.
       
    """
    start_time = time.time()
    
    # Ensure sigma is safe
    sigma_safe = np.clip(sigma, 1e-9, None)
    
    # Dynamic Beta
    # f_best is the lowest observed value.
    # For Rosenbrock in [-2,2]^2, max value is ~1600.
    # We normalize f_best by a large estimate of the max value.
    f_max_est = 2000.0
    f_norm = np.clip(f_best / f_max_est, 0.0, 1.0)
    
    # Beta starts high (4.0) when f_best is high (early) and decreases to 2.0 when f_best is low (late).
    beta = 2.0 + 2.0 * f_norm
    
    # LCB score (for minimization, we want to minimize mu - beta*sigma)
    # So we maximize beta*sigma - mu
    lcb_score = beta * sigma_safe - mu
    
    # Cost awareness: Divide by cost to prefer cheaper evaluations
    # Cost is 1 + mean(x_scaled). Range [1, 2].
    score = lcb_score / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - start_time > 320:
        pass

    return score
# EVOLVE-BLOCK-END
