"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for Thompson Sampling
_thompson_beta = 1.0
_last_time = 0.0

def acquisition(mu, sigma, f_best, cost):
    global _thompson_beta, _last_time
    
    # 1. Time Guard: Ensure we return quickly to allow the main loop to proceed
    #    The task states "Each evaluation must finish within 300 s".
    #    Our computation is vectorized and fast, but we add a safety check.
    #    In a real tight loop, this might not be needed, but it's safe.
    current_time = time.time()
    if current_time - _last_time > 290:
        # Fallback to simple EI if we are dangerously close to timeout
        z = (f_best - mu) / (sigma + 1e-9)
        from scipy.stats import norm
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-6)
    _last_time = current_time

    # 2. Adaptive Beta for Thompson Sampling
    #    We estimate the number of evaluations performed so far.
    #    Initial cost is roughly 5 * (1 + 0.5) = 7.5.
    #    Total budget is 30.
    #    We use a heuristic: beta decreases as we spend more cost,
    #    shifting from exploration (high beta) to exploitation (low beta).
    #    However, since we don't know the exact count, we use the mean cost
    #    of the current candidates as a proxy for the "current stage" if we assume
    #    the loop is evaluating points in order of cost? No, cost is per point.
    #    
    #    Better heuristic: 
    #    The GP is fit on previous points. 
    #    If we are early on, sigma is high. 
    #    If we are late, sigma is low (if converged).
    #    
    #    Let's use a fixed beta that balances well for this specific problem size.
    #    Literature suggests beta ~ 1.0-2.0 for small BO loops.
    #    The "inspiration" noted that reducing cost penalty helps.
    #    EI divides by cost. This heavily penalizes expensive points (x close to 5).
    #    The global minimum of Three Hump Camel is at (0, 0) which is in the center 
    #    (scaled 0.5, 0.5), cost = 1 + 0.5 = 1.5.
    #    The local minimum is at (-0.3, 0.3) approx? 
    #    Actually, 3HC minima: (0,0) value 0. (-0.3, 0.3) value ~0.0007? No.
    #    Minima: (0,0) is global. (-0.3, 0.3) is local.
    #    Scaled coords:
    #    (0,0) -> (0.5, 0.5), cost 1.5.
    #    (-0.3, 0.3) -> ((-0.3+5)/10, (0.3+5)/10) = (0.47, 0.53), cost 1 + (0.47+0.53)/2 = 1.5.
    #    Wait, the cost function is 1 + mean(x_scaled).
    #    So cost depends on the average of the two coordinates.
    #    (0,0) -> mean 0.5 -> cost 1.5.
    #    (-0.3, 0.3) -> mean 0.5 -> cost 1.5.
    #    So the cost difference between the two minima is negligible.
    #    
    #    Why did reducing cost penalty help? 
    #    Maybe the "expensive" region is not the minimum, but the *search* is being
    #    biased away from high-cost regions where the GP might have high uncertainty 
    #    simply because it's unexplored?
    #    
    #    Thompson Sampling naturally balances this.
    #    score = beta * mu + sigma * N(0,1)
    #    We want to MINIMIZE f. So we want LOW mu.
    #    Therefore, the "value" to maximize in TS should be:
    #    sample = beta * (-mu) + sigma * N(0,1)  ? 
    #    Or rather, we sample from the posterior of the objective.
    #    f ~ N(mu, sigma^2).
    #    We want to find the x with the lowest f.
    #    So we sample f_sample = mu + sigma * Z.
    #    We pick the x with the MINIMUM f_sample.
    #    But the acquisition function returns a score where HIGHER is better.
    #    So we return -f_sample.
    
    #    How to incorporate cost?
    #    Standard Cost-aware TS:
    #    score = -f_sample / cost
    #    Or: score = -f_sample - lambda * cost
    #    
    #    The prompt says "Reducing the cost penalty bias helps".
    #    The seed program used EI / cost.
    #    If we use -f_sample / cost, it's similar.
    #    If we use -f_sample - small_penalty * cost, it's different.
    #    
    #    Let's try a hybrid:
    #    Use Thompson Sampling for the selection, but apply a very weak cost penalty.
    #    Or, simply use Thompson Sampling without cost penalty, relying on the GP
    #    to learn the landscape. The cost constraint is handled by the loop stopping
    #    at total cost 30. The acquisition just picks the best *next* point.
    #    If we ignore cost in the acquisition, we might pick expensive points early
    #    and run out of budget. 
    #    However, the initial 5 points are uniform.
    #    
    #    Let's use a cost-aware TS:
    #    sample = mu + sigma * Z
    #    score = -sample / (cost ** 0.5)  # Softer penalty than /cost
    
    #    Let's refine the beta.
    #    If sigma is large, we explore. If sigma is small, we exploit.
    #    The noise scale is 1e-6, so sigma is driven by GP uncertainty.
    
    #    Implementation:
    Z = np.random.randn(len(mu))
    samples = mu + sigma * Z
    
    # We want to minimize f. So lower samples are better.
    # Score should be higher for lower samples.
    base_score = -samples
    
    # Cost penalty
    # The previous finding: "Reducing the cost penalty bias helps avoid the cheaper 
    # local minimum in favor of the more expensive global minimum".
    # This implies the "cheaper" local minimum was being preferred by the cost-aware EI.
    # If the global minimum is "more expensive" (higher cost), then dividing by cost
    # was hurting it.
    # So we should REDUCE the influence of cost.
    
    # Let's try: score = base_score / (1 + 0.1 * (cost - 1))
    # Or simply: score = base_score - 0.1 * cost
    # Or: score = base_score / (cost ** 0.2)
    
    # Let's use a mild power penalty.
    cost_penalty = cost ** 0.2
    scores = base_score / cost_penalty
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
