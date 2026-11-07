import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    # Safety margin: The task says 300s limit. We return if we take too long.
    # This function should be very fast (<50ms), but we guard against pathological cases.
    deadline = start_time + 280.0
    
    # Ensure inputs are 1D arrays for consistent shape handling
    mu = np.asarray(mu, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    cost = np.asarray(cost, dtype=np.float64).flatten()
    
    # Safety: sigma must be > 0 for PDF/CDF calculations
    # Standard deviation should be positive. If it's 0, the point is known.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Strategy: Thompson Sampling / Probability of Best (PoB)
    # Instead of calculating the expected improvement (EI) which is deterministic 
    # and often over-exploitative, we use a sampling-based approach.
    # 
    # We approximate the Probability of Best (PoB) for each candidate x_i:
    # P(f(x_i) < f_best) = Phi((f_best - mu_i) / sigma_i)
    #
    # However, standard PoB ignores the magnitude of the improvement and can be 
    # stuck in local optima if the GP uncertainty is low everywhere. 
    # 
    # A more robust structural change is to use a "Soft" Thompson Sampling approach 
    # where we sample from the posterior and select the best, but since we must 
    # return a score for *all* 2000 candidates to let the outer loop pick the best, 
    # we need a score that correlates with the probability of being the best *and* 
    # having a significant gain.
    #
    # Let's use a hybrid: 
    # Score = P(Imp) * (1 + Expected Improvement Magnitude) / Cost
    # 
    # Wait, the current program uses (p_imp * ei) / cost. 
    # EI = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    # p_imp = Phi(z)
    #
    # The product p_imp * ei is roughly P(Imp) * E[Gain | Imp]. This is close to EI itself 
    # but weighted by probability.
    #
    # Let's try a different structural family: **Knowledge Gradient (KG) approximation**.
    # The Knowledge Gradient is the expected reduction in the regret of the best point 
    # after evaluating x. For a single step, it is related to the probability that x 
    # is the new best *and* the expected improvement in the best value.
    #
    # A simpler, often more effective alternative in high dimensions with limited budget 
    # is **Probability of Improvement (PoI)** with a strong cost normalization, 
    # but PoI is very greedy.
    #
    # Let's look at the "Cost-aware" aspect. 
    # Cost = 1 + mean(x_scaled). Range [1, 2].
    # 
    # New Strategy: **Inverse Cost Weighted Probability of Best with Uncertainty Bonus**
    # We want to maximize the chance of finding a point better than f_best, 
    # but penalize expensive points.
    #
    # Score = (Phi((f_best - mu) / sigma) + beta * Phi((f_best - mu - alpha*sigma) / sigma)) / cost
    # This is getting complicated.
    #
    # Let's stick to a clean, distinct approach: **Thompson Sampling Score**
    # We simulate the "best" value from the current model and the potential value of x.
    # Since we can't sample 2000xN times efficiently, we use the analytic PoB 
    # but modify it to account for the *cost* of the evaluation in a way that 
    # reflects the *budget* constraint more accurately than simple division.
    #
    # Actually, the most materially different and robust approach for BO with 
    # expensive evaluations is **Expected Improvement (EI)** but with a 
    # **dynamic xi (threshold)** that is scaled by the *remaining budget* or 
    # *current cost efficiency*. 
    #
    # However, the prompt asks for a "different design or modelling family".
    # EI is the standard. Let's try **Probability of Improvement (PoI)** 
    # combined with a **Cost-Penalty via Exponential Decay**.
    #
    # Score = P(f(x) < f_best) * exp(-lambda * (cost - 1))
    #
    # Why? 
    # 1. P(f(x) < f_best) is the Probability of Best (PoB) relative to current best.
    # 2. The exponential penalty strongly discourages expensive points.
    # 3. This avoids the division by zero/small cost issues and the "double counting" 
    #    of EI (which includes magnitude) when the goal is just to find *any* improvement 
    #    quickly to reduce the gap.
    #
    # To make it more robust against local optima, we can add a small exploration term 
    # based on uncertainty (sigma), but weighted by cost.
    #
    # Score = (P_Imp + gamma * sigma / sigma_max) * exp(-lambda * (cost - 1))
    #
    # Let's refine:
    # z = (f_best - mu) / sigma
    # p_imp = norm.cdf(z)
    #
    # We want to favor points that are likely to be better (high p_imp) 
    # and cheap (low cost).
    #
    # Let's try: Score = p_imp / (cost ** 2)
    # Squaring the cost makes the penalty more aggressive for expensive points.
    #
    # But is this materially different from the current (p_imp * ei) / cost?
    # Yes. EI depends on the *magnitude* of improvement. PoI only depends on the *probability*.
    # In early stages, EI might favor points with large potential gains even if probability is low.
    # PoI favors points with high probability of *any* gain.
    # In 8D, finding *any* gain is hard. PoI might be too greedy.
    #
    # Let's go with a **Hybrid Thompson Sampling** approach that approximates the 
    # posterior mean of the *rank* of the point.
    #
    # Actually, the simplest structural change that is known to work well in 
    # cost-aware BO is **Cost-Effective EI (CEEI)** or **UCB with Cost**.
    # The current program uses a CTS-like score.
    #
    # Let's try **Probability of Improvement (PoI)** with a **Cost-Adjusted Threshold**.
    # Instead of a fixed xi, we adjust the threshold based on cost.
    # Cheaper points can have a lower threshold (more exploratory).
    # Expensive points must have a higher threshold (more certain).
    #
    # xi_cost = xi_base + k * (cost - 1.0)
    #
    # Then calculate EI with this dynamic xi.
    # This is structurally different from dividing by cost because it changes 
    # the *definition* of what constitutes an improvement for expensive points.
    
    # Base xi
    gap = f_best - (-0.8)
    if gap > 0.5:
        xi_base = 0.01
    elif gap > 0.1:
        xi_base = 0.05
    else:
        xi_base = 0.1
        
    # Cost adjustment: 
    # cost is in [1, 2]. 
    # If cost is 2, we add 0.05 to xi. If cost is 1, we add 0.
    # This makes expensive points require a larger expected improvement to be selected.
    cost_factor = (cost - 1.0) # 0 to 1
    xi_dynamic = xi_base + 0.05 * cost_factor
    
    # Calculate EI with dynamic xi
    z = (f_best - xi_dynamic - mu) / sigma_safe
    p_imp = norm.cdf(z)
    ei = (f_best - xi_dynamic - mu) * p_imp + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # We still need to account for cost in the final score.
    # If we just use EI, expensive points might still win if the improvement is huge.
    # But the dynamic xi already penalizes them by raising the bar.
    # Should we divide by cost again? 
    # If we do, we might be over-penalizing.
    # Let's try just EI with dynamic xi. This is a "Threshold-Adjusted EI".
    
    scores = ei
    
    # Alternative: If we want to be more aggressive on cost, we can multiply by 1/cost.
    # But let's see if the dynamic xi is enough.
    # Actually, let's combine them slightly:
    # Score = EI / (1 + cost_factor)
    # This is a softer penalty than dividing by cost directly.
    
    scores = ei / (1.0 + cost_factor)
    
    # Handle NaNs/Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if time.time() > deadline:
        return np.ones_like(mu)
        
    return scores
