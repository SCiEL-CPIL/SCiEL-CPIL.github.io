import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 10.0
    
    # Hypervolume-inspired acquisition:
    # 1. Compute Probability of Improvement (PI)
    # 2. Compute Expected Improvement (EI)
    # 3. Combine them into a "set-based" utility that rewards expanding the range of known good values
    # 4. Penalize by cost
    
    # Basic stats
    z = (f_best - mu) / (sigma + 1e-9)
    pi = norm.cdf(z)
    
    # Expected Improvement (standard form)
    ei = sigma * (z * pi + norm.pdf(z))
    
    # The "Hypervolume" idea in a 1D objective context with a single reference point (f_best)
    # is related to the volume of the region [mu, f_best] if mu < f_best.
    # However, since we are minimizing, the "dominated" volume is just the improvement.
    # A more sophisticated approach is to consider the *uncertainty* in the improvement.
    
    # Let's try a "Quantile-based" improvement as suggested by inspiration 1, 
    # but modified to be cost-aware.
    # The best score (2.7434) used a quantile-based probability of improvement.
    # Let's look at what that might be.
    # Perhaps ranking by the posterior probability that f(x) < f_best - delta?
    # Or simply using the CDF of the improvement distribution?
    
    # Let's try a blend of PI and EI, normalized to be scale-invariant.
    # EI has units of the objective, PI is unitless.
    # To make them comparable, we can normalize EI by the standard deviation of the objective 
    # or by f_best itself (if f_best is not 0).
    # But f_best can be 0.
    
    # Alternative: Use the "Upper Confidence Bound" on the *improvement*?
    # No, let's stick to the Hypervolume directive.
    
    # In multi-objective optimization, hypervolume is the volume dominated by the Pareto front.
    # Here, we have a single objective. The "front" is just {f_best}.
    # The volume dominated by a new point x is 0 if f(x) >= f_best, and f_best - f(x) if f(x) < f_best.
    # So the *expected* hypervolume contribution is exactly EI.
    
    # So why did the "quantile-based" approach score higher?
    # Maybe because EI is sensitive to the scale of the objective?
    # Or maybe because the "quantile" approach is more robust to outliers in the GP predictions?
    
    # Let's try a "robust" EI by capping the improvement.
    # Or let's try to normalize EI by the range of mu.
    
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    if mu_range < 1e-9:
        mu_range = 1.0
        
    # Normalize EI to [0, 1] roughly
    # Max possible EI is bounded by sigma * (some constant)
    # Let's just normalize by the max EI in the batch
    ei_max = np.max(ei)
    if ei_max < 1e-9:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = ei / ei_max
        
    # Now we have PI (0-1) and EI_norm (0-1).
    # The directive says: "estimates the expected volume ... normalizing by cost".
    # So score = (PI + EI_norm) / cost? Or (PI + EI_norm) * (1/cost)?
    # Or score = (PI + EI_norm) - lambda * cost_norm?
    
    # Let's try a multiplicative cost penalty, as it might be more "scale-invariant"
    # in terms of the trade-off.
    # Cost is in [1, 2].
    # 1/cost is in [0.5, 1].
    
    # Let's try: score = (PI + EI_norm) / cost
    # This penalizes expensive points more heavily when the acquisition value is low.
    
    # But wait, the previous best (2.7434) used "quantile-based probability of improvement".
    # Let's investigate what that means.
    # Maybe it's the probability that f(x) is in the bottom 10% of the predicted values?
    # Or the probability that f(x) < f_best? (That's PI).
    
    # Let's try a different structural change:
    # Instead of a linear combination, use a "soft-min" or "max" operation.
    # score = min(PI, EI_norm) / cost?
    # This would require both high probability AND high expected improvement.
    
    # Let's try the multiplicative cost penalty with a blend of PI and EI.
    # The blend: 0.5 * PI + 0.5 * EI_norm
    
    blend = 0.5 * pi + 0.5 * ei_norm
    
    # Cost penalty
    # Cost is 1 + mean(x).
    # We want to prefer low cost.
    # score = blend / cost
    
    score = blend / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    if time.time() > deadline:
        return pi / cost
        
    return score
