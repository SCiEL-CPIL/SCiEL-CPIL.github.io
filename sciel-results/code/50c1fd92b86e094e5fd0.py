import numpy as np
import time
from scipy.stats import norm

# Track elapsed time for deadline guard
_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin from 330s

def acquisition(mu, sigma, f_best):
    # Check time
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # If we are running out of time, return a safe fallback
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        return np.nan_to_num(-mu, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle edge cases
    if mu.size == 0:
        return np.array([])
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    # EI is defined for minimization: E[max(f_best - f(x), 0)]
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Compute UCB Score
    # For minimization, LCB = mu - kappa * sigma.
    # We want to maximize the "score", so Score_UCB = -LCB = -mu + kappa * sigma.
    
    # Adaptive Kappa based on the gap between f_best and the predicted minimum mu_min
    finite_mu = mu[np.isfinite(mu)]
    if finite_mu.size > 0:
        mu_min = np.min(finite_mu)
        gap = f_best - mu_min
        
        # If gap is large, the GP predicts a point much better than f_best.
        # We should trust this prediction (Exploit), so use lower kappa.
        # If gap is small, we are stuck or close. We might need to explore (Higher kappa).
        # Scale: Gap of 10 is considered "large" for this problem scale.
        t = np.clip(gap / 10.0, 0.0, 1.0)
        # kappa ranges from 0.5 (exploit) to 2.5 (explore)
        kappa = 0.5 + 2.0 * (1.0 - t)
    else:
        kappa = 2.0
        
    ucb_score = -mu + kappa * sigma_safe
    
    # 3. Hybrid Blending
    # Determine the blend weight based on progress towards the global minimum (3.0)
    # dist_from_min is large initially, small at the end.
    dist_from_min = max(f_best - 3.0, 1e-6)
    
    # w_ei: weight for EI. 
    # When dist is large (early), w_ei should be small (favor UCB).
    # When dist is small (late), w_ei should be large (favor EI).
    # Use exponential decay for smooth transition.
    # scale=5.0: At dist=5, w_ei ~ 0.63. At dist=0, w_ei ~ 1.0. At dist=10, w_ei ~ 0.13.
    w_ei = np.exp(-dist_from_min / 5.0)
    w_ucb = 1.0 - w_ei
    
    # Combine scores
    # Note: EI is always >= 0. UCB score can be negative.
    # To ensure they are on comparable scales for blending, we don't normalize them strictly,
    # but the dynamic weight helps. In early stages, UCB dominates. In late stages, EI dominates.
    scores = w_ei * ei + w_ucb * ucb_score
    
    # 4. Structural Change: Posterior Sampling / Expected Regret Approximation
    # The directive suggests using a posterior sampling-based acquisition that approximates
    # Expected Regret via Monte Carlo sampling, leveraging the known global minimum (3.0).
    #
    # Standard EI uses f_best. If f_best is stuck at a local minimum (e.g., 5.0) but the global
    # minimum is 3.0, EI will be small for points with mu > 5.0, even if they have high variance
    # and could potentially reach 3.0.
    #
    # "Expected Regret" relative to the true minimum (3.0) is E[max(3.0 - f(x), 0)]? No, regret is usually max(f(x) - f*, 0).
    # But for acquisition, we want to find points that are likely to be low.
    # Let's consider the probability that f(x) < f_best OR f(x) < 3.0 (if we think we can beat the current best).
    #
    # A more direct interpretation of "probability of improvement over the true optimum":
    # This is P(f(x) < 3.0).
    # P(f(x) < 3.0) = norm.cdf((3.0 - mu) / sigma)
    # This term is high when mu is low and sigma is high (uncertainty allows for low values).
    # This provides a strong exploration signal in regions with high variance that might contain the global minimum.
    
    # Calculate probability of beating the true minimum (3.0)
    z_true = (3.0 - mu) / sigma_safe
    prob_beat_true_min = norm.cdf(z_true)
    
    # This probability is in [0, 1].
    # To make it comparable to EI and UCB scores (which are in objective units, roughly 0-10+),
    # we can scale it. However, a probability is a soft score.
    # Let's blend this in.
    
    # We want to encourage exploration towards the global minimum.
    # If we are far from 3.0 (f_best > 3.0), this term adds value to high-variance regions.
    # If we are close to 3.0, this term becomes more focused.
    
    # Let's add a scaled version of this probability to the score.
    # Scale factor: How much do we value the chance of hitting the global min?
    # Let's use a scale that makes it comparable to EI.
    # Typical EI values are in the range of 0.1 to 2.0.
    # Prob is 0 to 1.
    # Let's multiply by a factor, say 1.0 or 2.0.
    # Also, we should probably only weight this heavily when we are not already close to the min.
    # But norm.cdf naturally handles this: if mu is close to 3 and sigma is small, prob is ~0.5.
    # If mu is 10 and sigma is 1, prob is ~0.
    # If mu is 10 and sigma is 5, prob is norm.cdf((3-10)/5) = norm.cdf(-1.4) ~ 0.08.
    # If mu is 5 and sigma is 2, prob is norm.cdf((3-5)/2) = norm.cdf(-1) ~ 0.16.
    
    # Let's blend this in with a weight that decreases as we get closer to the optimum?
    # Or just add it as a constant bonus?
    # The directive says: "providing a stronger exploration signal in regions with high variance that might contain the global mini[mum]"
    
    # Let's add: weight_true * prob_beat_true_min * scale
    # If we use a fixed weight, it might dominate early on?
    # Early on, mu is around 10-20, sigma is large. prob is small.
    # Late on, mu is around 3-5, sigma is small. prob is moderate if mu < 3.
    
    # Let's try adding it with a weight that is 1.0.
    # scores += 1.0 * prob_beat_true_min
    
    # Wait, if mu=3, sigma=0.1, prob = norm.cdf(0) = 0.5.
    # If mu=2.9, sigma=0.1, prob = norm.cdf(1) = 0.84.
    # This seems reasonable.
    
    # Let's refine the blend.
    # The current scores are a mix of EI and UCB.
    # We want to inject this "True Minimum Exploration" term.
    
    # Let's use a weight that is high when we are far from the minimum, to encourage searching for it.
    # w_true = 1.0 / (1.0 + dist_from_min) ?
    # Or just a constant?
    
    # Let's try a constant addition first, scaled to be comparable.
    # EI can be ~1.0. Prob is ~0.5.
    # Let's add 1.5 * prob_beat_true_min.
    
    scores += 1.5 * prob_beat_true_min
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
