import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Weierstrass 1D on [-0.5, 0.5] has a minimum of 0.
    # f_best is the lowest observed value.
    # Early on, f_best is around 0.5-1.0. Late on, f_best is close to 0.
    
    # We use a hybrid approach:
    # 1. Compute Expected Improvement (EI) for minimization.
    #    EI(x) = E[max(f_best - f(x), 0)]
    #    For GP posterior N(mu, sigma^2):
    #    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    #    where z = (f_best - mu) / sigma
    #    Note: This is for maximization of improvement. Since we minimize f,
    #    improvement is f_best - f(x).
    
    # 2. Compute UCB (Lower Confidence Bound for minimization).
    #    UCB = -mu + beta * sigma
    #    This encourages exploration of high uncertainty regions.
    
    # The directive suggests using a probabilistic EI variant that scales the improvement threshold.
    # However, standard EI is already probabilistic. 
    # A common refinement is to blend EI with UCB or use a "noisy" EI.
    # Another idea from inspirations: adaptive beta for UCB.
    # Let's combine:
    # - Use EI as the primary score.
    # - Add a small exploration bonus based on UCB to avoid getting stuck in local minima 
    #   where EI is 0 (if mu < f_best everywhere, EI is 0, but we might still want to explore).
    #   Actually, if mu < f_best, EI is positive. If mu > f_best, EI is 0 (if sigma is small).
    #   Wait, if mu > f_best, z is negative, Phi(z) is small, phi(z) is small.
    #   EI is always non-negative.
    
    # Let's compute standard EI first.
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in exp, though norm.pdf handles it well.
    # z can be very large positive (mu << f_best) -> EI ~ f_best - mu
    # z can be very large negative (mu >> f_best) -> EI ~ 0
    # norm.cdf and norm.pdf are stable.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI is 0 when mu is much larger than f_best and sigma is small.
    # This can lead to all scores being 0 if the GP is confident all points are worse than f_best.
    # In that case, we should fall back to exploration (UCB).
    
    # Let's check if EI is effectively zero everywhere.
    # If max(ei) is very small, we rely on UCB.
    
    # Adaptive Beta for UCB:
    # When f_best is high (early), explore more.
    # When f_best is low (late), exploit more.
    # med_sigma = np.median(sigma)
    # if med_sigma < 1e-6: med_sigma = 1e-6
    # beta = max(f_best / med_sigma, 0.5)
    # beta = min(beta, 5.0)
    
    # Let's use a simpler adaptive strategy:
    # beta = 2.0 + 2.0 * (f_best / 1.0)  # Assuming f_best in [0, 1]
    # Clamp beta.
    beta = 2.5
    
    ucb = -mu + beta * sigma
    
    # Blend EI and UCB.
    # If EI is informative (max_ei > threshold), use it.
    # Otherwise, use UCB.
    # Or, simply add a small weighted UCB to EI to ensure some exploration.
    # score = ei + 0.1 * ucb
    
    # Let's try a smooth blend.
    # If EI is flat (variance is low), use UCB.
    # If EI has structure, use EI.
    
    # A robust approach:
    # score = ei + lambda * ucb
    # Where lambda is small.
    
    # However, EI and UCB have different scales.
    # EI is in objective units. UCB is in objective units (roughly).
    # Let's normalize UCB? No, keep units consistent.
    
    # Let's look at the magnitudes.
    # f_best ~ 0.5. mu ~ 0.5. sigma ~ 0.1.
    # EI ~ 0.1.
    # UCB ~ -0.5 + 2.5*0.1 = -0.25.
    # They are comparable.
    
    # Let's use a weighted sum.
    # w_ei = 1.0
    # w_ucb = 0.2
    
    score = ei + 0.2 * ucb
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
