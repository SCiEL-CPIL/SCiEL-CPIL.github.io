import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Thompson Sampling-inspired heuristic.
    
    Strategy:
    - Base score is negative mean (exploitation).
    - Exploration term is scaled by a dynamic weight that depends on f_best.
    - Early on (high f_best), we explore more aggressively.
    - As f_best approaches 0 (the minimum of Ackley), we shift to exploitation.
    - We use a quantile-based normalization of sigma to make the exploration term robust.
    """
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return -mu
    
    # Ackley 2D minimum is 0.
    # f_best is the lowest observed value.
    # If f_best is close to 0, we should exploit.
    # If f_best is far from 0, we should explore.
    
    # Scale factor: 
    # When f_best is large (e.g., 10), weight is high.
    # When f_best is small (e.g., 0.01), weight is low.
    # Let's use a simple decay: weight = 1 / (1 + f_best)
    # Or perhaps a more aggressive decay: weight = 1 / (1 + f_best**2)
    # Let's try weight = 1.0 / (1.0 + f_best)
    
    # Also, we want to normalize sigma to avoid scale issues.
    # We can use the median or a quantile as a scale reference.
    # Let's use the 95th percentile of sigma as the scale.
    sigma_scale = np.percentile(sigma, 95)
    if sigma_scale < 1e-9:
        sigma_scale = 1.0
        
    normalized_sigma = sigma / sigma_scale
    
    # Dynamic beta
    # Base beta around 1.0, modulated by f_best
    # If f_best is 10, beta ~ 1/11 ~ 0.09? No, that's too low.
    # Let's try: beta = 2.0 / (1.0 + f_best)
    # If f_best = 10, beta = 0.18
    # If f_best = 1, beta = 1.0
    # If f_best = 0.1, beta = 1.8
    # If f_best = 0, beta = 2.0
    
    # Wait, the previous best (1.5379) used beta=1.0 constant.
    # The inspiration 3 (1.0933) used beta=1.5 constant.
    # So constant beta=1.0 was better than 1.5.
    # But the directive says to try Thompson Sampling heuristic with dynamic weight.
    
    # Let's refine the dynamic weight.
    # We want to encourage exploration early.
    # f_best starts around ~10 for Ackley in [-5,5]^2?
    # Actually, Ackley(0,0)=0. Random points in [-5,5]^2 can have values up to ~20-30.
    # So f_best starts high.
    
    # Let's use: beta = 1.0 + 2.0 * np.exp(-f_best)
    # If f_best = 10, beta ~ 1.0
    # If f_best = 1, beta ~ 1.7
    # If f_best = 0, beta ~ 3.0
    
    # This seems counter-intuitive: we explore MORE as we get closer to the minimum?
    # Usually, we explore less as we get closer.
    # But maybe the uncertainty structure changes?
    
    # Let's stick to the directive: "aggressively target high-variance regions early and sharply focus on the global minimum as f_best approaches zero."
    # This implies:
    # Early (f_best high): High exploration weight.
    # Late (f_best low): Low exploration weight.
    
    # So beta should decrease as f_best decreases.
    # beta = beta_max * exp(-f_best / scale)
    # If f_best = 10, beta ~ beta_max
    # If f_best = 0, beta ~ 0
    
    # Let's try: beta = 2.0 * np.exp(-f_best / 5.0)
    # f_best=10 -> beta ~ 0.27
    # f_best=5 -> beta ~ 0.74
    # f_best=1 -> beta ~ 1.7
    # f_best=0 -> beta ~ 2.0
    
    # This still increases beta as f_best decreases? No, exp(-x) decreases as x increases.
    # So as f_best decreases (approaches 0), exp(-f_best/5) increases.
    # So beta increases as we get closer to the minimum.
    # This contradicts "sharply focus on the global minimum".
    
    # We want beta to be HIGH when f_best is HIGH (early).
    # And beta to be LOW when f_best is LOW (late).
    
    # So beta should be proportional to f_best?
    # beta = 0.5 * f_best?
    # f_best=10 -> beta=5
    # f_best=1 -> beta=0.5
    # f_best=0 -> beta=0
    
    # This seems plausible. But we need to handle f_best=0 carefully.
    # Also, f_best can be negative? No, Ackley is non-negative.
    
    # Let's try: beta = 1.0 + 0.5 * f_best
    # f_best=10 -> beta=6
    # f_best=1 -> beta=1.5
    # f_best=0 -> beta=1.0
    
    # This maintains a base exploration of 1.0 (which was good) and adds more early on.
    
    # Let's combine with normalized sigma.
    beta = 1.0 + 0.5 * f_best
    
    # But wait, if f_best is very large, beta becomes very large.
    # This might cause us to pick very high uncertainty points, which might be far away.
    # That's good for global exploration.
    
    # Let's use the normalized sigma to keep the scale reasonable.
    scores = -mu + beta * normalized_sigma
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
