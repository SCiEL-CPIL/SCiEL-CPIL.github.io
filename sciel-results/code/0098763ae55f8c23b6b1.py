import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using GP-UCB with time-varying beta combined with Expected Improvement.
    
    The exploration parameter beta_t is estimated from the current best value f_best,
    mapping it to an effective step count t. This allows the algorithm to start with
    high exploration (large beta) when f_best is poor, and transition to exploitation
    (small beta) as f_best improves.
    
    The final score is a weighted sum of:
    1. GP-UCB score: -mu + beta_t * sigma (encourages exploration of high uncertainty)
    2. Normalized Expected Improvement: encourages exploitation of regions likely to improve f_best
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Handle non-finite f_best
    if not np.isfinite(f_best):
        f_best = 1e6
    
    # Estimate effective time step t from f_best
    # As f_best decreases, t increases, leading to lower beta (more exploitation)
    # Mapping: f_best ~ 100 -> t ~ 1, f_best ~ 10 -> t ~ 4, f_best ~ 1 -> t ~ 17, f_best ~ 0 -> t ~ 21
    f_best_clipped = max(f_best, 0.0)
    t = 1.0 + 20.0 * np.exp(-f_best_clipped / 5.0)
    t = max(t, 1.0)
    
    # Theoretical beta_t = sqrt(2 * t * log(1 + t))
    # This provides a principled schedule for exploration
    beta_t = np.sqrt(2.0 * t * np.log(1.0 + t))
    
    # Clamp beta to a reasonable range to prevent extreme behavior
    # Early on (small t), beta is small. Later (large t), beta grows.
    # However, for minimization, we want to reduce exploration over time.
    # The formula above increases beta with t.
    # Let's re-evaluate: 
    # If t is small (early, bad f_best), we want HIGH exploration.
    # If t is large (late, good f_best), we want LOW exploration.
    # The standard GP-UCB beta_t increases with t to guarantee regret bounds, 
    # but in practice for finite horizon BO, we often want to decay exploration.
    # 
    # Let's invert the logic or use a different mapping.
    # If f_best is large (bad), we are early. We want high beta.
    # If f_best is small (good), we are late. We want low beta.
    #
    # Let's define t_effective such that it is LARGE when f_best is LARGE.
    # t_eff = 1 + 20 * exp(f_best / 5)
    # f_best=100 -> t_eff ~ inf (cap it)
    # f_best=10 -> t_eff ~ 1 + 20*exp(2) ~ 150
    # f_best=1 -> t_eff ~ 1 + 20*exp(0.2) ~ 24
    # f_best=0 -> t_eff ~ 1 + 20 = 21
    #
    # Then beta_t = sqrt(2 * t_eff * log(1 + t_eff))
    # This will be large when f_best is large.
    # Let's try this mapping.
    
    t_eff = 1.0 + 20.0 * np.exp(f_best_clipped / 5.0)
    # Cap t_eff to avoid numerical issues and excessive beta
    t_eff = np.clip(t_eff, 1.0, 1000.0)
    
    beta_t = np.sqrt(2.0 * t_eff * np.log(1.0 + t_eff))
    
    # Clamp beta to a reasonable range [1, 15]
    # Too high beta leads to random search, too low leads to local minima
    beta_t = np.clip(beta_t, 1.0, 15.0)
    
    # Calculate GP-UCB score for minimization
    # We want to maximize the score.
    # UCB for minimization: minimize mu - beta*sigma? No, UCB is an upper bound on the true value.
    # We want to find points where the true value is likely low.
    # Lower Confidence Bound (LCB) is mu - beta*sigma. We want to minimize LCB.
    # So we maximize -LCB = -mu + beta*sigma.
    ucb_score = -mu + beta_t * sigma
    
    # Calculate Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    z = (f_best - mu) / sigma
    # Clip z to prevent overflow in exp
    z = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Normalize EI to have similar scale as UCB score
    # The scale of EI depends on f_best and the spread of mu.
    # A simple normalization is to divide by the standard deviation of mu or a characteristic scale.
    # Let's use the standard deviation of the UCB score to normalize EI, or just scale EI by a factor.
    # Alternatively, normalize EI to [0, 1] and UCB to [0, 1] and combine.
    
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    if ucb_max - ucb_min > 1e-9:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max - ei_min > 1e-9:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    # Combine normalized scores
    # Weight: More weight on UCB when beta is high (early), more on EI when beta is low (late)
    # beta_t ranges from 1 to 15.
    # weight_ucb = beta_t / (beta_t + 1)  # High beta -> high weight on UCB
    # weight_ei = 1 / (beta_t + 1)       # High beta -> low weight on EI
    
    # Let's use a smoother transition
    # When beta_t is large (exploration phase), we trust UCB more.
    # When beta_t is small (exploitation phase), we trust EI more.
    
    # Normalize beta_t to [0, 1] for weighting
    beta_norm = (beta_t - 1.0) / (15.0 - 1.0)
    beta_norm = np.clip(beta_norm, 0.0, 1.0)
    
    # Weight for UCB increases with beta_norm
    # Weight for EI decreases with beta_norm
    # Let's say weight_ucb = 0.5 + 0.5 * beta_norm
    # weight_ei = 0.5 - 0.5 * beta_norm
    # This ensures weights sum to 1 and transition from 0.5/0.5 to 1.0/0.0
    
    weight_ucb = 0.5 + 0.5 * beta_norm
    weight_ei = 0.5 - 0.5 * beta_norm
    
    score = weight_ucb * ucb_norm + weight_ei * ei_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
