"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Uses a hybrid acquisition that balances exploration and exploitation.
    We use a modified Expected Improvement with an exploration bonus based
    on uncertainty, which can be more effective in high-dimensional or 
    multimodal landscapes, but here for 1D sphere function, we want to 
    converge quickly to the minimum.
    
    Actually, for a simple 1D sphere function (x^2), the optimum is at x=0.
    The GP should be able to model this well. EI is a strong baseline.
    
    Let's try Upper Confidence Bound (UCB) which explicitly balances 
    exploitation and exploration:
    UCB = -mu + kappa * sigma
    
    Since we're minimizing, we want low mu and high sigma (uncertainty) 
    to encourage exploration in unexplored regions.
    
    However, the standard UCB for minimization is:
    acquisition = -mu + kappa * sigma
    
    We need to choose kappa. A common heuristic is to use a value that 
    decreases over time, but since we don't have access to the number 
    of iterations, we'll use a fixed value.
    
    Actually, let's think about this more carefully. The sphere function 
    in 1D is f(x) = x^2, with minimum at x=0. The domain is [-5, 5].
    
    With 5 initial points and 25 more evaluations, we have a total of 30 
    evaluations. The GP with Matern-5/2 kernel should be able to model 
    this quadratic function reasonably well.
    
    Let's try a different approach: use the Probability of Improvement 
    (PI) with a small improvement threshold, or use a combination of EI 
    and UCB.
    
    Actually, let's try a simple approach: use the negative mean plus a 
    weighted standard deviation. The weight kappa controls the trade-off.
    
    For minimization:
    score = -mu + kappa * sigma
    
    A typical value for kappa is around 1-2. Let's try kappa = 1.5.
    
    But wait, we need to be careful. If sigma is very small (we're certain), 
    we should exploit (use -mu). If sigma is large (we're uncertain), we 
    should explore (use +sigma).
    
    Let's also consider the scale. The function values range from 0 to 25.
    The GP is standardized, so mu and sigma are in objective units.
    
    Let's try:
    score = -mu + kappa * sigma
    
    with kappa = 1.0.
    
    Actually, I recall that for simple 1D functions, EI often works very well.
    The current program uses EI and gets a score of 1.0000.
    
    Let's try something different: use the Lower Confidence Bound (LCB) 
    for minimization, which is the same as UCB for maximization.
    
    LCB = mu - kappa * sigma (for minimization, we want to minimize this)
    So the acquisition (higher is better) would be: -LCB = -mu + kappa * sigma
    
    This is the same as UCB.
    
    Let's try kappa = 1.0 first.
    
    Actually, let me think about this differently. The key insight is that 
    for the sphere function, the optimal point is at x=0. The GP should 
    predict low mu near x=0 and high sigma far from the observed points.
    
    With 5 initial uniform points, the GP has some information about the 
    function. As we evaluate more points, the GP becomes more certain.
    
    EI is a good choice because it naturally balances exploration and 
    exploitation. However, it can be too conservative, leading to slow 
    convergence.
    
    UCB with a larger kappa might explore more aggressively, which could 
    help find the optimum faster.
    
    Let's try a hybrid: use EI but with a modified z-score that accounts 
    for the global scale of the problem.
    
    Actually, let's just try UCB with kappa = 2.0 and see if it performs 
    better.
    """
    # Upper Confidence Bound for minimization
    # score = -mu + kappa * sigma
    # Higher score means more promising
    
    kappa = 1.5
    score = -mu + kappa * sigma
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
