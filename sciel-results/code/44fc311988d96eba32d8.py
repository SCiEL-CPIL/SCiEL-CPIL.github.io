import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function for minimization.
    
    KG(x) = E[max(f_best - min(f(x), f_best), 0)]
         = E[max(f_best - f(x), 0)] when we consider the expected improvement
         of the best value after evaluating x.
    
    For a single-point evaluation with f(x) ~ N(mu, sigma^2):
    KG(x) = E[max(f_best - f(x), 0)]
         = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    This is actually the same as EI for a single evaluation. However, the true
    KG for minimizing the best over N evaluations would be different. Since we
    only have 30 total evaluations and are choosing the next one, the single-step
    KG equals EI.
    
    But wait - the inspiration says KG improved from 0.9682 to 1.0. Let me think
    about what makes KG different from EI in practice.
    
    Actually, for a single next evaluation, EI and KG are identical. The difference
    might come from how we handle the computation or add a small exploration bonus.
    
    Let me implement a hybrid: primarily EI but with a slight UCB-like exploration
    term to encourage more exploration early on, or use a modified version.
    
    Actually, let me reconsider. The key insight from the knowledge is that KG
    "adaptively balances exploration and exploitation based on the posterior 
    uncertainty and current best value." 
    
    One structural change: use a two-stage approach where we compute EI but also
    weight candidates by their potential to reduce uncertainty in the region.
    
    Let me try: EI + small * sigma (to encourage exploration of uncertain regions)
    This is essentially a UCB with a very small beta, but framed differently.
    
    Actually, a cleaner structural change: use the Probability of Improvement (PI) 
    combined with EI, or use a quantile-based approach.
    
    Let me try a different structural approach: instead of pure EI, use a 
    "expected maximum improvement" that accounts for the fact that we're doing
    multiple evaluations. A simple approximation is to use EI but with a slightly
    inflated sigma to account for the fact that future evaluations might find
    better points, so we should be more aggressive in exploring now.
    
    Or: use a mixture of EI and PI.
    
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu)/sigma)
    
    A common hybrid is: score = EI + alpha * PI, or score = EI * (1 + beta * sigma)
    
    Let me try: score = EI + 0.1 * sigma * PI, which adds a small bonus for
    uncertain regions that are also likely to be good.
    
    Actually, the simplest structural change that's known to work well: use
    a time-adaptive UCB where beta decreases over time. But we don't have
    step count info.
    
    Let me just implement a clean KG/EI with a small exploration bonus proportional
    to sigma, which effectively makes it a UCB with beta=0.1. This is a structural
    change from pure EI.
    """
    # Ensure sigma is non-negative and handle numerical stability
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute the standardized improvement
    z = (f_best - mu) / sigma
    
    # Compute Phi(z) and phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Add a small exploration bonus proportional to uncertainty
    # This encourages exploring regions with high sigma
    # The coefficient 0.1 is a compromise between pure exploitation (EI) and exploration
    exploration_bonus = 0.1 * sigma
    
    # Final score: EI + exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
