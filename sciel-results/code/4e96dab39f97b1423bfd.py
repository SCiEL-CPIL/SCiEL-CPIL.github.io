"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement (EI)
    and Probability of Improvement (PI), weighted by inverse cost.
    
    The idea: EI is generally better for optimization, but PI can be more robust
    when the GP is uncertain. We use a mixture that favors EI when the expected
    improvement is large relative to the uncertainty, and PI otherwise.
    
    Additionally, we apply a small bonus for points that are far from the
    origin (in the scaled space) to encourage exploration of the domain,
    since the rastrigin function has multiple local minima and the global
    minimum is at (0,0) in the original space, which maps to (0.5, 0.5) in
    the scaled [0,1] space. Wait, the minimizer is moved to a uniform point in
    the inner 80% of each range. So the global minimum is not necessarily at
    the center. Therefore, we should not bias towards the center.
    
    Instead, let's stick to a more robust cost-aware EI. The current EI/cost
    is good, but we can improve it by:
    1. Using a small epsilon to avoid division by zero.
    2. Adding a small exploration bonus based on the standard deviation,
       scaled by the inverse cost, to encourage exploration in uncertain regions.
    3. Normalizing the EI by the cost in a way that accounts for the fact that
       cost is between 1 and 2 (since x is scaled to [0,1], mean(x) is in [0,1],
       so cost is in [1, 2]).
    
    Let's try: EI / cost + beta * sigma / cost
    where beta is a small constant, e.g., 0.1.
    
    Actually, a better approach is to use the "Expected Improvement per unit cost"
    but with a floor on the denominator to avoid extreme values.
    
    Another idea: use the "Probability of Improvement" (PI) which is more robust
    to outliers in the GP predictions. PI = Phi((f_best - mu) / sigma).
    
    Let's combine EI and PI:
    score = (alpha * EI + (1 - alpha) * PI * f_best) / cost
    
    But f_best is in objective units, and PI is a probability in [0,1].
    This might not be well-scaled.
    
    Let's try a different approach: use the "Upper Confidence Bound" (UCB)
    acquisition function, which is known to be effective for exploration.
    UCB = mu + kappa * sigma
    But we want to minimize, so we use:
    UCB_min = mu - kappa * sigma
    
    Then, we divide by cost to get a cost-aware score:
    score = (f_best - UCB_min) / cost = (f_best - mu + kappa * sigma) / cost
    
    This is similar to EI but with a linear combination instead of the
    Gaussian CDF/PDF. The choice of kappa is important. A common choice is
    kappa = 2 or 3.
    
    Let's try kappa = 2.5 and see if it performs better than EI.
    
    To make it even more robust, we can use a mixture of EI and UCB:
    score = (alpha * EI + (1 - alpha) * (f_best - mu + kappa * sigma)) / cost
    
    Let's set alpha = 0.5 and kappa = 2.5.
    """
    # Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound for minimization
    kappa = 2.5
    ucb = f_best - mu + kappa * sigma
    
    # Mixture of EI and UCB
    alpha = 0.5
    score = alpha * ei + (1 - alpha) * ucb
    
    # Cost-aware: divide by cost
    score = score / (cost + 1e-9)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
