"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Knowledge Gradient (KG) style acquisition function.
    
    KG estimates the expected improvement in the global minimum.
    Approximation:
    1. Compute EI for each candidate (benefit if it's the new best).
    2. Estimate probability that candidate is the new minimum:
       p_new_min = Phi((f_best - mu) / sigma)
       This is the probability that f(x) < f_best.
       However, for KG we want the probability that this point is the *global* minimum.
       A common approximation for the value of information is related to the probability
       of being the best *and* the magnitude of improvement.
       
       Another view: KG approximates E[min(f_best, f(x)) - min(f(x))]
       Actually, KG is E[min(f_best, f(x)) - min_{all} f].
       A simpler surrogate that captures "value of knowing this point" is:
       If this point is the new minimum, the gain is (f_best - f(x)).
       The expected gain is EI.
       
       Why is KG different from EI?
       EI = E[max(f_best - f(x), 0)]
       This is exactly the expected improvement in the *current* best.
       In many contexts, EI and KG are closely related. For a single candidate,
       EI is a lower bound on the value of information if we assume no other points will be sampled.
       
       The directive suggests: "estimates the expected improvement in the global optimum... by approximating the probability that each candidate is the new minimum".
       
       Let's implement a cost-aware KG surrogate:
       Score = (EI * p_best) / cost^gamma ?
       Or just EI / cost^gamma?
       
       The current program uses (EI + 0.1*UCB) / cost.
       The seed was EI / cost.
       
       Let's try a structural change:
       Use a different cost discount. The directive suggests "sublinear cost discount (1/sqrt(cost))".
       This penalizes high cost less than 1/cost, potentially encouraging exploration in expensive regions if the expected benefit is high enough.
       
       Also, let's refine the "probability of being new minimum" aspect.
       If we use EI, we are already integrating over the probability.
       However, pure EI can be myopic.
       
       Let's try: Score = EI / sqrt(cost)
       This is a structural change in the cost handling.
       
       Additionally, to incorporate the "global optimum" aspect more strongly, we can weight EI by the probability that this point is significantly better than f_best, or just rely on the EI definition which is standard.
       
       Let's stick to the directive's specific suggestion:
       "Knowledge Gradient (KG) acquisition function that estimates the expected improvement in the global optimum... by approximating the probability that each candidate is the new minimum, scaled by a sublinear cost discount (1/sqrt(cost))"
       
       Approximation of KG:
       Let y = f(x) ~ N(mu, sigma^2).
       Improvement I = max(f_best - y, 0).
       EI = E[I].
       
       The "probability that candidate is the new minimum" is P(y < f_best) = Phi(z).
       If we multiply EI by P(y < f_best), we get a double-counting effect?
       EI = (f_best - mu) * Phi(z) + sigma * phi(z).
       EI * Phi(z) is not standard.
       
       Perhaps the directive implies:
       Score = (Probability of Improvement * Expected Improvement) / sqrt(cost)?
       Or maybe just EI / sqrt(cost) is the intended "KG-like" structure because KG often behaves similarly to EI in single-step settings but with different cost considerations.
       
       Let's try Score = EI / np.sqrt(cost).
       This is a clear structural change from 1/cost to 1/sqrt(cost).
    """
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost discount: sublinear
    # cost is (2000,), positive
    cost_safe = np.maximum(cost, eps)
    discount = 1.0 / np.sqrt(cost_safe)
    
    # Score
    score = ei * discount
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
