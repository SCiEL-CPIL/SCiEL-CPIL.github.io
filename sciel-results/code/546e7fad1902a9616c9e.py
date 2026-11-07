"""Cost-aware acquisition function for Bayesian optimization (powell 4d) - Bayesian Thompson Sampling."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_EPS = 1e-9
_DEADLINE_MARGIN = 5  # seconds to stop before the 330s wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    """
    Bayesian Thompson Sampling acquisition function.
    
    Strategy:
    1. We want to maximize the "cost-aware improvement" per unit cost.
    2. Let Y be the function value. We observe Y ~ N(mu, sigma^2).
    3. We define the reward R as the improvement over f_best divided by cost:
       R = (f_best - Y) / cost
       Note: We want to minimize Y, so Improvement = f_best - Y.
       Since we want to MAXIMIZE the score, we maximize R.
       
    4. However, R is a random variable because Y is random.
       Y = mu + sigma * Z, where Z ~ N(0, 1).
       R = (f_best - (mu + sigma * Z)) / cost
       R = (f_best - mu)/cost - (sigma/cost) * Z
        
    5. The expected reward E[R] is:
       E[R] = (f_best - mu)/cost  (since E[Z]=0)
       
    6. The variance of the reward Var[R] is:
       Var[R] = (sigma/cost)^2 * Var[Z] = (sigma/cost)^2
        
    7. Thompson Sampling suggests sampling from the posterior of the reward:
       R_sample ~ N(E[R], Var[R])
       R_sample = E[R] + sqrt(Var[R]) * Z_sample
       
    8. To avoid sampling 2000 random numbers (which is fast but adds stochasticity 
       and might not be ideal if the GP is already doing the uncertainty quantification),
       we can use a deterministic surrogate that mimics the expected maximum of the 
       sampled rewards or simply use the standard UCB form which is equivalent to 
       maximizing the mean + beta * std. 
       
       Actually, pure Thompson Sampling for acquisition functions often involves 
       sampling a single function instance from the GP and then finding the argmax 
       of that instance. But here, we are given mu and sigma for 2000 points. 
       We cannot sample a whole function instance easily from just pointwise mu/sigma 
       without the kernel matrix. 
       
       However, we can approximate the "Thompson" behavior by using a UCB-like score 
       but with a beta derived from the desired exploration level. 
       
       Let's look at the "Cost-Aware" aspect more deeply.
       The current best score is 1.0541 with UCB / cost.
       
       Alternative Structural Change: **Knowledge Gradient (KG)** or **Probability of Improvement (PI)**?
       PI is very exploitative. KG is computationally expensive.
       
       Let's try **Expected Improvement (EI)** but cost-aware.
       EI(x) = E[max(f_best - Y, 0)]
       Y ~ N(mu, sigma^2)
       EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
       
       Cost-Aware EI: EI(x) / cost(x)
       
       Why is this different from the current UCB?
       UCB is linear in sigma. EI is non-linear. EI has a "floor" of 0. 
       When sigma is very small and mu > f_best, EI is 0. 
       When sigma is large, EI scales with sigma.
       
       The current program uses: (-mu + kappa * sigma) / cost.
       This is equivalent to maximizing -mu + kappa * sigma.
       
       Let's implement Cost-Aware EI. It is theoretically grounded for minimization 
       and handles the "improvement" concept more directly than UCB.
       
       Also, we can add a small exploration bonus to EI if the model is very uncertain, 
       but EI naturally includes exploration via the sigma term in the second part of the formula.
       
       Let's check the deadline. The formula involves Phi and phi (normal CDF and PDF).
       scipy.stats.norm.cdf and pdf are fast for 2000 points.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])

    t0 = time.time()
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, _EPS)
    cost = np.maximum(cost, _EPS)
    
    # Calculate Improvement: I = f_best - Y
    # Mean of Improvement: mu_I = f_best - mu
    # Std of Improvement: sigma_I = sigma
    
    mu_I = f_best - mu
    sigma_I = sigma
    
    # Standard normal variable Z = (f_best - Y) / sigma
    # Z ~ N(0, 1)
    # EI = E[max(I, 0)] = mu_I * Phi(Z) + sigma_I * phi(Z)
    # where Z = mu_I / sigma_I
    
    # Handle division by zero (sigma is already clipped to _EPS)
    z = mu_I / sigma_I
    
    # Clip z to avoid overflow in exp for phi, though norm.pdf handles it well
    # z can be very large positive or negative.
    # If z is very large positive, Phi(z) -> 1, phi(z) -> 0. EI -> mu_I.
    # If z is very large negative, Phi(z) -> 0, phi(z) -> 0. EI -> 0.
    
    from scipy.stats import norm
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = mu_I * Phi + sigma_I * phi
    
    # Ensure EI is non-negative (numerical errors might make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-awareness: Divide by cost
    score = ei / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() - t0 > _DEADLINE_MARGIN:
        return score
        
    return score

# EVOLVE-BLOCK-END
