"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling logic.
    
    Instead of a fixed UCB beta, we simulate the posterior distribution
    at the candidate points. We draw a sample from N(mu, sigma) and use
    that sample as the acquisition score. This naturally balances
    exploration (high sigma leads to larger variance in samples) and
    exploitation (low sigma leads to samples close to mu).
    
    To make it more robust and deterministic for the evaluator (which 
    expects a single vector of scores), we can use the expected value 
    of the maximum of K samples, or simply a quantile. 
    However, standard Thompson Sampling usually picks the argmax of a 
    single sample. Since we must return scores for all points to let 
    the evaluator pick the argmax, we can return a score that is 
    equivalent to the expected improvement of the sample, or just 
    the sample itself if we were to do a single draw.
    
    But since the evaluator calls acquisition and picks the argmax, 
    returning a random sample would make the result stochastic. 
    The task implies a deterministic function signature, but 
    Thompson Sampling is inherently stochastic. 
    
    A common deterministic proxy for Thompson Sampling in this 
    context is to use the mean of the posterior, adjusted by a 
    term that reflects the probability of being the best. 
    Alternatively, we can use the "Expected Improvement" of the 
    posterior mean, which is standard EI. 
    However, the previous program used UCB. 
    Let's try a different angle: The "Knowledge Gradient" or 
    simply a more sophisticated UCB where beta is adaptive to 
    the spread of f_best relative to the current sigma.
    
    Actually, a very effective strategy for low-budget BO is 
    "Probability of Improvement" (PI) or "Expected Improvement" 
    (EI) with a small exploration bonus. 
    The previous success was UCB with beta=1.5. 
    Let's try a strategy that is fundamentally different: 
    Using the posterior predictive distribution to estimate the 
    "Expected Improvement" but with a twist: we maximize the 
    probability that the next point is significantly better than 
    f_best. This is essentially PI with a threshold.
    
    Let's try: Score = -mu + beta * sigma, but beta is calculated 
    to balance the number of points that "look promising".
    Or, let's use a simpler, robust method: 
    Score = -mu + 2 * sigma (Higher exploration than 1.5)
    Or, let's use the "Knowledge Gradient" approximation.
    
    Actually, the most distinct strategy from "Static UCB" is 
    "Adaptive UCB" where beta depends on the current state of 
    the search (e.g., how many points have been evaluated, or 
    the current uncertainty of the best point). 
    But we don't have the count. We have f_best.
    
    Let's try a "Greedy with Exploration" approach using the 
    "Confidence Bound" but with a different formulation: 
    Score = -mu + beta * sigma / (1 + sigma)
    This normalizes the exploration term so that very high 
    uncertainty doesn't dominate indefinitely.
    
    Another distinct strategy: "Bayesian Optimization with 
    Random Forests" is not available. 
    Let's try "Expected Improvement" (EI) properly. 
    EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    This is the standard, theoretically grounded approach. 
    The previous program moved AWAY from EI to UCB and improved. 
    This suggests EI was too greedy. 
    However, maybe the EI implementation was flawed or the 
    standard EI is just not good for this specific problem/budget.
    
    Let's try a "Hybrid" that is distinct from pure UCB: 
    A "Two-Phase" strategy. 
    Since we don't know the phase, we can use a heuristic: 
    If sigma is very high, prioritize exploration. 
    If sigma is low, prioritize exploitation.
    
    Let's try "Probability of Improvement" (PI):
    PI = Phi((f_best - mu) / sigma)
    This is very exploratory in the early stages because 
    any point with mu < f_best has a high PI. 
    It tends to be less greedy than EI but more exploitative 
    than UCB in the later stages.
    
    Let's implement PI. It is a different design family (Probability-based vs Bound-based).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # PI = Phi(z)
    # Using the error function for stability
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    pi_scores = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
    
    # We want higher scores to be better.
    # PI is already between 0 and 1, higher is better.
    # However, PI can be very flat. 
    # To make it more sensitive, we can raise it to a power or use EI.
    # Let's stick to PI as it is a distinct family.
    
    # Ensure finite
    scores = np.where(np.isfinite(pi_scores), pi_scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
