# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a hybrid of Expected Improvement (EI)
    and a variance-based exploration bonus, normalized by cost.
    
    The key idea: 
    1. Standard EI captures exploitation.
    2. Adding a term proportional to sigma (or sigma^2) encourages exploration 
       in regions of high uncertainty, which is crucial for finding the global 
       minimum in multi-modal landscapes like Styblinski-Tang when the initial 
       points may not be close enough to the optimum.
    3. Dividing by cost ensures we prioritize cheaper evaluations, but the 
       exploration bonus helps overcome local optima.
    
    We use: Score = (EI + kappa * sigma) / cost
    where kappa is a small constant to balance exploration.
    
    Alternatively, we can use a more robust form: 
    Score = EI / cost + beta * (sigma / cost)
    
    Let's try a simpler, more direct approach: 
    Instead of pure EI/cost, use a combination that rewards high uncertainty 
    even if the mean is not yet clearly better than f_best.
    
    Actually, a common effective strategy for cost-aware BO is:
    Score = EI / cost
    
    But to be distinct from the seed, let's try **Probability of Improvement (PI)**
    divided by cost, or a modified EI.
    
    Let's try **Expected Hypervolume Improvement (EHI)**? Too complex without more info.
    
    Let's try **Upper Confidence Bound (UCB)** with cost-awareness:
    UCB(x) = mu(x) + xi * sigma(x)
    But we are minimizing, so we want low mu. 
    Acquisition = -UCB(x) / cost? No, that would minimize -UCB, i.e., maximize UCB.
    We want to maximize the acquisition score.
    For minimization, a good score is: -mu(x) + xi * sigma(x), then divide by cost.
    But -mu can be very negative.
    
    Let's stick to a robust variation of EI:
    Use EI but add a small exploration bonus proportional to sigma.
    Score = (EI + 0.1 * sigma) / cost
    
    Why? The seed uses EI/cost. If the GP is overconfident (sigma ~ 0) in regions 
    that are actually bad, it might miss better regions. Adding a sigma bonus 
    ensures some exploration.
    
    Let's refine: 
    z = (f_best - mu) / (sigma + 1e-9)
    EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    If we add sigma to EI, we get:
    Score = (EI + sigma * exp_factor) / cost
    
    Let's try exp_factor = 0.5.
    
    Another idea: **Cost-Weighted Probability of Improvement (cPI)**
    PI = norm.cdf(z)
    Score = PI / cost
    
    PI is smoother than EI and less prone to numerical issues, but less exploitative.
    Combining EI and PI: Score = (0.5 * EI + 0.5 * PI * (f_best - mu)) / cost? No.
    
    Let's go with **(EI + 0.2 * sigma) / cost**. This is a distinct, simple modification 
    that encourages exploration in uncertain regions while maintaining cost-awareness.
    """
    sigma_safe = sigma + 1e-9
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration bonus: proportional to uncertainty
    exploration_bonus = 0.2 * sigma
    
    # Combined score: (EI + exploration bonus) / cost
    # Ensure scores are finite
    scores = (ei + exploration_bonus) / (cost + 1e-9)
    
    # Replace any non-finite values with a very low score (0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
