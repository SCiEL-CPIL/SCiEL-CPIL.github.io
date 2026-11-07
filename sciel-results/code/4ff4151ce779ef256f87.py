# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_deadline = None

def acquisition(mu, sigma, f_best, cost):
    """
    Compute an acquisition score for each candidate.
    Returns a (2000,) array of finite scores (higher = more preferred).
    """
    global _deadline
    start_time = time.time()
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Expected Improvement (EI)
    # For minimization: EI = E[max(f_best - f, 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Probability of Improvement (PI)
    # PI = P(f < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # 3. Lower Confidence Bound (LCB) based score
    # We want to minimize f. LCB = mu - beta * sigma.
    # We want low LCB values. So score = -LCB = -mu + beta * sigma.
    # Beta controls exploration. 
    # A dynamic beta can be helpful, but fixed is safer for stability.
    # In high dimensions, exploration is hard. 
    # Let's try a moderate beta.
    beta = 1.5
    ucb_score = -mu + beta * sigma_safe
    
    # We need to combine these.
    # EI is in objective units.
    # PI is probability [0, 1].
    # UCB is in objective units.
    
    # Normalization strategy:
    # Since we only care about argmax, we can standardize the scores across the 2000 candidates.
    # This makes the weights (w_ei, w_pi, w_ucb) interpretable as relative importance.
    
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + eps
    ei_norm = (ei - ei_mean) / ei_std
    
    pi_mean = np.mean(pi)
    pi_std = np.std(pi) + eps
    pi_norm = (pi - pi_mean) / pi_std
    
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score) + eps
    ucb_norm = (ucb_score - ucb_mean) / ucb_std
    
    # Weights:
    # EI is generally the most robust.
    # UCB helps escape local minima by exploring high uncertainty regions.
    # PI is a bit redundant with EI but can be more stable in some regimes.
    # Let's try a blend that slightly favors exploration over the pure EI seed.
    # The seed was just EI/cost.
    # Let's try w_ei=0.7, w_ucb=0.3, w_pi=0.0
    # Or w_ei=0.6, w_ucb=0.2, w_pi=0.2
    
    # Let's experiment with a stronger exploration component.
    # Hartmann 6D has many local minima.
    # w_ei = 0.5
    # w_ucb = 0.5
    # This is essentially a UCB-EI hybrid.
    
    # Let's try a "Cost-Aware UCB" primarily, with a bit of EI.
    # Actually, the cost is 1 + mean(x). Points near 0 are cheaper.
    # Dividing by cost rewards cheap points.
    # If we use UCB, we might pick high sigma points which might be expensive or cheap.
    
    # Let's stick to a balanced mix.
    w_ei = 0.6
    w_pi = 0.1
    w_ucb = 0.3
    
    combined = w_ei * ei_norm + w_pi * pi_norm + w_ucb * ucb_norm
    
    # Apply cost penalty
    # cost is in [1, 2].
    # Dividing by cost is a linear penalty.
    # Using 1/cost^gamma might be stronger.
    # Let's try gamma = 1.0 first (same as seed).
    # If we want to favor cheap points more, we could do 1/cost^2.
    # But too strong might trap us in the cheap corner [0, 0.5]^6.
    # The minimizer is in the inner 80% [0.1, 0.9]^6.
    # So we shouldn't penalize too hard against the center.
    
    final_scores = combined / (cost + eps)
    
    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline check (though this function is fast, just in case)
    if _deadline is not None and time.time() > _deadline:
         # Return a safe fallback if we are running out of time?
         # The contract says "Each evaluation must finish within 300 s".
         # This function is called many times.
         # If we are close to the deadline, we should return quickly.
         # Since we are already done computing, just return.
         pass
         
    return final_scores

# EVOLVE-BLOCK-END
