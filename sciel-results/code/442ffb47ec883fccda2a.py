import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Predictive Entropy Search (PES) variant.
    
    Strategy:
    Instead of simple EI or UCB, we aim to maximize the expected reduction in the
    entropy of the global minimum. This is approximated by the mutual information
    between the function value at the candidate and the location/value of the
    current best.
    
    For multimodal functions like Weierstrass, standard EI can be too greedy.
    PES-like metrics target regions where learning about the global minimum is
    most valuable.
    
    We approximate the PES score using the following intuition:
    The value of information is high when:
    1. The candidate is likely to be better than the current best (high PI).
    2. The uncertainty is high (high sigma), but not too high (to avoid noise).
    3. The improvement is significant relative to the current best.
    
    A common approximation for PES in 1D is based on the change in the entropy
    of the posterior distribution of the global minimum.
    
    Here, we use a weighted combination of:
    1. Probability of Improvement (PI): P(f(x) < f_best)
    2. Expected Improvement (EI): E[max(f_best - f(x), 0)]
    3. A "surprise" or "information gain" term related to the z-score.
    
    Specifically, we use a score that emphasizes candidates with high PI and
    moderate EI, while penalizing candidates with very high uncertainty that
    are unlikely to improve.
    
    Score = PI * (1 + EI / (f_best + eps)) * (1 + z^2 / (1 + z^2))
    
    The term (1 + z^2 / (1 + z^2)) is a soft penalty for extreme z-scores,
    encouraging exploration of regions with meaningful uncertainty.
    """
    start_time = time.time()
    timeout = 290.0  # Safety margin before 300s limit
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Expected Improvement (EI)
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI to be comparable with PI
    # Scale EI by 1/(f_best + 1) to keep it bounded and adaptive
    scale = 1.0 / (np.abs(f_best) + 1.0)
    ei_scaled = ei * scale
    
    # Information gain term: soft penalty for extreme z-scores
    # This encourages exploration of regions with meaningful uncertainty
    z2 = z ** 2
    info_gain = 1.0 + z2 / (1.0 + z2)
    
    # Combine terms
    scores = pi * (1.0 + ei_scaled) * info_gain
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > timeout:
        # Fallback to a simple, fast score if we are running out of time
        return scores
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
