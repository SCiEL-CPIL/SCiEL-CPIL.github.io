# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) approximation acquisition function.
    
    Strategy: 
    Instead of optimizing for immediate improvement (EI) or hypervolume expansion,
    this function approximates the "Knowledge Gradient" by estimating the expected
    reduction in the global maximum posterior standard deviation.
    
    Rationale:
    In multimodal landscapes like Branin, standard EI can get trapped in local minima
    by focusing too heavily on exploitation near the current best. Hypervolume proxies
    can be too exploratory. The Knowledge Gradient balances this by asking: "Which
    candidate, if evaluated, would most reduce our overall uncertainty about the
    global optimum?"
    
    Approximation:
    We approximate the KG by assuming that evaluating a point with high uncertainty
    (sigma) in a region where the mean is competitive (close to f_best) will most
    significantly reduce the "effective" uncertainty of the landscape.
    
    We use a weighted combination:
    1. Exploration term: sigma (reduces local uncertainty)
    2. Relevance term: exp(-((mu - f_best) / scale)^2) (ensures we don't explore
       regions that are clearly worse than f_best)
    3. Global impact proxy: We normalize sigma by the global max sigma to ensure
       the score is relative to the current model's uncertainty landscape.
    
    This is structurally different from EI (integral) and HV (product of width/height)
    as it uses an exponential decay kernel for relevance and a normalized uncertainty
    term, reflecting the probabilistic value of information.
    """
    # Deadline guard: ensure we return quickly
    t_start = time.time()
    deadline = 0.1 # 100ms safety margin per call, though this is O(N) and fast
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # 1. Relevance Score: How promising is the mean?
    # We want points where mu is close to or better than f_best.
    # Use a Gaussian kernel centered at f_best.
    # Scale is set to a fraction of the problem range or f_best magnitude.
    # Branin range is roughly [0, 300]. f_best starts high, decreases.
    # Using f_best as the scale adapts the exploration radius as we converge.
    # If f_best is large, we explore broadly. If f_best is small, we focus.
    # However, f_best can be 0.39. A scale of max(f_best, 1.0) is safe.
    scale = max(f_best, 1.0)
    
    # Distance from best
    diff = mu - f_best
    # Relevance is high when mu <= f_best, decays as mu >> f_best
    relevance = np.exp(-(diff / scale) ** 2)
    
    # 2. Uncertainty Score: How much information would we gain?
    # Normalize sigma to [0, 1] relative to the current candidate set.
    max_sigma = np.max(sigma)
    if max_sigma < 1e-12:
        max_sigma = 1.0
    norm_sigma = sigma / max_sigma
    
    # 3. Combine: 
    # The "Value of Information" is highest where uncertainty is high AND the region is relevant.
    # We use a product, but the relevance term is a smooth decay, not a hard cutoff like EI.
    # This allows exploration of points with mu slightly above f_best if sigma is very high,
    # which is a key difference from EI (which gives 0 for mu > f_best in the simple case,
    # or low EI for mu >> f_best).
    
    scores = relevance * norm_sigma
    
    # 4. Add a small exploitation boost for points already better than f_best
    # To ensure we don't ignore a point that is clearly better,
    # add a linear term for mu < f_best.
    # This prevents the "pure exploration" trap if the GP finds a new minimum.
    improvement = np.maximum(0, f_best - mu)
    # Weight the improvement by a small factor to keep it secondary to the KG signal
    # but strong enough to exploit immediate wins.
    scores += 0.1 * improvement / (scale + 1e-6)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # If time is running out (unlikely for O(N) but good practice), return current
    if time.time() > t_start + deadline:
         return scores
         
    return scores
# EVOLVE-BLOCK-END
