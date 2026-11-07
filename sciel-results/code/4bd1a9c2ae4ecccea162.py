import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 310  # 310s out of 330s budget, extra safety margin

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() > _DEADLINE:
        return -mu / (cost + 1e-9)
    
    # Ensure inputs are clean
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Structural Change: Uncertainty-Weighted Improvement (UWI) with local baseline
    # Instead of using global f_best as the only baseline, we estimate a local
    # "expected floor" using the GP mean itself, which captures local landscape structure.
    # This targets basin boundaries where variance is high relative to local predictions.
    
    # Compute z-score relative to global best (standard EI component)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # Standard Expected Improvement from global best
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0)
    
    # Local baseline approach: use mu as a soft local reference
    # Points where sigma is large relative to |mu - f_best| are at basin boundaries
    # UWI: rewards high uncertainty in regions that are not obviously bad (mu < f_best)
    # The ratio sigma / (|mu - f_best| + sigma) is a normalized uncertainty measure
    # that peaks at basin boundaries
    
    # Distance from best (how much worse than best)
    dist_from_best = np.maximum(f_best - mu, 0)
    
    # Normalized uncertainty: high when sigma >> dist_from_best (uncertain about improvement)
    # Low when sigma << dist_from_best (confident it's worse)
    # This is essentially the probability of improvement weighted by uncertainty
    norm_uncertainty = sigma / (dist_from_best + sigma)
    
    # UWI score: combines EI with uncertainty-weighted exploration
    # The norm_uncertainty term acts as a local exploration bonus that
    # is strongest at basin boundaries (high sigma, moderate dist_from_best)
    uwi = ei + 0.8 * norm_uncertainty * sigma
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    score = uwi / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
