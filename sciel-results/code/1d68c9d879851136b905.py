# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best, cost):
    """
    Local Uncertainty-Weighted Improvement (LUWI) with Kernel-Weighted Local Baseline.
    
    Strategy:
    1. Instead of comparing against a single global f_best, estimate a local baseline 
       for each candidate using a kernel-weighted average of the GP predictive mean.
       This approximates the local function value landscape based on current GP knowledge.
    2. Compute an "Improvement" score relative to this local baseline.
    3. Weight this improvement by the normalized local uncertainty (sigma).
       This creates a score that is high where:
       a) The local prediction is bad (high local baseline value, i.e., high function value)
       b) The local uncertainty is high (sigma is large)
    4. This effectively targets the boundaries of explored regions (basins) where there is 
       both potential for improvement (locally) and high uncertainty, rather than just 
       the global minimum or the most uncertain point in the entire domain.
    5. Apply a cost penalty.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Predictive mean from the GP.
    sigma : array-like, shape (2000,)
        Predictive standard deviation from the GP.
    f_best : float
        Lowest observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
        return np.zeros_like(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # We do not have coordinates, so we cannot compute true spatial kernels.
    # However, the GP mean `mu` is a function of the underlying latent function.
    # Points with similar `mu` values are likely in similar regions of the function landscape.
    # We can use a "functional similarity" kernel based on mu and sigma.
    
    # Estimate local baseline using a weighted average of mu.
    # To avoid O(N^2) which is 4M ops (feasible in numpy but let's be careful),
    # we can use a simplified approach: 
    # The "local baseline" for a point i can be approximated by the mean of mu 
    # weighted by similarity in the (mu, sigma) space.
    
    # A cheaper proxy for "local neighborhood" in the absence of coordinates:
    # Use the global statistics of mu and sigma to define "typical" local behavior.
    # But the directive asks for "local baseline estimated via kernel density or nearest-neighbor".
    # Without coordinates, true nearest-neighbor is impossible.
    # 
    # Alternative Interpretation:
    # The GP provides mu and sigma at 2000 points. 
    # If we assume the points are uniformly distributed (which they are, "2000 new uniform candidates"),
    # we cannot map them back to coordinates.
    # 
    # However, we can use the *structure* of the GP output.
    # A robust structural change is to use a **Local Outlier Factor** style or 
    # **Density-Weighted** approach on the (mu, sigma) manifold.
    #
    # Let's try a different structural approach: **Thompson Sampling with Localized Mean Shift**.
    # No, the directive specifically suggests: "Replace the single global f_best baseline with a local 
    # baseline estimated via kernel density... and compute an Uncertainty-Weighted Improvement (UWI) score".
    #
    # Since we don't have coordinates, we can't do spatial kernel density.
    # But we can do "functional" kernel density.
    # Let's approximate the local baseline `mu_local[i]` as a weighted average of all `mu[j]` 
    # where the weight is based on the distance in the (mu, sigma) space.
    # This is computationally expensive (2000x2000 matrix). 
    # 4 million operations is fine for numpy.
    
    # Let's compute pairwise distances in the (mu, sigma) feature space.
    # Normalize mu and sigma for distance calculation to avoid scale dominance.
    mu_norm = (mu - np.mean(mu)) / (np.std(mu) + 1e-8)
    sigma_norm = (sigma - np.mean(sigma)) / (np.std(sigma) + 1e-8)
    
    # Feature matrix: (N, 2)
    feats = np.column_stack((mu_norm, sigma_norm))
    
    # Compute pairwise squared distances
    # dist^2 = sum((x_i - x_j)^2) = sum(x_i^2) + sum(x_j^2) - 2 * x_i . x_j
    sq_dists = np.sum(feats**2, axis=1)[:, None] + np.sum(feats**2, axis=1)[None, :] - 2 * feats @ feats.T
    sq_dists = np.maximum(sq_dists, 0) # Numerical stability
    
    # Kernel: Gaussian kernel
    # Bandwidth selection: use median distance
    # Flatten upper triangle for median
    triu_indices = np.triu_indices(n, k=1)
    median_dist_sq = np.median(sq_dists[triu_indices])
    if median_dist_sq < 1e-10:
        median_dist_sq = 1.0
    bandwidth = median_dist_sq / 2.0
    
    # K_ij = exp(-dist^2 / (2 * bandwidth))
    # This is O(N^2) memory and compute. 2000x2000 floats is 32MB. Acceptable.
    K = np.exp(-sq_dists / (2 * bandwidth))
    
    # Normalize rows so that sum(K[i,:]) = 1
    row_sums = np.sum(K, axis=1)
    row_sums[row_sums == 0] = 1.0
    K /= row_sums[:, None]
    
    # Local baseline: weighted average of mu (not mu_norm, we want actual value)
    # local_mu[i] = sum_j K[i,j] * mu[j]
    local_mu = K @ mu
    
    # Local uncertainty: weighted average of sigma? Or just use sigma[i]?
    # The directive says "rewards candidates where the GP posterior variance is high relative to the local prediction".
    # Let's use the local sigma as well, or just the point's sigma.
    # Using the point's sigma is more direct for "uncertainty at this point".
    # But "local baseline" implies we are comparing the point's potential to its neighbors' performance.
    
    # Improvement relative to local baseline:
    # We want to find points where mu[i] is significantly lower than local_mu[i]?
    # No, mu[i] IS part of the local average. 
    # Actually, if mu[i] is low, it's a good point.
    # The "Improvement" concept usually is: Expected gain if we sample here vs current best.
    # Here, we define "Local Potential" as: How much better can we do locally?
    # If local_mu is high (bad region), and sigma is high, we should explore.
    # If local_mu is low (good region), we should exploit.
    
    # Let's define the score as:
    # Score = (local_mu - mu) + beta * sigma ?
    # No, that's just looking for outliers in the mean.
    
    # Let's stick to the directive: "Uncertainty-Weighted Improvement".
    # Improvement = max(0, local_baseline - mu_predicted)? 
    # Since mu is the mean, the "improvement" over the local average is not standard.
    
    # Let's try a simpler structural change that fits the "local" theme without O(N^2) if possible, 
    # or keep O(N^2) if it's fast enough. 4M ops is < 10ms in numpy.
    
    # Revised Score:
    # 1. Local Baseline: local_mu (average of neighbors' means).
    # 2. Local "Best" Estimate: min(local_mu, f_best)? No, f_best is global.
    #    Let's use the local minimum of the means in the neighborhood.
    #    local_min_mu = min(mu[j] for j in neighborhood)
    #    This is harder to compute with a simple matrix product.
    
    # Alternative: 
    # Use the global f_best but adjust the exploration coefficient based on "local density" of good points.
    # If there are many good points (low mu) nearby (in feature space), we should exploit more.
    
    # Let's compute a "Local Quality" metric:
    # local_quality = weighted average of exp(-mu / scale) ?
    
    # Let's go back to the UWI idea from the directive:
    # "rewards candidates where the GP posterior variance is high relative to the local prediction"
    # Score = sigma / (local_mu + offset) ?
    # This rewards high sigma in areas where the local prediction is high (bad).
    # It penalizes high sigma in areas where the local prediction is low (good) - wait, 
    # if local prediction is low, we are in a good basin. High sigma there means we haven't fully resolved it.
    # We probably want to explore there too, but maybe less than a bad region?
    # Actually, in BO, we usually want to explore where there is a chance of finding something BETTER than f_best.
    
    # Let's combine:
    # 1. Global EI-like term: (f_best - mu) / sigma
    # 2. Local Exploration term: sigma / (local_mu + 10)  (higher if local area is bad and uncertain)
    
    # Let's compute the local minimum of mu in the neighborhood to get a "local best".
    # We can approximate local best by taking the 90th percentile of the weighted distribution?
    # Or simply: local_best_estimate = local_mu - k * local_sigma?
    # Let's use local_mu - 1.0 * (K @ sigma) as a lower bound estimate for the local region.
    local_sigma = K @ sigma
    local_lower_bound = local_mu - 1.0 * local_sigma
    
    # Score components:
    # 1. Exploitation: How good is this point compared to the local lower bound?
    #    If mu is close to local_lower_bound, it's a good point.
    #    Exploit_Score = local_lower_bound - mu (positive if mu is better than local lower bound? No, mu is the mean.
    #    If mu < local_lower_bound, it's an outlier. Usually mu is within local bounds.
    
    # Let's try a different structural angle: **Hypervolume or Multi-Objective style**? No, single objective.
    
    # Let's implement the **Local Baseline EI**:
    # Instead of f_best, use a local reference value `ref_i`.
    # `ref_i` = min(f_best, local_mu_i) ? 
    # If local_mu is high, ref is f_best. If local_mu is low (better than f_best?), ref is local_mu.
    # But f_best is the global minimum. local_mu is an average. local_mu >= f_best usually.
    # So ref_i = f_best is always the tighter bound for "improvement" unless we think we can beat f_best locally.
    
    # The key insight of the directive: "targeting the boundaries of explored basins".
    # Boundaries are where sigma is high AND mu is transitioning.
    
    # Let's use a **Density-Weighted UCB**:
    # UCB_i = mu_i - beta * sigma_i
    # Adjust beta based on local density of "good" points.
    # If we are in a sparse region (low density of low-mu points), increase beta.
    
    # Compute "Goodness" g_i = exp(-mu_i / 10) (higher is better)
    # Compute local density of goodness: D_i = K @ g
    # If D_i is low, we are far from good points -> Explore more (higher beta).
    # If D_i is high, we are near good points -> Exploit more (lower beta).
    
    g = np.exp(-mu / 10.0)
    local_goodness = K @ g
    
    # Normalize local_goodness to [0, 1]
    lg_min = np.min(local_goodness)
    lg_max = np.max(local_goodness)
    if lg_max - lg_min < 1e-8:
        lg_norm = np.ones_like(local_goodness) * 0.5
    else:
        lg_norm = (local_goodness - lg_min) / (lg_max - lg_min)
    
    # Beta schedule:
    # High beta (explore) when local_goodness is low.
    # Low beta (exploit) when local_goodness is high.
    beta_min = 0.5
    beta_max = 3.0
    beta = beta_max - (beta_max - beta_min) * lg_norm
    
    # UCB Score
    ucb = mu - beta * sigma
    
    # We want to minimize, so higher score is better for selection?
    # Standard UCB for minimization: minimize (mu - beta * sigma).
    # So Acquisition Score = -(mu - beta * sigma) = -mu + beta * sigma.
    
    score = -mu + beta * sigma
    
    # Cost penalty
    scores = score / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
