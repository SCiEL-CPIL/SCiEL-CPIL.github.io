"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Hybrid Pareto-Front Acquisition for Minimizing Branin 2D.
    
    Strategy:
    Instead of a fixed linear combination (UCB) or a single scalar metric (EI),
    this method treats the acquisition process as a multi-objective selection problem.
    We define two objectives for each candidate:
    1. Expected Improvement (EI): Measures exploitation potential.
    2. Predictive Uncertainty (Sigma): Measures exploration potential.
    
    We compute the Pareto front of these two objectives (candidates not dominated
    by any other candidate in both EI and Sigma). From this front, we select the
    candidate that maximizes a normalized composite score. This avoids the
    sensitivity of UCB to the choice of beta and the local bias of standard EI.
    
    To ensure robustness and finite scores for all 2000 candidates (as required),
    we assign scores based on the rank of candidates within the Pareto set and
    a penalty for dominated candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e-9, neginf=1e-9)
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 1. Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # Where Phi is CDF and phi is PDF of standard normal
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # 2. Identify Pareto Front
    # A point i dominates point j if ei_i >= ei_j AND sigma_i >= sigma_j 
    # AND at least one inequality is strict.
    # We want to keep points that are NOT dominated by any other point.
    
    n = len(mu)
    is_dominated = np.zeros(n, dtype=bool)
    
    # Efficient Pareto front extraction using sorting
    # Sort by EI descending. If EI is same, sort by Sigma descending.
    # For each point in sorted order, if its Sigma is greater than the max Sigma 
    # seen so far (among points with higher or equal EI), it is on the front.
    
    # Sort indices by EI descending
    ei_sorted_idx = np.argsort(-ei)
    
    # We need to be careful with ties. 
    # Let's use a more robust O(N log N) approach.
    # Sort by EI descending.
    # Iterate through sorted points. Keep track of max_sigma_seen.
    # If current sigma > max_sigma_seen, it's on the front.
    # Update max_sigma_seen = max(max_sigma_seen, current_sigma).
    
    max_sigma_seen = -np.inf
    pareto_mask = np.zeros(n, dtype=bool)
    
    for idx in ei_sorted_idx:
        if sigma[idx] > max_sigma_seen:
            pareto_mask[idx] = True
            max_sigma_seen = sigma[idx]
        # If sigma[idx] <= max_sigma_seen, it is dominated by a previous point 
        # with higher EI and higher/equal Sigma.
        
    # 3. Score Assignment
    # We want higher scores for better candidates.
    # Candidates on the Pareto front get high scores.
    # Candidates off the front get lower scores.
    
    # To create a continuous score that respects the "higher is better" contract
    # and distinguishes within the Pareto front, we can use a rank-based score.
    
    # However, the task requires returning scores for ALL 2000 candidates.
    # The evaluator picks the argmax.
    
    # Strategy:
    # - Points on Pareto front: Score = 100 + normalized_ei + normalized_sigma
    # - Points not on Pareto front: Score = 0 + small noise or 0
    
    # Actually, a simpler and more effective structural change:
    # Use the Pareto front to define a "valid" set, but score them based on
    # a composite metric that balances the two.
    
    # Let's refine: 
    # Compute a composite score for ALL points:
    # score = w1 * norm_ei + w2 * norm_sigma
    # But weight points on the Pareto front heavily?
    
    # Alternative Structural Change: 
    # Use "Probability of Improvement" (PI) combined with "Uncertainty" via a 
    # different aggregation: The Product.
    # PI = Phi((f_best - mu)/sigma)
    # Score = PI * Sigma
    # This rewards high probability of improvement AND high uncertainty.
    # It is structurally different from UCB (linear) and EI (integral).
    
    # Let's try the Product Score (PI * Sigma) as the primary mechanism.
    # It is computationally cheap and structurally distinct.
    
    pi = norm.cdf((f_best - mu) / sigma)
    
    # The product PI * Sigma peaks when there is a reasonable chance of improvement
    # but also significant uncertainty. 
    # If sigma is very small, PI is either ~0 or ~1. 
    #   If PI~1 (mu << f_best), Score ~ 1*small = small. (Exploitation of known good)
    #   If PI~0 (mu >> f_best), Score ~ 0*small = 0.
    # If sigma is large, PI ~ 0.5 (mu ~ f_best), Score ~ 0.5*large = large. (Exploration)
    
    # This might over-explore. 
    # Let's stick to the Pareto Front idea but implement it as a filter + rank.
    
    # Score construction:
    # 1. Assign a base score of 0 to all.
    # 2. For Pareto front points, assign a score based on their position.
    #    We want the "best" Pareto point to have the highest score.
    #    What is "best"? Usually, we want the one closest to the "ideal" point (max EI, max Sigma).
    #    Since we can't have both max, we use the rank.
    
    # Let's use a simpler, robust structural change:
    # **Weighted Sum with Adaptive Weights based on Posterior Statistics**
    # Instead of fixed beta, we compute beta dynamically based on the distribution
    # of sigma across the candidates.
    
    # If the GP is very confident (low sigma everywhere), we should exploit more (lower beta).
    # If the GP is uncertain (high sigma), we should explore more (higher beta).
    
    # Let's try: beta = 1.0 * (median(sigma) / (median(sigma) + epsilon))
    # This doesn't change much if sigma is constant.
    
    # Let's go back to the **Pareto Front** implementation, it's the most "different".
    
    # To make it work for argmax:
    # We assign scores such that the Pareto front points have scores > 100.
    # Among them, we rank by a composite score: 
    # Composite = (ei / max_ei) + (sigma / max_sigma)
    # This favors points that are good in BOTH.
    
    max_ei = np.max(ei) if np.max(ei) > 0 else 1e-9
    max_sigma = np.max(sigma) if np.max(sigma) > 0 else 1e-9
    
    # Normalize
    ei_norm = ei / max_ei
    sigma_norm = sigma / max_sigma
    
    # Composite score
    composite = ei_norm + sigma_norm
    
    # Final Scores:
    scores = np.zeros(n, dtype=np.float64)
    
    # Only Pareto points get the composite score scaled up
    scores[pareto_mask] = 100.0 + composite[pareto_mask]
    
    # Non-Pareto points get 0.
    # This ensures the argmax is always on the Pareto front.
    # Among Pareto points, the one with the highest sum of normalized EI and Sigma wins.
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=100.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
