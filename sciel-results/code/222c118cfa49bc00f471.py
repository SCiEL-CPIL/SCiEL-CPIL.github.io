import numpy as np
from scipy.stats import norm
import time

# Global state
_start_time = None
_step_counter = 0
_last_f_best = None
_selected_indices = []

def acquisition(mu, sigma, f_best):
    global _start_time, _step_counter, _last_f_best, _selected_indices
    
    if _start_time is None:
        _start_time = time.time()
    
    # Detect new seed/run
    # If f_best is significantly higher than the last observed min, assume new run
    if _last_f_best is not None and f_best > _last_f_best + 0.5:
        _step_counter = 0
        _selected_indices = []
    else:
        _step_counter += 1
        
    _last_f_best = f_best
    
    # Total expected optimization steps is 25 (30 total evals - 5 initial)
    step = min(_step_counter, 24)
    total_steps = 25.0
    
    # Deadline guard: if we are running low on time, just return a simple heuristic
    elapsed = time.time() - _start_time
    if elapsed > 300:
        # Fallback to simple EI to ensure return
        sigma_safe = np.maximum(sigma, 1e-9)
        improvement = f_best - mu
        z = improvement / sigma_safe
        ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei

    # 1. Compute a base score for each candidate
    # We use a combination of EI and LCB to get a strong initial ranking.
    # However, the core of this strategy is the batch selection.
    # We will select a batch of size K from the 2000 candidates.
    # The evaluator picks the one with the highest score from our returned array.
    # To influence the evaluator, we can assign a high score to the best of our batch,
    # but we must return scores for ALL 2000 points.
    
    # Strategy:
    # 1. Compute a preliminary score (e.g., EI) for all 2000 points.
    # 2. Select a diverse batch of size K (e.g., 5 or 10) using greedy hypervolume maximization
    #    on the (mu, sigma) or (score, sigma) space.
    # 3. Assign a high "bonus" score to the points in the selected batch, 
    #    specifically scaling the score so that the best point in the batch (by some metric) 
    #    is likely to be picked, OR simply return the hypervolume-contribution based score.
    
    # Actually, the evaluator picks the argmax of the returned scores.
    # If we want to enforce diversity, we can't just return one point.
    # But the evaluator only picks ONE point per step.
    # The "greedy batch selection" idea usually applies when you can evaluate multiple points.
    # Here, we can only evaluate one.
    # However, we can use the batch selection logic to determine WHICH single point to pick,
    # by simulating the addition of that point to a virtual set of "promising" points
    # and maximizing the hypervolume of the set.
    
    # Let's refine the strategy:
    # We want to pick a point that maximizes the hypervolume of the set of points we have "considered" or "would evaluate".
    # Since we only evaluate one, we can maintain a history of "virtual" high-potential points?
    # No, the GP is updated with the actual evaluated point.
    
    # Alternative Interpretation of "Greedy Batch Selection" for single-point evaluation:
    # We select a candidate set S of size K from the 2000 points that maximizes diversity (hypervolume).
    # Then, we pick the point in S that has the highest "expected improvement" or "LCB" score.
    # This prevents picking a point that is locally optimal in EI but redundant with other high-EI points.
    
    # Let's implement this:
    # 1. Calculate EI for all 2000 points.
    # 2. Identify top M candidates by EI (e.g., M=50).
    # 3. From these M, select a subset B of size K (e.g., K=5) that maximizes the hypervolume
    #    of their (mu, sigma) coordinates (or transformed coordinates).
    # 4. Return a score array where the points in B have their original EI scores,
    #    but we boost the score of the point in B that has the highest EI (or LCB).
    #    Actually, to make the evaluator pick the "best" of the diverse set, we can just
    #    return the EI scores, but ensure that the global max EI is part of the diverse set?
    #    No, the global max EI might be redundant.
    
    # Let's try a different approach:
    # Compute scores for all 2000 points.
    # Score = EI + Diversity_Bonus.
    # How to compute Diversity_Bonus efficiently?
    # We can approximate the diversity contribution of a point p as the increase in hypervolume
    # if p is added to a set of "anchor" points.
    # Anchors can be the initial 5 points (if we knew them, we don't) or the top few points from the previous step?
    # We don't have access to previous points' coordinates, only mu and sigma for the current 2000.
    
    # Wait, the task says "No coordinates are passed".
    # This is a crucial constraint. We cannot compute Euclidean distances between candidates.
    # We only have mu and sigma.
    # Therefore, "spatial coverage" in the input space [0,1]^2 is not directly accessible.
    # We must define diversity in the (mu, sigma) space or rely on the GP's uncertainty.
    
    # If we can't use coordinates, "hypervolume" in (mu, sigma) space is the only option.
    # Points with high sigma and low mu are interesting.
    # Hypervolume maximization in (mu, sigma) space favors points that are far from the "nadir" point.
    
    # Let's define a score that balances exploitation (low mu) and exploration (high sigma).
    # The current program uses a blend of LCB and EI.
    # The "Hypervolume" idea in (mu, sigma) space:
    # We want to maximize the area dominated by the point relative to a reference point (worst mu, worst sigma).
    # This is essentially what LCB or EI does, but hypervolume is a more geometric measure.
    
    # Since we can't use coordinates, the "diversity-aware" aspect must be derived from the distribution of mu and sigma.
    # One way: Penalize points that are very similar to other high-scoring points in the (mu, sigma) space.
    
    # Let's implement a simple greedy diversity selection in (mu, sigma) space:
    # 1. Compute EI for all 2000.
    # 2. Sort by EI descending.
    # 3. Greedily select K points from the top M (e.g., M=100, K=10) such that the hypervolume of the selected set is maximized.
    # 4. The point to evaluate is the one in the selected set with the highest EI.
    # 5. Return a score array where the selected point has a very high score, and others have lower scores.
    #    To be safe, we can just return the EI scores, but ensure the argmax is the one we selected?
    #    If we return EI scores, the evaluator picks the global max EI.
    #    If the global max EI is not in our diverse set, we fail to enforce diversity.
    #    So we MUST modify the scores to make our selected point the global max.
    
    # Revised Plan:
    # 1. Compute EI for all 2000.
    # 2. Identify top M candidates (M=50) by EI.
    # 3. From these M, select a subset B of size K (K=5) that maximizes the hypervolume in (mu, sigma) space.
    #    Hypervolume in 2D: Area of the rectangle from (min_mu, min_sigma) to (max_mu, max_sigma)?
    #    No, hypervolume is usually defined relative to a reference point.
    #    Let's use the "dominated area" relative to a reference point (f_best, 0) or similar.
    #    Actually, simpler: Maximize the sum of distances to already selected points in (mu, sigma) space.
    #    This is a common greedy diversity metric (MaxMin).
    
    # Let's use MaxMin diversity in (mu, sigma) space among the top M EI points.
    # 1. Compute EI.
    # 2. Get top M indices by EI.
    # 3. Greedily select K points from these M to maximize the minimum distance to the selected set in (mu, sigma) space.
    # 4. The final pick is the point in this diverse set with the highest EI.
    # 5. Assign a score of 1e9 to this point, and 0 to others? No, that's risky if the GP is wrong.
    #    Better: Assign scores proportional to EI, but add a large bonus to the selected diverse point.
    
    # Let's implement this MaxMin strategy.
    
    sigma_safe = np.maximum(sigma, 1e-9)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize mu and sigma for distance calculation
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    mu_range = mu_max - mu_min if mu_max > mu_min else 1.0
    sigma_range = sigma_max - sigma_min if sigma_max > sigma_min else 1.0
    
    mu_norm = (mu - mu_min) / mu_range
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # 1. Get top M candidates by EI
    M = 50
    top_m_indices = np.argsort(ei)[-M:]
    
    # 2. Greedy MaxMin selection from top_m_indices
    K = 5  # Batch size for diversity
    selected = []
    
    # Start with the point having the highest EI
    first_pick = top_m_indices[np.argmax(ei[top_m_indices])]
    selected.append(first_pick)
    
    for _ in range(K - 1):
        # Compute min distance from each candidate in top_m_indices to the selected set
        min_dists = np.full(M, np.inf)
        for s_idx in selected:
            # Distance in (mu_norm, sigma_norm) space
            dists = np.sqrt((mu_norm[top_m_indices] - mu_norm[s_idx])**2 + 
                            (sigma_norm[top_m_indices] - sigma_norm[s_idx])**2)
            min_dists = np.minimum(min_dists, dists)
        
        # Pick the candidate with the maximum min distance
        next_pick_idx_in_m = np.argmax(min_dists)
        next_pick = top_m_indices[next_pick_idx_in_m]
        selected.append(next_pick)
        
        # Check for duplicates (shouldn't happen if distances are unique, but safe)
        if next_pick in selected[:-1]:
             # If duplicate, break or pick next best
             pass
             
    # 3. From the selected diverse set, pick the one with the highest EI
    selected_eis = [ei[s] for s in selected]
    best_in_diverse_set = selected[np.argmax(selected_eis)]
    
    # 4. Construct the final score array
    # We want the evaluator to pick best_in_diverse_set.
    # We can return the EI scores, but boost the score of best_in_diverse_set.
    # To be safe, let's return the EI scores, but add a large constant to the best_in_diverse_set.
    # However, if the EI of the global max is much higher, we might still want to pick it?
    # No, the strategy is to enforce diversity. So we force the pick.
    
    scores = ei.copy()
    scores[best_in_diverse_set] += 1000.0  # Large boost to ensure it's the max
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1000.0, neginf=-1000.0)
    
    return scores
