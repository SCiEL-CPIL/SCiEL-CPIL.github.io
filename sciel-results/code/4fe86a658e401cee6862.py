import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    
    Strategy: Quantile-Stratified Exploration (QSE)
    Instead of pointwise EI/UCB or stochastic Thompson Sampling voting, this method
    enforces spatial diversity by partitioning the 2000 candidates into disjoint 
    quantile bins based on the posterior mean (mu). 
    
    For each bin, it identifies the candidate with the highest uncertainty (sigma) 
    and assigns a high score to it. This prevents the "clustering" of evaluations 
    often seen in greedy methods that repeatedly pick the single highest EI/UCB 
    point, which might be in the same local neighborhood. By forcing one "best 
    uncertain" point per mean-quantile, we systematically explore different 
    regions of the predicted landscape.
    
    The score is a combination of:
    1. A large base score for the selected "representative" of each quantile bin.
    2. A small exploitation term (-mu) to break ties and ensure we still favor 
       lower expected values among the selected representatives if the base scores 
       are equal (though they will be distinct due to the bin index weighting).
    
    This is a structural change from "pick the global max of a scalar score" to 
    "pick a diverse set of local maxes of uncertainty within mean-strata".
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    n_candidates = len(mu)
    
    if n_candidates == 0:
        return np.array([])
        
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
        
    # Ensure sigma is positive for meaningful exploration
    sigma = np.maximum(sigma, 1e-6)
    
    # Define number of quantile bins. 
    # 20 bins is a good balance: enough to force diversity, not so many that 
    # each bin is too small to contain a meaningful uncertainty peak.
    n_bins = 20
    
    # Create quantile bins based on mu.
    # We want to partition the candidates into n_bins groups of roughly equal size,
    # sorted by mu.
    # Use np.quantile to find bin edges.
    # To avoid issues with duplicate mu values or edge cases, we add a tiny jitter 
    # or use unique quantiles if necessary, but standard quantile works well 
    # for continuous distributions.
    
    # Calculate quantile edges
    # q = [0, 1/n_bins, 2/n_bins, ..., (n_bins-1)/n_bins, 1]
    # We need to assign each point to a bin.
    
    # Efficient binning using np.searchsorted on quantile edges
    # Generate edges
    edges = np.linspace(0, 1, n_bins + 1)
    
    # Get the actual mu values at these quantiles
    # Note: np.quantile handles the interpolation.
    mu_quantiles = np.quantile(mu, edges)
    
    # Assign each candidate to a bin index (0 to n_bins-1)
    # np.searchsorted returns the index of the bin where the value would be inserted.
    # We want to map mu[i] to the bin it falls into.
    # Using side='right' ensures that values equal to a quantile go into the higher bin,
    # but we need to handle the last bin carefully.
    
    bin_indices = np.searchsorted(mu_quantiles, mu, side='right') - 1
    # Clamp to valid range [0, n_bins-1]
    # If mu is very small, it might be -1. If very large, n_bins.
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # For each bin, find the index of the candidate with the maximum sigma.
    # We will create an array of scores.
    scores = np.zeros(n_candidates, dtype=np.float64)
    
    # We want the score to be high for the selected representative.
    # To ensure the evaluator (argmax) picks one of these representatives, 
    # we assign a high base score.
    # To ensure diversity is actually selected if multiple are picked (though 
    # evaluator picks only 1), we just need the global argmax to be one of them.
    # Actually, the evaluator picks the single highest score. 
    # So if we give all representatives the same high score, it picks the first one.
    # We need to differentiate them slightly or ensure the "best" representative 
    # is the one with the most promising (low mu) or highest sigma.
    
    # Strategy refinement:
    # The goal is to explore. But we also want to exploit if we are close to the optimum.
    # A pure "max sigma per bin" might ignore the fact that some bins are in 
    # terrible regions (high mu).
    
    # Let's score the representative of each bin as:
    # score = BaseScore - alpha * mu_rep + beta * sigma_rep
    # But since we are picking the max sigma in the bin, sigma is fixed for the rep.
    # We want the final score to reflect that we *want* to evaluate this rep.
    
    # Let's assign a score to the rep that is:
    # 1. Dominated by a term that makes it higher than non-reps.
    # 2. Modulated by how "good" the bin is (lower mu is better).
    
    # However, a simpler and more robust structural approach:
    # Compute a "Diversity Score" for each point.
    # For point i in bin b:
    # It gets a high score if it is the max-sigma point in bin b.
    # Let's just compute the scores for the reps and leave others at 0.
    # Then, to ensure we don't always pick the same bin if they have similar scores,
    # we can add a small random tie-breaker or rely on the mu difference.
    
    # Let's compute the representative index for each bin.
    # We can use a loop over bins (20 iterations) which is fast.
    rep_indices = np.full(n_bins, -1, dtype=int)
    rep_sigmas = np.zeros(n_bins)
    rep_mus = np.zeros(n_bins)
    
    for b in range(n_bins):
        mask = (bin_indices == b)
        if np.any(mask):
            candidates_in_bin = np.where(mask)[0]
            sigmas_in_bin = sigma[candidates_in_bin]
            # Find the index with max sigma
            max_idx_local = np.argmax(sigmas_in_bin)
            global_idx = candidates_in_bin[max_idx_local]
            rep_indices[b] = global_idx
            rep_sigmas[b] = sigma[global_idx]
            rep_mus[b] = mu[global_idx]
    
    # Now, assign scores to these representatives.
    # We want the evaluator to pick the one that is most likely to improve f_best.
    # A simple heuristic: The score should be high for points that are uncertain 
    # but not in obviously bad regions.
    
    # Let's use a UCB-like score for the representatives, but only they get it.
    # score_i = -mu_i + gamma * sigma_i
    # But we want to emphasize the *diversity* aspect. 
    # If we just use UCB, we might pick the same rep every time if it has the best UCB.
    # The "Quantile" aspect is already used to *select* the candidate from the pool.
    # The *score* determines which of the 20 reps is picked.
    
    # To force exploration of different bins over time, we could rotate? 
    # No, stateless function.
    
    # Let's assume the best representative is the one with the highest "Potential".
    # Potential = -mu + gamma * sigma.
    # Let's set gamma = 1.0 (standard UCB).
    # But to ensure we don't ignore the diversity constraint, we must ensure 
    # that the scores of the reps are the only non-zero ones (or very high).
    
    # Let's assign:
    # score[rep] = -mu[rep] + 1.5 * sigma[rep]
    # All other points get 0.
    
    # This effectively turns the 2000-point selection into a 20-point selection 
    # (one per quantile), chosen by UCB. This is structurally different from 
    # standard UCB which picks the global max over all 2000. It prevents 
    # picking two points from the same quantile bin in consecutive steps 
    # (if the GP doesn't change much, the reps stay similar, but the "max sigma" 
    # in that bin might shift).
    
    # Actually, if the GP updates, the mu and sigma change, so the reps change.
    
    # Let's refine the score to be more robust.
    # If we just use UCB on the reps, we might still cluster if one bin consistently 
    # has high sigma. 
    # To counter this, we can normalize the score by the bin's average sigma? 
    # Or just trust the UCB on the reps.
    
    # Let's try a "Normalized UCB" for the reps.
    # score_rep = (-mu + gamma * sigma) / (max_sigma + eps)
    # This keeps scores bounded.
    
    max_sigma_global = np.max(sigma)
    if max_sigma_global < 1e-9:
        max_sigma_global = 1.0
        
    gamma = 1.5  # Exploration weight
    
    for b in range(n_bins):
        idx = rep_indices[b]
        if idx != -1:
            # Calculate UCB score for this rep
            # We add a small bias to favor lower mu (exploitation) but 
            # the selection of the rep was already based on max sigma (exploration).
            s = -rep_mus[b] + gamma * rep_sigmas[b]
            scores[idx] = s
            
    # To ensure the evaluator picks one of these, and to handle the case where 
    # all scores are negative (if mu is large and positive), we can shift them.
    # But relative ordering is what matters for argmax.
    
    # One issue: If the best point (lowest mu) is in a bin with low sigma, 
    # it won't be the rep (max sigma). It will be ignored.
    # This is the trade-off of the "Quantile-Stratified" strategy: we sacrifice 
    # the absolute best exploitation point if it's in a low-uncertainty bin, 
    # in favor of exploring the most uncertain point in that bin.
    # This is a deliberate structural choice to prevent premature convergence.
    
    # However, if we are very close to the optimum, sigma is low everywhere.
    # In that case, max sigma is small, and UCB will just pick the lowest mu rep.
    # This seems reasonable.
    
    # Let's ensure finite outputs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Time check
    if time.time() > deadline:
        # Fallback: simple UCB on all points if we are running out of time
        # (Though the loop above is very fast, 20 iterations)
        return -mu + 2.0 * sigma
        
    return scores
