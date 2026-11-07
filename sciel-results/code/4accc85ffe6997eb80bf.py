import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) Acquisition Function.
    Estimates the expected reduction in the global minimum's value by approximating
    the posterior distribution of the minimum over all candidates.
    
    For each candidate x_i, KG_i = E[f_best - min(f_best, f(x_i))]
    This encourages exploration toward regions most likely to contain the true global optimum.
    """
    sigma = np.maximum(sigma, 1e-9)
    
    # Time guard: track elapsed time to ensure we finish within budget
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s limit
    
    # Number of candidates
    n = len(mu)
    
    # Compute z-scores for all candidates relative to current best
    z = (f_best - mu) / sigma
    
    # Expected Improvement for each candidate (standard EI for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Knowledge Gradient approximation:
    # The full KG considers the joint distribution of all f(x_i).
    # We approximate by considering the probability that x_i is the new minimum
    # and the expected improvement it provides.
    
    # Probability that candidate i achieves a new minimum:
    # P(f(x_i) < f_best) = Phi(z_i) where z_i = (f_best - mu_i) / sigma_i
    p_new_min = norm.cdf(z)
    
    # For the KG, we want to estimate E[max(0, f_best - f(x_i))] but weighted by
    # the likelihood that this is truly the global minimum among all candidates.
    
    # Approximate KG as:
    # KG_i ≈ EI_i * w_i
    # where w_i accounts for the competitive landscape (other candidates might be better)
    
    # A practical approximation: use the "expected minimum improvement" concept
    # The expected minimum over all n candidates is approximately:
    # E[min(f(x_1), ..., f(x_n))] ≈ min(mu) - c * sigma_max
    # But we need per-candidate KG scores.
    
    # Better approximation: KG_i = E[f_best - min(f_best, f(x_i))]
    # This is exactly EI_i for independent candidates.
    # For correlated candidates (which they are via GP), we need to account for correlation.
    
    # Practical KG surrogate:
    # 1. Base score is EI (expected improvement over current best)
    # 2. Add a bonus for candidates in high-uncertainty regions (exploration value)
    # 3. Penalize candidates that are likely to be dominated by other high-value candidates
    
    # Step 1: Base EI score
    kg_base = ei.copy()
    
    # Step 2: Exploration bonus - candidates with high sigma that have reasonable mu
    # This encourages exploring uncertain regions
    sigma_norm = sigma / (sigma.max() + 1e-9)
    mu_norm = (mu - mu.min()) / (mu.max() - mu.min() + 1e-9)
    
    # Exploration value: high uncertainty + not obviously bad (mu not too high)
    exploration_bonus = sigma_norm * (1.0 - mu_norm) * 0.3 * ei.max()
    
    # Step 3: Competitive penalty - if many other candidates are likely to be better,
    # the KG value of this candidate is reduced
    # Approximate by counting how many candidates have lower mu + higher sigma
    # This is O(n^2) which is too slow for n=2000, so use a simpler proxy
    
    # Use the rank of mu as a proxy for competitiveness
    mu_ranks = np.argsort(np.argsort(mu)) / n  # 0 to 1, lower is better
    
    # Candidates with good mu (low rank) get less penalty
    competition_penalty = mu_ranks * 0.1 * ei.max()
    
    # Combine: KG ≈ EI + exploration_bonus - competition_penalty
    scores = kg_base + exploration_bonus - competition_penalty
    
    # Time check: if we're running out of time, return what we have
    if time.time() > deadline - 5.0:
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
        return scores
    
    # Refinement: Use a more principled KG approximation
    # The true KG for candidate i is:
    # KG_i = E[f_best - min_{j}(f(x_j))] - E[f_best - min_{j≠i}(f(x_j))]
    # 
    # Approximate E[min_j f(x_j)] using the fact that for correlated Gaussians,
    # the minimum is approximately:
    # E[min] ≈ min(mu) - k * sqrt(var(min))
    # where k depends on the correlation structure
    
    # Simpler practical approach: use the "probability of being the minimum" 
    # as a weight on the EI
    
    # Probability that candidate i is the argmin among all candidates
    # P(f(x_i) = min_j f(x_j)) ≈ exp(-lambda * (mu_i - mu_min)) * exp(-gamma * sigma_i)
    # This is a heuristic approximation
    
    mu_min = mu.min()
    sigma_max = sigma.max()
    
    # Decay rates (tuned for the problem scale)
    lambda_decay = 5.0 / (mu.max() - mu_min + 1e-9)
    gamma_decay = 2.0 / (sigma_max + 1e-9)
    
    p_argmin = np.exp(-lambda_decay * (mu - mu_min)) * np.exp(-gamma_decay * sigma)
    p_argmin = p_argmin / (p_argmin.sum() + 1e-9)  # Normalize
    
    # Weighted KG: candidates that are likely to be the global minimum get higher scores
    kg_weighted = ei * (1.0 + p_argmin * n / 10.0)  # Scale the boost
    
    # Blend the two approximations
    scores = 0.5 * scores + 0.5 * kg_weighted
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
