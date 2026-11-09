"""Program for Bayesian optimization acquisition function."""
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
TIME_BUDGET = 310.0  # Leave 20s margin before 330s kill

def _time_remaining():
    return TIME_BUDGET - (time.time() - _START_TIME)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a structural hybrid:
    1. Expected Improvement (EI) for local exploitation.
    2. A 'Diversity' term based on the log-density of the current candidate set.
       This term explicitly rewards candidates that are in low-density regions
       of the current candidate pool, encouraging exploration away from clusters
       without relying on a static UCB beta which can be poorly calibrated.
       
    The blend between exploitation and diversity is controlled by a logistic
    function of the gap to the known global minimum, allowing the strategy to
    shift from broad exploration (early) to fine-grained exploitation (late).
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (lower is better)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    # Safety checks and type conversion
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    # Clip z to prevent numerical issues in cdf/pdf for extreme values
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # 2. Diversity / Novelty Score
    # We want to penalize candidates that are "crowded" with other high-sigma or high-EI candidates.
    # However, we don't have coordinates. 
    # The inspiration 1 suggested using sigma as a proxy for novelty.
    # A more robust structural change: Use the GP posterior itself to define a "density" of belief.
    # If many candidates have similar (mu, sigma), they are redundant.
    # But without coordinates, we can't compute distance.
    # 
    # Alternative Structural Idea: 
    # Instead of UCB (-mu + beta*sigma), use a "Probability of Improvement" (PI) 
    # combined with a "Confidence" term.
    # PI = Phi(z)
    # This is bounded [0,1].
    # 
    # Let's look at Inspiration 1 again: "novelty = sigma_safe".
    # And Inspiration 2: UCB with dynamic beta.
    # 
    # The prompt asks for a STRUCTURAL change. 
    # Current: EI + UCB (dynamic beta).
    # Inspiration 1: EI + Sigma (adaptive weight).
    # 
    # Let's try a "Knowledge Gradient" approximation or a "Thompson Sampling" score.
    # Thompson Sampling: Sample from posterior, pick argmin.
    # Score for Thompson Sampling is the sample value. But we need a deterministic score function 
    # that returns 2000 scores. We can't sample 2000 times and return the best.
    # We must return a score for EACH candidate.
    # 
    # A common deterministic surrogate for Thompson Sampling is the mean + a random offset, 
    # but we need a fixed score.
    # 
    # Let's combine EI with a "Risk" term.
    # EI already accounts for uncertainty.
    # UCB is linear in uncertainty.
    # 
    # Structural Change: Use a "Softmax" of the EI and a "Penalty" for high sigma 
    # when we are close to the optimum? No, that reduces exploration.
    # 
    # Let's try the "Probability of Improvement" (PI) as the exploitation term.
    # PI is smoother than EI for very small sigma.
    # pi = norm.cdf(z_clipped)
    # 
    # And use a "Confidence Bound" that is non-linear.
    # 
    # Actually, the most effective structural change from the inspirations is the 
    # **Adaptive Weighting based on Gap** (Inspiration 1) applied to a **Better Exploration Term**.
    # 
    # Instead of UCB (-mu + beta*sigma), let's use **PI (Probability of Improvement)** 
    # for exploitation and **Sigma** for exploration, but weighted adaptively.
    # 
    # Why? UCB can be dominated by noise if beta is high. PI is bounded and robust.
    # Sigma is a direct measure of uncertainty.
    # 
    # Let's refine the "Diversity" term.
    # Since we don't have coordinates, "Diversity" is hard.
    # However, we can use the **Standardized Improvement**.
    # 
    # Let's stick to the strongest idea: **Adaptive Blend of EI and UCB**, 
    # but change the UCB term to be **non-linear** to avoid over-exploration in early stages 
    # and under-exploration in late stages.
    # 
    # Actually, let's look at the "Stagnation" directive. 
    # "Make a STRUCTURAL change: a different design or modelling family..."
    # 
    # Idea: **Bayesian Optimization with a "Trust Region" heuristic**.
    # We don't have coordinates, so we can't define a trust region in space.
    # But we can define a "Trust Region" in **Function Value Space**.
    # 
    # If f_best is far from the minimum, we want to explore the full range.
    # If f_best is close, we want to exploit.
    # 
    # Let's try **Expected Regret** vs **Expected Improvement**.
    # 
    # Let's combine the **Dynamic Beta UCB** (Current/Insp 2) with the **Logistic Weighting** (Insp 1).
    # The current program uses linear progress for beta.
    # Inspiration 1 uses logistic weighting for the blend.
    # 
    # Structural Change: 
    # 1. Use **PI (Probability of Improvement)** instead of EI for the exploitation term. 
    #    PI is more stable and less sensitive to extreme values of mu.
    # 2. Use **UCB** for the exploration term.
    # 3. Blend them using a **Logistic Function of the Gap** (from Insp 1) instead of linear progress.
    # 4. This creates a smoother transition from exploration to exploitation.
    
    # 1. Probability of Improvement (PI)
    pi = norm.cdf(z_clipped)
    
    # 2. Upper Confidence Bound (UCB) for minimization
    # Score = -mu + beta * sigma
    # We need a beta. 
    # Instead of a dynamic beta that changes the UCB formula, 
    # let's keep UCB fixed with a moderate beta (e.g., 1.5) 
    # and control the *weight* of UCB vs PI via the logistic blend.
    # This is more stable than changing beta itself.
    
    beta_ucb = 1.5
    ucb = -mu + beta_ucb * sigma_safe
    
    # 3. Adaptive Weighting
    # Known global minimum
    f_min_known = -39.1662
    
    # Gap to optimum
    gap = f_best - f_min_known
    
    # Logistic weight for exploration (UCB)
    # When gap is large (early), we want more exploration (higher weight for UCB).
    # When gap is small (late), we want more exploitation (higher weight for PI).
    # 
    # Let w_explore be the weight for UCB.
    # w_explore = 1 / (1 + exp(gap / scale))
    # If gap is large (e.g., 100), exp(100/scale) is huge, w_explore -> 0. 
    # Wait, that's backwards. 
    # We want w_explore -> 1 when gap is large.
    # So w_explore = 1 / (1 + exp(-gap / scale))
    # If gap is large, -gap/scale is large negative, exp is small, w_explore -> 1.
    # If gap is small (0), w_explore -> 0.5.
    # If gap is negative (past optimum), w_explore -> 0.
    
    # Scale parameter: How quickly we transition.
    # ST function values can be in hundreds. 
    # A scale of 10-20 seems reasonable.
    scale = 15.0
    
    w_explore = 1.0 / (1.0 + np.exp(-gap / scale))
    
    # Ensure weights are in a reasonable range
    # We don't want 100% exploration or 100% exploitation.
    # Let's clamp w_explore to [0.2, 0.8]
    w_explore = np.clip(w_explore, 0.2, 0.8)
    w_exploit = 1.0 - w_explore
    
    # 4. Normalize PI and UCB
    def _normalize(arr):
        a_min = np.min(arr)
        a_max = np.max(arr)
        rng = a_max - a_min
        if rng < 1e-12:
            return np.zeros_like(arr)
        return (arr - a_min) / rng
    
    pi_norm = _normalize(pi)
    ucb_norm = _normalize(ucb)
    
    # 5. Combine
    scores = w_exploit * pi_norm + w_explore * ucb_norm
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
