"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Multi-Start Local Refinement Acquisition Function.
    
    Instead of just scoring the 2000 provided candidates, we:
    1. Compute EI for all candidates to identify top-k promising regions.
    2. For each top-k candidate, perform a local grid search in the index space
       (approximating the continuous space) to find a better local maximum of EI.
    3. Return the scores for the original 2000 candidates, but boosted if they are
       part of or near the refined local optima.
       
    Note: Since we only return scores for the 2000 candidates, the "refinement" 
    helps us determine the *true* EI values more accurately if the 2000 grid is sparse,
    but primarily this strategy uses the refined local optima to *re-weight* or 
    *boost* the candidates that are closest to these refined optima, effectively 
    creating a sharper peak in the acquisition landscape for the evaluator to pick.
    
    However, a simpler and more robust structural change is to use **Probability of Improvement (PI)**
    combined with a **Look-Ahead** heuristic. PI is less prone to numerical issues and 
    often better in low-noise regimes. But let's stick to the "Different Design Family" directive.
    
    Actually, the most impactful structural change for sparse candidate sets (2000) in 2D
    is **Batch-Adaptive EI** or **Local Search Refinement**.
    
    Let's implement **Local Search Refinement**:
    1. Calculate EI for all 2000 points.
    2. Identify the top 5 points.
    3. For each of these 5, check their neighbors in the sorted index space (approximating spatial locality)
       to see if a neighbor has a slightly higher EI. If so, the "true" peak is likely there.
    4. This doesn't change the return signature, but we can use this to *sharpen* the distribution.
    
    Wait, the evaluator picks the candidate with the *highest score*. If we just return EI, it picks the best of 2000.
    If the best of 2000 is not the global max of the acquisition function, we might miss it.
    But we cannot add new candidates. We can only score the existing ones.
    
    Therefore, the strategy must be a **different acquisition function** that is better at identifying the
    global optimum of the *underlying function* given the limited sample size.
    
    Let's switch to **Knowledge Gradient (KG)** approximation or **Upper Confidence Bound (UCB)** with adaptive kappa.
    UCB is structurally different from EI. It balances exploration/exploitation differently.
    UCB(x) = mu(x) + kappa * sigma(x)
    
    For minimization:
    UCB_min(x) = mu(x) - kappa * sigma(x)
    
    The current program uses EI. Let's try **UCB with adaptive kappa**.
    Kappa can be tuned based on the number of evaluations. We don't know the eval count directly,
    but we can estimate it from the "tightness" of the GP fit or just use a fixed schedule.
    
    Actually, a better structural change is **Expected Improvement with a Penalty for Uncertainty**
    or **Posterior Sampling (Thompson Sampling)**?
    Inspiration 2 tried Thompson Sampling and scored 0.5. So that's worse.
    
    Let's try **Probability of Improvement (PI)**.
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    
    PI is smoother than EI and often leads to more diverse exploration in the early stages.
    
    Let's combine PI with a small exploitation term to create a **Hybrid PI-EI**.
    Or, let's use **UCB** which is known to be competitive.
    
    Let's go with **Adaptive UCB**.
    kappa = c / sqrt(t) where t is estimated eval count.
    Since we don't have t, we can estimate t by looking at the minimum sigma.
    If min(sigma) is very small, t is large.
    
    Alternative: **Look-Ahead EI**.
    This is computationally expensive.
    
    Let's try **UCB** with a fixed kappa that is tuned for the Beale function scale.
    Beale function range is roughly [0, ~30].
    Standardized targets mean mu and sigma are in objective units.
    
    Let's use **UCB**:
    score = mu - kappa * sigma
    
    To make it adaptive, we can use the spread of mu to estimate progress.
    If mu is concentrated, we are late-stage, so reduce kappa (exploit).
    If mu is spread, we are early-stage, increase kappa (explore).
    
    Let's implement Adaptive UCB.
    """
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to simple EI if we're running out of time
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    n = len(mu)
    
    # Estimate progress
    # If the best prediction is close to 0, we are likely close to the minimum.
    # Also, if sigma is small everywhere, we have explored well.
    
    mu_min = np.min(mu)
    sigma_max = np.max(sigma)
    
    # Heuristic for kappa
    # Early on: sigma_max is large. Late on: sigma_max is small.
    # We want kappa to decrease as we get better.
    # Let's map sigma_max to kappa.
    # Beale function scale is ~1-30.
    # If sigma_max > 5, we are early. kappa ~ 2.0
    # If sigma_max < 0.1, we are late. kappa ~ 0.1
    
    # Logarithmic mapping
    log_sigma = np.log10(sigma_max + 1e-6)
    # log_sigma range: -6 to 1.5 approx
    # Map to [0.1, 2.0]
    # Normalize log_sigma
    # min_log = -6, max_log = 1.5
    norm_log = np.clip((log_sigma - (-6)) / (1.5 - (-6)), 0, 1)
    kappa = 0.1 + 1.9 * norm_log
    
    # UCB for minimization: mu - kappa * sigma
    ucb = mu - kappa * sigma
    
    # Ensure finite
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)
    
    # To make it more robust, we can blend with EI
    # score = 0.5 * normalized_uci + 0.5 * normalized_ei
    
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Normalize both to [0, 1] for blending
    def normalize(x):
        min_x = np.min(x)
        max_x = np.max(x)
        if max_x - min_x < 1e-10:
            return np.zeros_like(x)
        return (x - min_x) / (max_x - min_x)
    
    ucb_norm = normalize(ucb)
    ei_norm = normalize(ei)
    
    # Blend
    # Weight UCB more in early stages (high sigma), EI in late stages
    # norm_log is high in early stages
    w_ucb = norm_log
    w_ei = 1 - norm_log
    
    score = w_ucb * ucb_norm + w_ei * ei_norm
    
    # Scale back to a reasonable range
    score = score * (np.max(ei) + 1e-6)
    
    return score
# EVOLVE-BLOCK-END
