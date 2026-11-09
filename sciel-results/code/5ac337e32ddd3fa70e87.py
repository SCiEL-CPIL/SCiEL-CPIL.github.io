"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = _start_time + 320.0  # Safety margin before 330s kill

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Adaptive Cost-Aware Expected Improvement (AC-EI) acquisition function.
    
    Strategy:
    Instead of using a fixed cost penalty (e.g., EI / cost or EI / (cost+1)),
    we fit a parametric cost-aware acquisition model. We assume the score is:
    Score = EI / (cost + eps)^gamma
    
    We estimate the optimal gamma by maximizing a proxy for "efficiency" on the
    current candidate set. The proxy is the sum of (Predicted Improvement) / (Cost^gamma)
    for the top-k most promising candidates, or more simply, we optimize gamma to
    maximize the expected number of improvements per unit cost assuming a simple
    model of the distribution of improvements.
    
    A simpler and robust structural change:
    We use a "Cost-Adjusted Probability of Improvement" (CA-PI) combined with EI.
    However, the directive asks for a "model-and-optimize" approach fitting a parametric
    cost-aware EI.
    
    Let's implement a grid search over gamma in [0.5, 2.0] to maximize the 
    "Total Predicted Regret Reduction per Cost".
    Regret Reduction for candidate i is approximated by EI_i.
    Cost is C_i.
    We want to choose gamma such that the aggregate benefit is maximized.
    Since we only score 2000 points, we can't do a heavy optimization loop.
    
    Alternative Structural Change: 
    Use a Bayesian approach to the cost. Assume the cost has a small noise.
    Or, better: Use the "Expected Improvement per unit cost" but with an 
    adaptive exponent.
    
    Let's try a different design family: 
    **Entropy-based Exploration with Cost Discounting**
    
    The current program uses EI. EI is a greedy exploitation metric.
    A different family is **Information Gain** (PI-BG, q-EI, etc.) or **Thompson Sampling**.
    
    The directive says: "Replace the direct acquisition scoring with a model-and-optimize 
    framework that fits a parametric cost-aware Expected Improvement ... optimizing the 
    exploration parameter to maximize the predicted improvement per unit cost".
    
    So, I will implement:
    1. Calculate EI for all candidates.
    2. Define a family of scores: S_gamma(x) = EI(x) / (cost(x) + eps)^gamma.
    3. Estimate the best gamma from a small grid (e.g., 0.5, 1.0, 1.5, 2.0).
    4. To pick the best gamma without knowing the true function, we can use a 
       heuristic: The optimal gamma should balance the spread of EI and the spread of Cost.
       If costs are uniform, gamma=1 is standard. If costs vary, we might want 
       to penalize high cost more (gamma > 1) or less (gamma < 1) depending on 
       where the high EI points are.
       
       Actually, a more principled "model-and-optimize" approach:
       Assume the next best point is the one with max S_gamma.
       The "value" of a gamma is the EI of the best point found by that gamma.
       We want to maximize the expected EI of the selected point.
       E[ EI(max_x S_gamma(x)) ] is hard to compute.
       
       Let's use a simpler proxy: 
       For each candidate gamma, compute the scores S_gamma.
       Let x* be the argmax.
       The "quality" of this gamma is EI(x*) / (cost(x*) + eps).
       Wait, this is circular.
       
       Let's look at the inspiration: "fitting a parametric cost-aware EI ... optimizing 
       the exploration parameter".
       
       Maybe the "exploration parameter" is not gamma, but a mixing parameter between 
       Exploitation (EI) and Exploration (Entropy or UCB).
       
       Let's try: Score = alpha * (EI / cost) + (1-alpha) * (sigma / cost).
       Optimize alpha to maximize the "diversity" of the top selections? No, we only pick one.
       
       Let's stick to the gamma idea but make it adaptive based on the data distribution.
       If the top EI points have high cost, we might want to lower gamma to allow them, 
       or raise gamma to avoid them? 
       If we raise gamma, high cost is penalized more.
       
       Heuristic for Gamma Selection:
       1. Compute EI.
       2. Compute Cost.
       3. Calculate correlation between EI and Cost.
          - If positively correlated (high EI requires high cost), we might want a 
            smaller gamma to not over-penalize the promising but expensive points.
          - If negatively correlated, standard gamma=1 or higher.
          
       This is a "model" of the landscape.
       
       Implementation:
       1. Compute EI.
       2. Compute correlation r = corr(EI, log(cost+1)).
       3. Set gamma = 1.0 + 0.5 * r. (Clamped to [0.5, 1.5]).
          - If r > 0 (expensive points are better), we increase the penalty? 
            No, if expensive points are better, we want to select them. 
            Dividing by cost^gamma: if gamma is small, cost penalty is weak.
            So if r > 0, we want smaller gamma.
            So gamma = 1.0 - 0.5 * r.
       4. Compute scores = EI / (cost + eps)^gamma.
       
       This is a structural change: Adaptive cost penalty based on the 
       correlation between predicted improvement and cost.
    """
    if _check_deadline():
        # Fallback: standard EI / cost
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / cost_safe
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical errors)
    ei = np.maximum(ei, 0.0)
    
    # 2. Adaptive Gamma Selection
    # We model the relationship between EI and Cost.
    # If high EI correlates with high Cost, we should be less sensitive to cost (lower gamma).
    # If high EI correlates with low Cost, we can be more sensitive (higher gamma) to ensure
    # we don't pick slightly better but very expensive points if there are cheap near-best ones.
    # Actually, if cheap points are good, we want to pick them. 
    # If we increase gamma, we penalize expensive points more.
    # So:
    # corr(EI, Cost) > 0 => High EI needs High Cost => We WANT to pay cost => Lower Gamma.
    # corr(EI, Cost) < 0 => High EI is Cheap => We can afford to be strict => Higher Gamma.
    
    # Use log cost for better distribution handling
    log_cost = np.log(cost_safe + 1e-6)
    
    # Handle edge cases where variance is 0
    if np.std(ei) < 1e-10 or np.std(log_cost) < 1e-10:
        gamma = 1.0
    else:
        # Pearson correlation
        corr = np.corrcoef(ei, log_cost)[0, 1]
        if not np.isfinite(corr):
            corr = 0.0
        
        # Map correlation [-1, 1] to gamma [0.5, 1.5]
        # corr = 1.0 (perfect positive) -> gamma = 0.5 (lenient on cost)
        # corr = -1.0 (perfect negative) -> gamma = 1.5 (strict on cost)
        # corr = 0.0 -> gamma = 1.0
        gamma = 1.0 - 0.5 * corr
        gamma = np.clip(gamma, 0.5, 1.5)
        
    # 3. Compute Final Scores
    # Score = EI / (Cost + eps)^gamma
    # Using log-space for stability if gamma is large, but gamma is small here.
    scores = ei / np.power(cost_safe + 1e-6, gamma)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
