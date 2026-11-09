"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state to track time budget
_start_time = time.time()
TIME_LIMIT = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Thompson Sampling-like approach.
    
    Strategy:
    1. We want to maximize the expected improvement per unit cost.
    2. Standard EI is (f_best - mu) * Phi(z) + sigma * phi(z).
    3. To introduce a different strategy family, we use a "Optimistic" upper confidence bound (UCB) 
       combined with cost normalization, but adjusted to be more aggressive in exploration 
       when uncertainty is high, unlike EI which is more conservative.
       
    Actually, let's stick to a refined Cost-EI but with a twist: 
    The current seed uses standard EI / cost. 
    A materially different approach is **Probability of Improvement (PI) weighted by cost** 
    or **Upper Confidence Bound (UCB) / cost**.
    
    Let's try UCB / cost.
    UCB = mu + kappa * sigma.
    Since we are minimizing, we want low mu and high sigma (uncertainty).
    Score = (mu + kappa * sigma) / cost? No, that favors high mu.
    We want to minimize f. So we want candidates that *could* be lower than f_best.
    
    Let's look at the "Different Strategy" requirement.
    Current: EI / cost.
    
    Alternative: **Expected Regret** is similar to EI.
    Alternative: **Knowledge Gradient**? Too complex.
    
    Let's try a **Gaussian Process-Lower Confidence Bound (GP-LCB)** approach.
    LCB = mu - kappa * sigma.
    We want to maximize the score. So Score = -LCB / cost = (kappa * sigma - mu) / cost.
    This is essentially the same family as EI (both are linear combinations of mu and sigma 
    with different nonlinear weighting). EI is the optimal Bayesian acquisition.
    
    However, the prompt asks for a DIFFERENT strategy.
    Let's try **Thompson Sampling** logic adapted for batch scoring.
    In TS, we sample from the posterior and pick the best.
    Here we must return scores for 2000 points.
    We can approximate the "value" of a point by sampling a single function value from the GP posterior:
    sample_val = mu + sigma * randn
    Then we prioritize points that have a low sampled value (since we minimize).
    Score = -sample_val / cost.
    
    This is stochastic. It might be less stable than EI.
    But it explores differently.
    
    Let's try a hybrid: **Cost-Effective UCB**.
    Score = (mu - beta * sigma) / cost  <-- Wait, for minimization, we want low mu.
    If we define Score such that higher is better:
    We want to pick x with low mu and high sigma (potential).
    Score = (-mu + beta * sigma) / cost.
    
    Let's compare EI vs UCB.
    EI is non-linear in sigma. UCB is linear.
    UCB is often more robust and computationally cheaper.
    
    Let's implement a **Cost-Adjusted UCB** (Upper Confidence Bound for minimization is Lower Confidence Bound).
    LCB = mu - beta * sigma.
    We want to maximize the "desirability".
    Desirability = -LCB / cost = (beta * sigma - mu) / cost.
    
    We need to choose beta. A common heuristic is beta = log(n * pi^2 * d / 6) or similar, 
    or a fixed value like 2.0 or 3.0.
    Given the small domain and few evaluations, a fixed beta around 2.0-3.0 is reasonable.
    
    Also, we should guard against division by zero or very small costs.
    Cost is 1 + mean(x_scaled). x_scaled in [0,1]. Cost in [1, 2].
    So cost is always >= 1. No division by zero risk.
    
    Let's refine the "Different Strategy" claim.
    The seed is EI/cost.
    This is UCB/cost (specifically LCB for minimization).
    This changes the exploration-exploitation trade-off dynamics. EI is myopic and optimal for one step. 
    UCB is a heuristic that balances exploration more explicitly via the beta term.
    
    Implementation details:
    1. Calculate z = (f_best - mu) / sigma for reference, but we won't use norm.cdf.
    2. Score = (beta * sigma - mu) / cost.
    3. Handle edge cases where sigma is very small.
    
    Wait, if mu < f_best, EI is positive.
    In UCB: if mu is very low, -mu is very high (good).
    If sigma is high, beta*sigma is high (good).
    
    One issue: UCB doesn't explicitly use f_best in the formula (except implicitly in how we might tune beta).
    EI uses f_best to scale the improvement.
    If f_best is very low, EI becomes small. UCB remains the same.
    This might cause UCB to keep exploring even when we are close to the optimum, 
    whereas EI would switch to exploitation.
    
    To make it "Cost-aware" and robust, we can blend it or adjust beta based on f_best?
    No, let's stick to pure UCB/cost as the distinct strategy.
    
    However, a pure UCB might ignore the current best value's magnitude.
    Let's try a **Modified UCB** that incorporates f_best:
    Score = (f_best - mu + beta * sigma) / cost.
    This is actually very close to EI's linear approximation for large z.
    
    Let's look at **Probability of Improvement (PI)**.
    PI = Phi((f_best - mu) / sigma).
    Score = PI / cost.
    This is bounded [0, 1]. It saturates.
    EI is unbounded.
    PI is less sensitive to large improvements (once you are sure it's better, PI=1, EI keeps growing).
    PI might be better for cost-awareness because it doesn't over-value distant points that are *sure* to be better but costly?
    Actually, if a point is sure to be better (high PI), and it's cheap, it's great.
    
    Let's go with **PI / cost**. It is a distinct family (probabilistic vs expected value).
    
    Wait, the seed is EI. PI is often considered inferior to EI for BO.
    But the task asks for a DIFFERENT strategy.
    
    Let's try **Thompson Sampling** (TS) again, but deterministic per call?
    No, TS requires sampling.
    
    Let's try **Expected Improvement with a different cost model**.
    The current cost model is `1 + mean(x)`.
    The seed divides by cost.
    
    What if we use **Log-Cost** or **Inverse Cost**?
    Score = EI * (1/cost). This is the seed.
    
    Let's try **UCB with adaptive beta**.
    beta = 2.0 + 0.5 * log(1 + (f_best - min(mu))/max(sigma, 1e-6))
    This is getting too complex.
    
    Let's stick to **UCB / cost** with a fixed beta. It is simple, distinct from EI (linear vs non-linear), and fast.
    Beta = 2.5.
    
    Score = (2.5 * sigma - mu) / cost
    
    One risk: If mu is very high (bad region), -mu is very negative.
    If sigma is very low (explored region), 2.5*sigma is small.
    So it correctly avoids bad/explored regions.
    
    If mu is low (good) and sigma is high (uncertain), score is high.
    
    Let's add a small epsilon to cost to be safe, though cost >= 1.
    
    Also, we must ensure the output is finite.
    """
    # Calculate UCB-like score for minimization
    # We want to maximize: (Potential for being low) / Cost
    # Potential = beta * sigma - mu
    
    beta = 2.5
    # Ensure sigma is not zero to avoid numerical issues, though GP sigma shouldn't be 0
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Score = (beta * sigma - mu) / cost
    scores = (beta * sigma_safe - mu) / cost
    
    # Handle any non-finite values (shouldn't happen with valid inputs, but safety first)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
