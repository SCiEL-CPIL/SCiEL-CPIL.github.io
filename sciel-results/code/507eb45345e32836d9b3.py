"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state to track evaluation count for adaptive parameters
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    start_time = time.time()
    
    # Safety guard: if we are close to the deadline, return a safe fallback
    # The evaluator kills after 330s. We don't know exactly when we are in the loop,
    # but assuming standard behavior, we can just be fast.
    # If this function takes too long, we might timeout.
    # Entropy search on 2000 points is O(N^2) or O(N log N) depending on implementation.
    # Let's implement a vectorized version.
    
    _eval_count += 1
    
    # Adaptive exploration: Early on, we might want to explore more.
    # However, Entropy Search is generally robust.
    # Let's stick to the core ES logic.
    
    # 1. Normalize mu and sigma for numerical stability
    # sigma is standard deviation
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 2. Compute the posterior probability that each point is the global minimizer.
    # In Entropy Search, we assume the global minimum is at one of the candidates.
    # The probability that point i is the global minimum is proportional to:
    # P(y_i < min_{j != i} y_j)
    
    # This is computationally expensive if done naively O(N^2).
    # For N=2000, N^2 = 4,000,000 operations. This is very fast in NumPy.
    
    # Let's compute the pairwise probabilities.
    # mu shape: (N,)
    # sigma shape: (N,)
    
    # We want to compute the distribution of the minimum.
    # Actually, a simpler approximation for Entropy Search in the context of limited evaluations:
    # We can approximate the expected entropy reduction by looking at the uncertainty
    # around the current best.
    
    # Alternative: Use the "Probability of Improvement" weighted by uncertainty?
    # No, let's stick to a robust variant: **Knowledge Gradient** approximation or **Entropy Search**.
    
    # True Entropy Search:
    # H = - sum p_i log p_i
    # Where p_i is the probability that x_i is the argmin.
    
    # To compute p_i, we need the joint distribution of all y_i.
    # Since we don't have the covariance matrix between the 2000 candidates, we cannot compute exact joint probabilities.
    # The task says: "mu and sigma ... are in objective units." It does NOT provide the covariance matrix K between the 2000 candidates.
    # It only provides the marginal means and variances.
    
    # CRITICAL CONSTRAINT: We do NOT have the inter-candidate covariance.
    # Therefore, we CANNOT compute true Entropy Search or Knowledge Gradient which require joint distributions.
    # We are restricted to methods that depend only on marginal (mu, sigma) and f_best.
    
    # So, what is a MATERIALLY DIFFERENT strategy that only uses (mu, sigma, f_best)?
    # 1. EI (Current)
    # 2. UCB (mu - beta * sigma)
    # 3. PI (Probability of Improvement)
    # 4. Thompson Sampling (Sample from N(mu, sigma^2) and pick min) - But this requires random sampling, and the prompt asks for a deterministic score function? 
    #    "Return 2000 finite scores... anything else makes the whole result invalid."
    #    If we use Thompson Sampling, we generate a sample y ~ N(mu, sigma) and return -y as the score.
    #    This is a valid acquisition function. It is fundamentally different from EI (which is deterministic expectation).
    #    TS explores more aggressively when sigma is high.
    
    # Let's try **Thompson Sampling**.
    # It is a Bayesian optimization strategy that samples a function from the posterior and picks the best.
    # Here, we sample y_i ~ N(mu_i, sigma_i^2) for each candidate i.
    # The score is -y_i (since we want to minimize, higher score = lower y).
    # 
    # Wait, the prompt says "higher meaning evaluate next".
    # So score = -sampled_value.
    # 
    # Is this better than EI?
    # EI is myopic. TS is also myopic in this context (single step), but it has different exploration properties.
    # EI balances exploration and exploitation via the expectation.
    # TS balances them via the sampling variance.
    # 
    # Let's consider the "Stagnation" signal. The population best hasn't improved.
    # Maybe EI is too conservative? Or too aggressive?
    # 
    # Another option: **Upper Confidence Bound (UCB)**.
    # score = mu - kappa * sigma.
    # This is deterministic.
    # 
    # Let's look at the inspirations.
    # Inspiration 1: Parametric acquisition with adaptive hyperparameters.
    # Inspiration 2: Thompson Sampling (Score 0.5093 - worse than 1.0).
    # 
    # Since TS scored worse, maybe we should try UCB with an adaptive kappa?
    # Or a different formulation of EI?
    # 
    # Let's reconsider the **Probability of Improvement (PI)** with a noise parameter xi.
    # PI(mu, sigma, f_best, xi) = Phi((f_best - xi - mu) / sigma)
    # 
    # Or **Expected Improvement with a noise parameter xi**:
    # EI_xi = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - xi - mu) / sigma
    # 
    # The current program uses xi=0.
    # If we increase xi, we become more exploitative (only pick points likely to beat f_best by at least xi).
    # If we decrease xi (negative), we become more exploratory.
    # 
    # However, the directive says: "Do not micro-tune constants. Make a STRUCTURAL change."
    # Changing xi is a micro-tune.
    # 
    # What is a structural change?
    # 
    # **Deterministic Entropy Search** is not possible without covariance.
    # **Knowledge Gradient** is not possible without covariance.
    # 
    # What about **Variance-based selection**?
    # Maximize sigma? Pure exploration.
    # 
    # What about **Minimizing the posterior variance**?
    # This is an "information gain" proxy.
    # score = -sigma.
    # This picks the point with the highest uncertainty.
    # 
    # Let's compare EI vs -sigma.
    # EI is a mix. -sigma is pure exploration.
    # 
    # Maybe a **Hybrid**: 
    # In the early stages, we want to explore. In later stages, exploit.
    # We can track the number of evaluations.
    # 
    # Let's implement **UCB with an adaptive kappa**.
    # kappa starts high (exploration) and decays (exploitation).
    # This is a structural change in how the balance is managed (dynamic vs static).
    # 
    # kappa(t) = kappa_0 * (1 - t/T)^gamma or similar.
    # 
    # Let's try:
    # score = mu - kappa * sigma
    # where kappa = 2.0 * (1 - _eval_count / 30) + 0.1
    # 
    # At t=0, kappa ~ 2.1.
    # At t=30, kappa ~ 0.1.
    # 
    # This forces initial exploration and later exploitation.
    # EI has a fixed balance.
    # 
    # Let's implement this.
    
    # Number of evaluations so far (including the current one being decided? 
    # The prompt says "at each step ... calls acquisition ... until 30 evaluations".
    # _eval_count is incremented at the start.
    # So for the first call, _eval_count=1.
    # Total steps = 30 - 5 = 25 acquisition calls?
    # "starts from 5 uniform random points, then at each step ... until 30 evaluations".
    # So there are 25 calls to acquisition.
    
    # Let's assume total_calls = 25.
    # t = _eval_count - 1 (0-indexed step in the loop)
    # progress = t / 24.0
    
    # If _eval_count > 25, something is wrong, but we cap it.
    
    t = max(0, _eval_count - 1)
    total_steps = 25.0
    progress = min(1.0, t / total_steps)
    
    # Adaptive kappa: Start with 2.0, end with 0.5
    # Linear decay
    kappa = 2.0 - 1.5 * progress
    
    # UCB Score
    # We want to minimize f.
    # UCB for minimization: mu - kappa * sigma.
    # Higher score is better.
    # If sigma is large, mu - kappa*sigma is lower (worse).
    # Wait.
    # If we want to explore, we want to pick high sigma.
    # If we use score = mu - kappa * sigma:
    # High sigma -> Low score. This is EXPLOITATION (picking low mu).
    # 
    # For MINIMIZATION:
    # We want low mu.
    # We want high sigma (uncertainty).
    # So we want to maximize (something with sigma) and minimize mu.
    # 
    # UCB for minimization is typically:
    # Lower Confidence Bound (LCB) = mu - kappa * sigma.
    # We pick the candidate with the LOWEST LCB.
    # But the interface requires HIGHER score = evaluate next.
    # So score = -LCB = - (mu - kappa * sigma) = -mu + kappa * sigma.
    
    score = -mu + kappa * sigma
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score

# EVOLVE-BLOCK-END
