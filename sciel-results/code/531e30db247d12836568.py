"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track progress and adapt parameters
_start_time = time.time()
_eval_count = 0
_last_mu = None
_last_sigma = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using an adaptive Thompson Sampling-like strategy.
    
    Strategy:
    1. Normalize mu and sigma relative to the scale of the problem (f_best and sigma range).
    2. Create a "quality" score that balances exploitation (low mu) and exploration (high sigma).
    3. Apply a cost penalty that is less aggressive in early stages and more aggressive later.
    4. Add a small amount of stochastic noise to the scores to encourage diversity in selection
       (simulating Thompson Sampling's random draws, but deterministic per call based on index 
       to avoid breaking reproducibility if the same inputs are passed, though in a loop 
       the inputs change).
       
    Since we cannot truly sample from the posterior in a vectorized deterministic way without 
    storing state, we use a deterministic "expected improvement" variant but modulated by 
    a dynamic exploration rate that decays as the budget is consumed.
    """
    global _eval_count, _start_time, _last_mu, _last_sigma
    
    # Deadline guard
    if time.time() - _start_time > 325.0:
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Increment eval counter (approximate, as we don't know exact call count vs eval count)
    # We can infer progress from f_best stability or just use a fixed schedule based on cost.
    # Total cost limit is 30. Initial cost is ~5 * (1 + 0.5) = 7.5.
    # So we have ~22.5 cost units left.
    # We can estimate progress by looking at the average cost of evaluated points? 
    # We don't have that. Let's use a simple heuristic:
    # If f_best is very close to 0 (for functions with min 0), we are exploiting.
    # If sigma is high, we are exploring.
    
    # 1. Scale normalization
    # Use f_best as a scale reference if it's not zero, otherwise 1.0
    # For minimization, f_best starts high and goes down.
    # Let's normalize mu and sigma by the range of mu to make scores comparable across functions.
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = max(mu_max - mu_min, 1e-9)
    
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = max(sigma_max - sigma_min, 1e-9)
    
    # Normalize to [0, 1] where 0 is best (lowest mu) and 1 is worst (highest mu)
    mu_norm = (mu - mu_min) / mu_range
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # 2. Dynamic Exploration Parameter
    # We want to explore more early on and exploit later.
    # Estimate progress: 
    # If we assume the initial 5 points cost ~7.5, and total budget is 30, 
    # we have roughly (30 - current_cost) / (30 - 7.5) fraction left.
    # We don't know current_cost directly, but we can approximate it.
    # A simpler proxy: The number of times this function is called is roughly proportional to evaluations.
    # However, we can't easily count calls without side effects that might be reset.
    # Let's use the spread of sigma as a proxy for uncertainty. 
    # Early on, sigma is high everywhere. Later, sigma is low near f_best and high elsewhere.
    
    # Alternative: Use a fixed schedule based on the magnitude of f_best.
    # If f_best is large, we are early. If small, we are late.
    # But f_best scales differ (Ackley min 0, Hartmann min -3.86).
    
    # Let's stick to a robust relative metric:
    # beta_t: Exploration weight.
    # Start with high beta, decay it.
    # We can use the average sigma as a proxy for "how much we know".
    avg_sigma = np.mean(sigma)
    # If avg_sigma is high, we are early. If low, we are late.
    # Let's map avg_sigma to a beta in [0.5, 3.0].
    # We need a reference scale. Let's use the median of cost as a stable reference? No.
    # Let's use the initial sigma scale.
    
    # Simpler adaptive strategy:
    # Score = (f_best - mu + beta * sigma) / cost
    # But make beta depend on the "confidence" of the GP.
    # High confidence (low sigma) -> low beta.
    # Low confidence (high sigma) -> high beta.
    
    # Let's try a different angle: 
    # Use the "Probability of Improvement" (PI) but with a cost-aware twist.
    # PI = Phi((f_best - mu) / sigma)
    # Score = PI * (1 + alpha * sigma) / cost
    
    # To make it "Thompson Sampling" like, we can add a term that rewards
    # points where the lower bound of the credible interval is low.
    # Lower Bound = mu - k * sigma
    # We want to minimize Lower Bound. So Score = -Lower Bound / cost = (-mu + k*sigma) / cost.
    # This is UCB.
    
    # The key insight from the inspiration is to use a soft-max or a different aggregation.
    # But let's try a **Cost-Effective Expected Improvement (CEEI)** with a dynamic xi.
    
    # Standard EI:
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # CEEI = EI / cost
    
    # To improve on the current program (which uses max(EI/cost, UCB/cost, PI/cost)),
    # let's use a **weighted sum** where the weight shifts from exploration to exploitation
    # based on the "regret" so far.
    
    # Regret proxy: f_best - min_possible. We don't know min_possible.
    # But we can look at the improvement rate.
    
    # Let's implement a **Adaptive UCB** where beta is determined by the 
    # ratio of the best possible improvement to the current uncertainty.
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute EI
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB score (minimization version: -mu + beta*sigma)
    # We need to scale this to be comparable to EI.
    # EI is in objective units. UCB term (beta*sigma) is in objective units.
    # So we can just add them or take a max.
    
    # Let's use a "Soft-UCB" which is a smooth maximum.
    # Or, let's try **Knowledge Gradient** approximation? Too complex.
    
    # Let's try **PI with Cost Discounting** but with a dynamic threshold.
    # The current program uses max(EI/c, UCB/c, PI/c).
    # The problem with max is it can be dominated by a single noisy PI or EI spike.
    # A sum is more robust.
    
    # Strategy: Score = (EI + lambda * UCB_term) / cost
    # Where UCB_term = max(0, f_best - (mu - beta*sigma))  # Improvement over lower bound?
    # Actually, UCB for minimization is mu - beta*sigma.
    # If mu - beta*sigma < f_best, there is potential improvement.
    # Potential Improvement = f_best - (mu - beta*sigma) = (f_best - mu) + beta*sigma.
    
    # Let's define:
    # Exploit Score = EI / cost
    # Explore Score = ((f_best - mu) + beta * sigma) / cost  (This is always positive if beta is large enough)
    
    # Dynamic Beta:
    # If we are early (high average sigma), we want higher beta.
    # If we are late (low average sigma near optimum), we want lower beta.
    
    # Estimate "stage" by the ratio of the median sigma to the initial sigma.
    # We don't have initial sigma. 
    # Let's assume the sigma scale is roughly consistent.
    # Let's use the 90th percentile of sigma as a measure of "remaining uncertainty".
    sigma_90 = np.percentile(sigma, 90)
    sigma_10 = np.percentile(sigma, 10)
    
    # If sigma_90 is much larger than sigma_10, we have high uncertainty in parts of the domain.
    # Let's set beta based on this spread.
    # beta = 1.0 + 2.0 * (sigma_90 / (sigma_90 + sigma_10 + 1e-9))
    # This maps spread to beta in [1.0, 3.0].
    
    # Actually, let's look at the "regret" relative to the best possible.
    # Since we don't know the best possible, let's just use a fixed but slightly higher beta
    # than the current program's 2.0, but apply it in a sum rather than max.
    
    # The current program uses max. Summing allows both exploitation and exploration to contribute.
    # Score = (EI + 0.5 * (f_best - mu + 2.0 * sigma)) / cost
    
    # Let's refine the exploration term.
    # If mu is already below f_best, EI is high. The exploration term (f_best - mu + ...) is also high.
    # This double counts.
    # We should only add exploration if EI is low?
    # Or use a "gated" exploration:
    # Explore = (f_best - mu + beta * sigma) * (1 - Phi(z))  # Only explore if improvement is not certain?
    # No, that's not right.
    
    # Let's try a **Log-Barrier** approach as hinted in the inspirations.
    # Score = log(1 + EI/cost) + gamma * log(1 + sigma/cost)
    # This prevents one term from dominating.
    
    cost_safe = np.maximum(cost, 1e-9)
    
    term_ei = ei / cost_safe
    term_explore = sigma / cost_safe
    
    # Dynamic gamma:
    # If f_best is improving (we don't track history easily), 
    # let's just use a fixed gamma that balances the scales.
    # EI is usually smaller than sigma in early stages?
    # Let's normalize them.
    
    # Let's use the **Probability of Improvement** as the base, but multiply by a 
    # "confidence" factor that rewards high sigma only when mu is also reasonably low.
    
    # Score = PI * (1 + beta * sigma) / cost
    # PI = Phi(z)
    # This is the "Thompson Sampling" heuristic: 
    # You sample from N(mu, sigma^2). 
    # The expected value of the sample is mu. 
    # But we want to maximize the chance of finding a better point.
    # A common heuristic is to rank by -mu + beta*sigma.
    
    # Let's try a **Weighted Sum of EI and UCB** with weights that depend on the 
    # "distance" from the best point.
    
    # Distance = f_best - mu.
    # If distance is large (mu is much worse than f_best), we rely more on UCB (exploration).
    # If distance is small (mu is close to f_best), we rely more on EI (exploitation).
    
    # We can use a sigmoid to blend.
    # blend = sigmoid(k * (f_best - mu))
    # Score = (blend * EI + (1-blend) * UCB_term) / cost
    
    # Let's implement this.
    # k controls the sharpness.
    # If mu is far below f_best, blend -> 1 (Exploit).
    # If mu is far above f_best, blend -> 0 (Explore).
    
    # Scale for sigmoid:
    # Let's use the standard deviation of mu as the scale.
    mu_std = np.std(mu)
    if mu_std < 1e-9:
        mu_std = 1e-9
        
    # z_blend = (f_best - mu) / (2 * mu_std)
    # blend = sigmoid(z_blend)
    
    # Sigmoid implementation
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    
    z_blend = (f_best - mu) / (2.0 * mu_std)
    blend = sigmoid(z_blend)
    
    # EI term
    ei_term = ei / cost_safe
    
    # UCB term (Minimization: we want low mu - beta*sigma. Score = -(mu - beta*sigma) = -mu + beta*sigma)
    # To make it comparable to EI (which is 0 if no improvement), 
    # let's define UCB_improvement = max(0, f_best - (mu - beta*sigma))
    beta = 2.0
    ucb_improvement = np.maximum(0, f_best - mu + beta * sigma_safe)
    ucb_term = ucb_improvement / cost_safe
    
    # Blend
    scores = (blend * ei_term + (1.0 - blend) * ucb_term)
    
    # Add a small exploration bonus for high sigma regions that are not already covered by UCB
    # to ensure we don't ignore very uncertain areas if mu is also uncertain.
    # Actually, UCB already covers this.
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
