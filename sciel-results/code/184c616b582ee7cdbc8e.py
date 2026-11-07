import numpy as np
import time
from scipy.stats import norm

# Global state for time management and caching
_start_time = None
_last_mu = None
_last_sigma = None
_last_f_best = None
_last_cost = None
_cached_scores = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    global _last_mu, _last_sigma, _last_f_best, _last_cost, _cached_scores
    
    # Time guard: if we are close to the limit, return cached or simple fallback
    # The task says each evaluation must finish within 300s, but the loop runs while cost < 30.
    # The total wall clock for the whole run is 330s.
    # We assume acquisition is called many times. We should be fast.
    # If this is the first call, record start time.
    if _start_time is None:
        _start_time = time.time()
        
    elapsed = time.time() - _start_time
    # Safety margin: if we've been running for > 325s, return a safe, fast result
    # This prevents timeout.
    if elapsed > 325.0:
        # Fallback: simple EI-like score, very fast
        mu_arr = np.asarray(mu)
        sigma_arr = np.asarray(sigma)
        cost_arr = np.asarray(cost)
        sigma_safe = np.maximum(sigma_arr, 1e-9)
        z = (f_best - mu_arr) / sigma_safe
        ei = (f_best - mu_arr) * norm.cdf(z) + sigma_arr * norm.pdf(z)
        scores = ei / np.maximum(cost_arr, 1e-9)
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Check for caching (if inputs haven't changed, though unlikely in BO loop)
    # But in BO, inputs change every step. So caching is less useful unless the loop
    # calls acquisition multiple times with the same GP state? No, it scores 2000 candidates.
    # So mu, sigma, cost are fixed for one acquisition call.
    # We don't cache across calls because mu/sigma change.

    # 1. Compute EI relative to f_best (standard EI)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei_standard = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Approximate the posterior distribution of the global minimum
    # The GP posterior at the best observed point (x_best) has mean mu_best and std sigma_best.
    # However, we don't have x_best indices here, only arrays.
    # We can approximate the "uncertainty of the minimum" by looking at the best point in the current candidate set?
    # No, f_best is the global best observed so far.
    # The minimum of the GP sample is roughly mu_min - k * sigma_min.
    # A common approximation for the posterior of the global minimum $m$ is that it is a random variable.
    # Let's approximate the distribution of the true minimum $f_{min}$ as a Normal distribution.
    # Mean of $f_{min}$: slightly lower than f_best.
    # Std of $f_{min}$: related to the uncertainty at the best point.
    
    # Find the index of the candidate with the lowest mu (proxy for x_best location in the current cloud? No, x_best is from history).
    # Actually, we don't know where x_best is in the 2000 candidates.
    # But we can use the statistics of the current GP model.
    # Let's assume the uncertainty of the global minimum is proportional to the minimum sigma in the current candidate set?
    # Or better, use the sigma at the point where mu is minimized?
    
    # Let's find the candidate with the lowest mean.
    idx_min_mu = np.argmin(mu)
    mu_best_proxy = mu[idx_min_mu]
    sigma_best_proxy = sigma[idx_min_mu]
    
    # The true minimum is likely lower than f_best.
    # Let's model the "shift" delta.
    # If we sample a value $f_{sampled\_min}$ from a distribution centered at $f_{best}$ with some uncertainty,
    # then the improvement of a candidate $x$ is $f_{sampled\_min} - f(x)$.
    # We want to maximize the probability that $f(x) < f_{sampled\_min}$ or the expected improvement relative to it.
    
    # A simpler Thompson Sampling approach for minimization:
    # Sample a single function value $y^*$ from the posterior at the "best" location.
    # Since we don't have the exact location, we approximate the posterior of the global minimum.
    # Let $m \sim \mathcal{N}(\mu_{min\_est}, \sigma_{min\_est})$.
    # $\mu_{min\_est}$ can be approximated as $f_{best} - \text{bias}$.
    # $\sigma_{min\_est}$ can be approximated as $\sigma_{best\_proxy}$.
    
    # Let's sample one global shift $\delta$ from $\mathcal{N}(0, \sigma_{best\_proxy}^2)$.
    # This represents the uncertainty in where the true minimum lies relative to our current best estimate.
    # Actually, a more robust TS approach:
    # Sample a single value $v$ from $\mathcal{N}(f_{best}, \sigma_{best\_proxy}^2)$.
    # Then for each candidate $i$, the "improvement" is $v - (mu_i + \epsilon_i)$? No, that's not quite right.
    
    # Standard TS: Sample a full function $f_s(x) = mu(x) + sigma(x) * z(x)$ where $z(x)$ is a GP sample.
    # We can't sample a full GP easily in 2000 points without Cholesky (too slow? 2000x2000 Cholesky is ~26M ops, doable in C, but Python overhead?).
    # However, the directive says "sampling a single global shift". This implies a simplified model.
    
    # Simplified Model:
    # Assume the function value at any point $x$ is $f(x) = \mu(x) + \delta + \epsilon(x)$?
    # If we assume the uncertainty is dominated by a global shift $\delta$ (which is a strong assumption for a GP, but let's follow the directive),
    # then $f(x) \approx \mu(x) + \delta$.
    # We sample $\delta \sim \mathcal{N}(0, \sigma_{global})$.
    # What is $\sigma_{global}$? We can use the mean sigma or the sigma at the best point.
    # Let's use $\sigma_{global} = \sigma_{best\_proxy}$.
    
    # Sample delta
    delta = np.random.normal(0, sigma_best_proxy)
    
    # The estimated function value for candidate i is:
    # f_est_i = mu_i + delta
    # We want to minimize f_est_i.
    # So the score should be $-f_{est\_i}$ or $f_{best} - f_{est\_i}$.
    # Score_i = f_best - (mu_i + delta) = (f_best - delta) - mu_i.
    
    # This is equivalent to EI with a shifted baseline.
    # If delta > 0, we are pessimistic (think the min is higher), so we exploit more (lower mu).
    # If delta < 0, we are optimistic (think the min is lower), so we explore more (we need very low mu to beat it? No, if min is lower, we need to find it).
    
    # Wait, if we think the true minimum is $f_{best} + \delta$, and we want to find a point lower than that,
    # the improvement is $(f_{best} + \delta) - f(x)$.
    # Expected improvement relative to this sampled minimum:
    # $E[\max(0, (f_{best} + \delta) - f(x))]$.
    # Since $f(x) \sim \mathcal{N}(\mu_x, \sigma_x^2)$,
    # Let $C = f_{best} + \delta$.
    # $EI_x = (C - \mu_x) \Phi(\frac{C - \mu_x}{\sigma_x}) + \sigma_x \phi(\frac{C - \mu_x}{\sigma_x})$.
    
    # This is computationally more expensive than just $-\mu_x$, but captures the uncertainty.
    # However, the directive says "scoring candidates by their deterministic expected improvement relative to this sampled minimum".
    # "Deterministic expected improvement" might mean we treat the sampled minimum as a fixed constant and calculate EI relative to it.
    
    C = f_best + delta
    
    # Calculate EI relative to C
    z_c = (C - mu) / sigma_safe
    ei_c = (C - mu) * norm.cdf(z_c) + sigma * norm.pdf(z_c)
    
    # Cost-awareness
    # The original program divided by cost.
    # Dividing by cost can be unstable if cost is small.
    # Cost is $1 + \text{mean}(x_{scaled})$. Range [1, 2].
    # So dividing by cost is safe and bounded.
    # Let's keep the cost division to maintain the cost-awareness aspect of the task.
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei_c / cost_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
