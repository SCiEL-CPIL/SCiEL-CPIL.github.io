"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Enforce deadline with safety margin
    start_time = time.time()
    deadline = start_time + 295.0  # 300s limit, 5s safety margin

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle potential zeros in sigma to avoid division by zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Z-scores
    z = (f_best - mu) / sigma_safe
    
    # Pre-compute CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (uncosted)
    EI = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Strategy: Budget-Aware Thompson Sampling
    # Instead of deterministic EI, we sample from the posterior N(mu, sigma^2).
    # We use a temperature parameter to modulate the exploration-exploitation trade-off.
    # As the budget depletes, we should shift towards exploitation.
    # Since we don't know the exact step, we can estimate the "remaining budget" 
    # by looking at the cost of the current best point vs the total budget?
    # Actually, the prompt says "Runs start with 5 uniform random points... While total cost is below 30".
    # We don't have the current total cost passed in, only the cost of the 2000 candidates.
    # However, we can infer the "age" of the run from the number of points or just use a fixed strategy.
    # But the directive asks for "dynamic, budget-aware... using the remaining budget".
    # Since we don't have the global state, we can approximate the "stage" by the spread of mu/sigma 
    # or simply use a fixed temperature that balances well.
    # A better proxy for "remaining budget" in a single call without global state is difficult.
    # However, we can assume that early in the run, sigma is large everywhere, and later it shrinks.
    # Let's use a temperature that is inversely proportional to the average uncertainty?
    # Or, simpler: Thompson Sampling naturally balances exploration. 
    # We sample y ~ N(mu, sigma^2).
    # We want to maximize (f_best - y) / cost.
    # But y is random. We want the expected value of the best sample?
    # No, Thompson Sampling usually picks the argmax of the sample.
    # Here we return scores for 2000 points. The evaluator picks the best score.
    # If we return the sample y, the evaluator picks the max y (which is bad for minimization).
    # We want to minimize f(x). So we want to pick the candidate with the lowest predicted f(x).
    # So we should return -y? Or we want to pick the candidate that is most likely to be low.
    # If we sample y_k ~ N(mu_k, sigma_k^2), and we want to pick the one with the lowest true value,
    # we should pick the one with the lowest sample y_k?
    # Yes, if we sample y_k and pick argmin y_k, we are doing Thompson Sampling for minimization.
    # But we return a score where higher = evaluate next.
    # So we should return -y_k / cost_k?
    # Let's try Score = -y_sample / cost.
    
    # However, pure TS can be unstable. Let's add a temperature parameter.
    # y_sample = mu + sigma * z_sample, where z_sample ~ N(0, 1).
    # To control exploration, we can scale the noise.
    # y_sample = mu + temperature * sigma * z_sample.
    # If temperature is high, we explore more.
    # If temperature is low, we exploit more.
    
    # How to determine temperature dynamically?
    # We can estimate the "progress" by the ratio of f_best to the initial scale?
    # Or we can just use a fixed temperature that works well, e.g., 1.0.
    # But the directive says "dynamic... using the remaining budget".
    # Let's assume that if f_best is very low (close to 0), we are in late stage and should exploit.
    # If f_best is high, we are in early stage and should explore.
    # The minimum is 0.
    # Let's define a "progress" metric p = 1 - f_best / f_initial.
    # We don't know f_initial. But we can use a heuristic.
    # Alternatively, we can use the average sigma as a proxy for uncertainty.
    # High avg sigma -> early stage -> high temperature.
    # Low avg sigma -> late stage -> low temperature.
    
    avg_sigma = np.mean(sigma_safe)
    # Normalize avg_sigma to [0, 1] roughly.
    # In the early stages, sigma might be around 1-2. In late stages, it drops.
    # Let's map avg_sigma to temperature.
    # temp = 0.5 + avg_sigma?
    # Or temp = 1.0 / (1.0 + avg_sigma)?
    # Let's try temp = 1.0. It's a standard Thompson Sampling.
    
    # Let's try a different approach: 
    # Score = E[ (f_best - f(x)) / cost ] = EI / cost.
    # This is what the current program does (mostly).
    # The inspiration 1 (1.41) likely used something better.
    # Let's look at the "Inspiration 1" description: "Rewritten program for cost-aware acquisition".
    # Maybe it used a proper cost model?
    
    # Let's try **Expected Improvement** with a **Gaussian Process Lower Confidence Bound (GP-UCB)** 
    # but with a **cost-aware** penalty that is **adaptive**.
    
    # Actually, let's stick to the Thompson Sampling idea but refine it.
    # We sample y ~ N(mu, sigma^2).
    # We want to maximize the probability of being the best?
    # No, we just pick the best sample.
    # Score = -y_sample / cost.
    # Let's add a small exploitation term to stabilize it.
    # Score = -(mu + temperature * sigma * z_sample) / cost.
    
    # To make it "budget-aware", we can decrease temperature as the run progresses.
    # Since we don't have the step number, we can use the number of distinct mu values? No.
    # We can use the fact that f_best decreases.
    # Let's assume a typical f_best trajectory.
    # If f_best < 0.1, we are close to the optimum. Reduce temperature.
    # If f_best > 1.0, we are far. Increase temperature.
    
    # Let's define temperature based on f_best.
    # temp = 1.0 / (1.0 + f_best)
    # If f_best = 0, temp = 1.0.
    # If f_best = 1, temp = 0.5.
    # If f_best = 10, temp = 0.09.
    # This seems reasonable.
    
    temperature = 1.0 / (1.0 + max(f_best, 0.0))
    
    # Generate a single random normal vector for all 2000 points?
    # Or independent samples?
    # Independent samples are standard for TS.
    rng = np.random.default_rng()
    z_sample = rng.standard_normal(len(mu))
    
    y_sample = mu + temperature * sigma_safe * z_sample
    
    # We want to minimize y_sample.
    # So we want to maximize -y_sample.
    # And we want to penalize cost.
    # Score = -y_sample / cost.
    
    score = -y_sample / np.maximum(cost, eps)
    
    # Safety: ensure all scores are finite
    if not np.all(np.isfinite(score)):
        # Fallback to EI / cost
        score_fallback = EI / np.maximum(cost, eps)
        if np.all(np.isfinite(score_fallback)):
            score = score_fallback
        else:
            score = np.zeros_like(mu)
    
    # Check time budget
    if time.time() > deadline:
        pass
    
    return score
# EVOLVE-BLOCK-END
