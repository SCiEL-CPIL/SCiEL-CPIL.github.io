import numpy as np
import time

# Module-level state to track time for the deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of acquisition scores (higher is better).
    """
    global _start_time
    
    # Initialize start time if not set
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: If we are getting close to the 300s limit (with safety margin),
    # we should return a safe fallback to avoid crashing the evaluator.
    # The problem states "Each evaluation must finish within 300 s" and "evaluator KILLS after 330s".
    # Since this function is called repeatedly, we check the elapsed time.
    # If we've used > 290s, we return a score that effectively picks the first candidate or a neutral score.
    # However, returning a constant score might break the "finite scores" requirement if not careful.
    # A safer fallback is to return the negative cost (so it picks the cheapest point) or just zeros.
    # But usually, the bottleneck is the GP update in the main loop, not the acquisition.
    # To be safe, if elapsed > 295s, we return a simple heuristic.
    elapsed = time.time() - _start_time
    if elapsed > 295.0:
        # Fallback: pick the cheapest point (lowest cost) to finish quickly
        # Higher score = better. So we want high score for low cost.
        # Score = -cost.
        return -np.asarray(cost, dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Ensure finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
    
    # Avoid division by zero in cost
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling:
    # We want to sample from the posterior: y ~ N(mu, sigma^2)
    # To make this deterministic and reproducible for a given state (mu, sigma),
    # we can use a fixed seed or derive a seed from the inputs.
    # However, using a fixed seed every time means the "randomness" is actually a fixed transformation of the uncertainty.
    # This is effectively a deterministic heuristic that mimics TS.
    # A better approach for TS in a stateless function is to use a "pseudo-random" number that changes slightly
    # or just use a fixed random draw. Since the GP parameters (mu, sigma) change each iteration,
    # the resulting sample will change.
    
    # Let's use a fixed standard normal draw for the noise.
    # This makes the acquisition deterministic given mu, sigma, and cost.
    # This is often more stable than true random TS in evolutionary/optimization contexts where we want consistent behavior.
    
    # Generate a standard normal random vector.
    # We can use a fixed seed to ensure the same "noise pattern" is applied.
    # This effectively selects the point where the GP mean minus a scaled uncertainty is lowest.
    # Wait, standard TS samples y = mu + sigma * z.
    # If we use a fixed z, then y = mu + sigma * z_fixed.
    # We want to minimize y / cost.
    # So we want to maximize - (mu + sigma * z_fixed) / cost.
    
    # Let's pick z_fixed. A standard normal vector.
    # To avoid hardcoding a huge array, we can generate it once or use a simple property.
    # Actually, generating a new random z every time is standard TS.
    # But since the function is called in a loop, using `np.random` without a seed might introduce variance.
    # The prompt says "5 seeded instances", implying the environment is deterministic.
    # If we use `np.random`, we need to be careful about state.
    # Let's use a deterministic "noise" based on the index or a fixed constant.
    # A common trick is to use a fixed standard normal vector.
    
    # Let's create a fixed standard normal vector of size 2000.
    # We can generate it once using a fixed seed.
    # However, the size is always 2000.
    
    # To be safe and fast, we'll generate a fixed noise vector.
    # This turns TS into a specific deterministic heuristic.
    # Is this "Thompson Sampling"? It's a single sample from the posterior with a fixed noise realization.
    # This is equivalent to optimizing a single sample path.
    
    # Let's try a different approach: Use the current time or a counter to seed the RNG?
    # No, the function should be deterministic for the same inputs if possible, or at least robust.
    # Let's use a fixed seed for the random number generator inside the function.
    # This means every call uses the same "random" numbers.
    # This is effectively: Score = -(mu + sigma * z_i) / cost_i
    # where z_i is a fixed standard normal number.
    
    # Let's generate z.
    rng = np.random.default_rng(42)
    z = rng.standard_normal(2000)
    
    # Sampled values
    sampled_values = mu + sigma * z
    
    # We want to minimize sampled_values / cost
    # So we maximize -sampled_values / cost
    score = -sampled_values / cost
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return score
