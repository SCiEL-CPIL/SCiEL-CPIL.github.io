"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Internal state for Thompson Sampling
# We need to maintain the count of evaluations to adjust exploration noise
# if we wanted, but standard TS uses the GP posterior directly.
# The signature is acquisition(mu, sigma, f_best).
# mu and sigma are the GP mean and std dev at 2000 candidate points.
# We sample from N(mu, sigma^2) and return the indices/values that would be "best"
# but the function must return scores for the 2000 points.
# In Thompson Sampling for maximization, we sample f(x) ~ N(mu(x), sigma(x)^2)
# and pick x with max f(x).
# Here we are minimizing. So we want to minimize the sampled function.
# The acquisition function returns scores where HIGHER means evaluate next.
# So we should return -sampled_value, or a score proportional to how "good" (low) the sample is.
# However, standard TS implementation in an acquisition function context:
# The evaluator picks argmax(score).
# If we sample y ~ N(mu, sigma), and we want to minimize y,
# we want to pick the x with the smallest y.
# So score(x) = -y_sample(x).

# To ensure robustness and handle the "finite scores" requirement:
# We generate a random sample for each of the 2000 points.
# This introduces stochasticity.

# One potential issue: pure TS can be too greedy or too exploratory depending on GP fit.
# But the prompt says "different strategy... Thompson Sampling".
# Let's implement pure TS.

# We need to be careful about the random seed. The evaluator handles the loop,
# but our function is called repeatedly. If we use np.random without seeding,
# it uses the global state. This is usually fine.

# Deadline guard:
# The task says "Each evaluation must finish within 300 s".
# "evaluation" here likely refers to the whole BO loop for one seed, or a single call?
# "until 30 evaluations... For each seed... 300s wall-clock" usually implies the whole run.
# However, the acquisition function itself is called 25 times (30-5 initial).
# 2000 points is small. Thompson sampling is O(N) for sampling.
# It should be very fast.

# Let's add a simple time check just in case, though it's likely not a bottleneck.
_start_time = time.time()
_TIMEOUT = 290.0  # Safety margin for 300s limit

def acquisition(mu, sigma, f_best):
    # Check time
    if time.time() - _start_time > _TIMEOUT:
        # Fallback to something fast and safe, e.g., EI or just return -mu
        # If we time out, we should return valid finite scores.
        # Returning -mu is a safe deterministic fallback (greedy exploitation).
        return -mu

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Sample from the posterior predictive distribution
    # y ~ N(mu, sigma^2)
    # Since we are minimizing, lower y is better.
    # The acquisition function returns scores, higher is better.
    # So score = -y_sample
    z = np.random.normal(0, 1, size=mu.shape)
    y_sample = mu + sigma * z
    
    # Return negative sampled value as score
    # This ensures that the point with the lowest predicted value (highest score) is selected
    return -y_sample
# EVOLVE-BLOCK-END
