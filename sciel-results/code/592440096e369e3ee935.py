import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Posterior Sampling-based acquisition function approximating Expected Improvement.
    
    Instead of a deterministic UCB, we draw samples from the GP posterior 
    (approximated here by sampling from N(mu, sigma^2)) and compute the 
    average of the maximum improvements across these samples.
    
    Parameters:
    mu: array of predicted means (shape N,)
    sigma: array of predicted standard deviations (shape N,)
    f_best: best observed function value so far
    
    Returns:
    scores: array of scores (shape N,), higher is better
    """
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s limit
    
    N = len(mu)
    if N == 0:
        return np.array([])
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Number of posterior samples to draw
    # 50 samples provides a reasonable approximation of the expected max improvement
    # without being too computationally expensive for N=2000
    n_samples = 50
    
    # We want to estimate E[ max_x (f_best - f(x)) ] where f(x) ~ N(mu(x), sigma(x)^2)
    # Note: We are minimizing, so improvement is f_best - f(x).
    # We maximize this improvement.
    
    # Optimization: Instead of storing all samples (50x2000), we can accumulate
    # the max improvement for each sample.
    # However, the "score" for a candidate x is not just the max over samples,
    # but the expectation of the improvement at x? 
    # No, the task asks for scores for each of the 2000 candidates.
    # Standard EI is E[max(0, f_best - f(x))] for a specific x.
    # The prompt says: "approximates Expected Improvement by drawing 50 latent function samples... and taking the maximum improvement over f_best across these samples".
    # This phrasing is slightly ambiguous. 
    # Interpretation A: Score(x) = (1/S) * sum_s max(0, f_best - f_s(x)). This is standard EI approximation.
    # Interpretation B: Score(x) is derived from the global max? No, we need a score for each candidate to pick the best one.
    # So we compute the approximate EI for each candidate x.
    
    # Let's compute EI approximation for each x.
    # f_s(x) ~ N(mu[x], sigma[x]^2)
    # We generate S samples for each x? That would be S*N random numbers.
    # S=50, N=2000 -> 100,000 random numbers. This is very fast.
    
    # Generate random samples: shape (S, N)
    # We use standard normal and scale by sigma
    # To save memory and time, we can do this in chunks or just vectorize.
    # 50 * 2000 is small.
    
    # Check time
    if time.time() > deadline:
        # Fallback to simple UCB if somehow we are slow (should not happen)
        beta = 2.5
        return -mu + beta * sigma

    # Draw samples
    # Z ~ N(0, 1)
    Z = np.random.randn(n_samples, N)
    
    # f_samples[i, j] = mu[j] + sigma[j] * Z[i, j]
    # Broadcasting: mu and sigma are (N,), Z is (S, N)
    f_samples = mu[None, :] + sigma[None, :] * Z
    
    # Compute improvement for each sample and candidate
    # Improvement = max(0, f_best - f_samples)
    # Shape: (S, N)
    improvements = np.maximum(0.0, f_best - f_samples)
    
    # Average over samples to get approximate EI for each candidate
    # Shape: (N,)
    scores = np.mean(improvements, axis=0)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
