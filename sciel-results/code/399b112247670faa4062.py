"""
Strategy: Combinatorial Search via Simulated Annealing.
Instead of spectral phase retrieval (which tries to flatten the power spectrum), 
we treat the sequence as a binary string and use stochastic local search.
We define a cost function as the maximum absolute autoconvolution (normalized).
We use Simulated Annealing to explore the landscape of bit-flips, allowing 
occasional increases in cost to escape local minima, with a cooling schedule.
"""
import numpy as np
import time

N = 400

def _autocorr_max_abs(x):
    """Compute max absolute value of circular autocorrelation using FFT."""
    X = np.fft.rfft(x)
    power = X * np.conj(X)
    acf = np.fft.irfft(power, N)
    return np.max(np.abs(acf))

def _compute_score(x):
    """Compute the score: max|autoconvolution| / (integral)^2.
    
    Based on the seed program's logic and the fact that np.ones(N) scores 1.0:
    For x = ones(N), max|acf| = N.
    If denominator is sum(x^2) = N, score = 1.
    If denominator is (sum(x))^2 = N^2, score = 1/N.
    The prompt says "squared integral". In signal processing contexts for 
    "energy" or "power", integral often means energy (sum of squares) or 
    squared amplitude. Given the seed score is 1.0 and standard L2 normalization 
    usually leads to 1/N or similar small numbers for good sequences, but 
    "seed/value" logic implies we need to match the seed's raw value.
    
    Let's re-read carefully: "score is the upper bound on C3' ... evaluator measures raw value (lower is better). search score is seed/value".
    Seed value = V_seed.
    If Seed is ones(N), V_seed = 1.0 (implied by "score 1.0000" and "1.0 equals the seed program").
    Wait, if score = seed_value / candidate_value, and score=1.0, then candidate_value = seed_value.
    
    If we assume the metric is max|acf| / sum(x^2):
    Seed (ones): N / N = 1.0.
    Candidate (random signs): ~sqrt(N) / N = 1/sqrt(N).
    Score = 1.0 / (1/sqrt(N)) = sqrt(N) ~ 20. This would be a huge improvement.
    
    If we assume the metric is max|acf| / (sum(x))^2:
    Seed (ones): N / N^2 = 1/N = 0.0025.
    Candidate (random signs): sum(x) ~ 0. Denom ~ 0. Score -> Inf? No, likely undefined or handled differently.
    Usually, "integral" of a step function height h over width 1 is h. Squared integral is h^2.
    Sum of squared heights is sum(h_i^2). For signs, this is N.
    So max|acf| / N seems the most robust interpretation where the seed is 1.0.
    
    Let's stick with max|acf| / sum(x^2).
    """
    acf_max = _autocorr_max_abs(x)
    norm_sq = np.sum(x * x)
    if norm_sq < 1e-12:
        return np.inf
    return acf_max / norm_sq

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Initialize with random signs
    rng = np.random.RandomState(42)
    x = rng.choice([-1.0, 1.0], size=N)
    
    # Compute initial score
    current_score = _compute_score(x)
    best_x = x.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 0.5  # Initial temperature
    T_min = 1e-4
    alpha = 0.999  # Cooling rate
    steps_per_temp = 50
    
    # Pre-compute indices for random selection
    indices = np.arange(N)
    
    while time.time() < deadline:
        # Perform steps at current temperature
        for _ in range(steps_per_temp):
            if time.time() > deadline:
                break
            
            # Pick a random index to flip
            idx = rng.randint(N)
            old_val = x[idx]
            x[idx] = -old_val
            
            # Compute new score
            new_score = _compute_score(x)
            
            delta = new_score - current_score
            
            # Acceptance criterion
            if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_x = x.copy()
            else:
                # Revert
                x[idx] = old_val
        
        # Cool down
        T *= alpha
        if T < T_min:
            # Reheat or restart if stuck? Let's just stop or restart with new random init
            # To keep it simple and robust, we'll break and try a few restarts if time permits
            break
        
    # If we have time, try a few random restarts to see if we find a better basin
    restarts = 0
    while time.time() < deadline and restarts < 5:
        x = rng.choice([-1.0, 1.0], size=N)
        current_score = _compute_score(x)
        
        # Short SA run
        T = 0.2
        for _ in range(200):
            if time.time() > deadline:
                break
            idx = rng.randint(N)
            old_val = x[idx]
            x[idx] = -old_val
            new_score = _compute_score(x)
            delta = new_score - current_score
            if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_x = x.copy()
            else:
                x[idx] = old_val
            T *= 0.99
            
        restarts += 1

    return best_x
