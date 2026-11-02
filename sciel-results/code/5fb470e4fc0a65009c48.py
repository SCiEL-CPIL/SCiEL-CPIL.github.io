import numpy as np
import time

N = 400
DEADLINE = 145.0  # safety margin before 150s kill


def _compute_c3_bound(signs):
    """
    Compute an upper bound on C3 for a signed step function.
    Uses L1 norm of autoconvolution as the metric.
    """
    dx = 0.5 / N
    n_fft = 2 * N
    F = np.fft.rfft(signs.astype(float), n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft) * dx
    return np.sum(np.abs(conv)) * dx


def _compute_c3_bound_fast(signs, dx, n_fft):
    """
    Faster version using precomputed FFT size.
    """
    F = np.fft.rfft(signs.astype(float), n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft) * dx
    return np.sum(np.abs(conv)) * dx


def _greedy_flip_optimize(signs, dx, n_fft, max_rounds=50, deadline=140.0):
    """
    Greedy coordinate descent: flip individual signs if it reduces the C3 bound.
    Multiple passes until no improvement or max rounds reached.
    """
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound_fast(current, dx, n_fft)

    for round_num in range(max_rounds):
        improved = False
        # Try flipping each position
        for i in range(N):
            if time.time() > deadline:
                return current, current_score
            old_val = current[i]
            current[i] = -old_val
            new_score = _compute_c3_bound_fast(current, dx, n_fft)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current[i] = old_val
        if not improved:
            break

    return current, current_score


def _random_restarts_optimize(initial_signs, dx, n_fft, num_restarts=10, deadline=140.0):
    """
    Multiple random restarts with greedy optimization.
    """
    best = initial_signs.astype(np.float64).copy()
    best_score = _compute_c3_bound_fast(best, dx, n_fft)
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create a perturbed version: flip a random subset of signs
        flip_mask = np.random.random(N) < 0.3  # flip ~30% of signs
        perturbed = initial_signs.astype(np.float64).copy()
        perturbed[flip_mask] *= -1
        
        optimized, score = _greedy_flip_optimize(perturbed, dx, n_fft, max_rounds=20, deadline=deadline)
        if score < best_score:
            best_score = score
            best = optimized
    
    return best, best_score


def construct():
    start_time = time.time()
    
    # Precompute constants
    dx = 0.5 / N
    n_fft = 2 * N
    
    rng = np.random.RandomState(42)
    
    # Start with a structured pattern that should give good initial C3
    # Use a smooth transition pattern: sine wave at low frequency
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    initial_pattern = np.sign(np.sin(2 * np.pi * 2 * x))
    initial_pattern[initial_pattern == 0] = 1
    
    # Run greedy optimization on the initial pattern
    optimized, opt_score = _greedy_flip_optimize(
        initial_pattern, dx, n_fft, 
        max_rounds=30, 
        deadline=DEADLINE - 30
    )
    
    # Try random restarts if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 20:
        best, best_score = _random_restarts_optimize(
            optimized, dx, n_fft,
            num_restarts=5,
            deadline=DEADLINE - 10
        )
        if best_score < opt_score:
            optimized = best
            opt_score = best_score
    
    # Ensure we return integer signs
    return np.sign(optimized).astype(int)
