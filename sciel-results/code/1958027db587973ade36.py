import numpy as np
import time

N = 400
DEADLINE = 140.0  # safety margin before 150s kill

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

def _sine_sign_pattern(freq, phase, N=N):
    """
    Generate a signed step function from zero crossings of a sine wave.
    sign(sin(2*pi*freq*x + phase)) where x in [-1/4, 1/4)
    """
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    pattern = np.sin(2 * np.pi * freq * x + phase)
    # Handle zeros: assign +1
    pattern[pattern == 0] = 1.0
    return np.sign(pattern)

def _greedy_flip_optimize(signs, max_rounds=50, time_limit=None, start_time=None):
    """
    Greedy coordinate descent: flip individual signs if it reduces the C3 bound.
    Multiple passes until no improvement or max rounds reached.
    """
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    
    for rnd in range(max_rounds):
        improved = False
        # Try flipping each position
        for i in range(N):
            old_val = current[i]
            current[i] = -old_val
            new_score = _compute_c3_bound(current)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current[i] = old_val
        if not improved:
            break
        # Time check
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
    
    return current, current_score

def _spectral_search(start_time):
    """
    Search over sine frequency and phase parameters to find good initial patterns.
    Low frequency = fewer sign transitions = less high-frequency energy.
    """
    best_pattern = None
    best_score = np.inf
    
    # Search over a range of frequencies and phases
    # Frequencies: number of half-periods in the interval
    # Low frequencies (1-10) should give smooth patterns
    freqs = np.arange(1, 21)  # 1 to 20 half-periods
    phases = np.linspace(0, 2*np.pi, 36, endpoint=False)
    
    for f in freqs:
        for ph in phases:
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 40:
                return best_pattern, best_score
            
            pattern = _sine_sign_pattern(f, ph)
            score = _compute_c3_bound(pattern)
            
            if score < best_score:
                best_score = score
                best_pattern = pattern.copy()
    
    return best_pattern, best_score

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Phase 1: Spectral search over sine parameters
    best_pattern, best_score = _spectral_search(start_time)
    
    # If spectral search didn't find anything, fall back to simple patterns
    if best_pattern is None:
        # All ones
        best_pattern = np.ones(N)
        best_score = _compute_c3_bound(best_pattern)
    
    # Phase 2: Refine with greedy coordinate descent
    elapsed = time.time() - start_time
    remaining = DEADLINE - 30 - elapsed
    
    if remaining > 10:
        # Use more rounds if we have more time
        max_rounds = min(50, max(5, int(remaining / 2)))
        optimized, opt_score = _greedy_flip_optimize(
            best_pattern, 
            max_rounds=max_rounds,
            time_limit=remaining,
            start_time=start_time
        )
        if opt_score < best_score:
            best_score = opt_score
            best_pattern = optimized
    
    # Phase 3: Try a few random perturbations of the best pattern if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 20:
        for trial in range(3):
            # Randomly flip a small fraction of signs
            n_flips = N // 10
            flip_indices = rng.choice(N, size=n_flips, replace=False)
            perturbed = best_pattern.copy()
            perturbed[flip_indices] *= -1
            
            pert_score = _compute_c3_bound(perturbed)
            if pert_score < best_score:
                best_score = pert_score
                best_pattern = perturbed.copy()
            
            # Quick refinement
            elapsed = time.time() - start_time
            if elapsed < DEADLINE - 15:
                optimized, opt_score = _greedy_flip_optimize(
                    best_pattern,
                    max_rounds=10,
                    time_limit=DEADLINE - 15 - elapsed,
                    start_time=start_time
                )
                if opt_score < best_score:
                    best_score = opt_score
                    best_pattern = optimized
    
    # Ensure we return integer signs
    return np.sign(best_pattern).astype(int)
