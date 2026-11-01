"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _max_sidelobe(h):
    """Compute max |autocorrelation| at non-zero lags using FFT."""
    n = len(h)
    n_fft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(h.astype(np.float64), n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:n]
    return np.max(np.abs(acf[1:]))


def _c3_bound(h):
    """
    Compute the C3 upper bound for a signed step function.
    The bound involves the autocorrelation sidelobes.
    For a sequence h of length N with values +/-1, the C3 bound
    is related to the maximum sidelobe of the autocorrelation.
    
    Based on the task description, we need to minimize the upper bound on C3.
    The score is the raw value (lower is better).
    
    Let's use the max sidelobe as the metric, as that's what the current
    program optimizes and what the inspirations seem to target.
    """
    return _max_sidelobe(h)


def _continuous_objective(x):
    """
    Objective function for continuous relaxation.
    x is a vector of length N in [-1, 1] (or unbounded, we'll clip).
    We compute the autocorrelation of the clipped/sign-processed signal.
    
    Actually, for continuous relaxation, we should compute the autocorrelation
    of the continuous signal and then minimize some norm of the sidelobes.
    """
    # Clip to reasonable range to avoid numerical issues
    h = np.tanh(x)  # Use tanh to keep values in (-1, 1)
    
    n = len(h)
    n_fft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:n]
    
    # Return sum of squares of sidelobes (smooth objective)
    sidelobes = acf[1:]
    return np.sum(sidelobes ** 2)


def _continuous_gradient(x):
    """
    Gradient of the continuous objective.
    This requires some care with the FFT-based autocorrelation.
    
    For now, let's use numerical gradient or let scipy compute it.
    """
    # Let scipy handle the gradient computation via finite differences
    # or we can derive it analytically. For simplicity, use finite differences.
    eps = 1e-7
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (_continuous_objective(x_plus) - _continuous_objective(x_minus)) / (2 * eps)
    return grad


def _local_search_discrete(h, deadline):
    """
    Discrete local search: try flipping each bit, keep if it improves.
    """
    n = len(h)
    best_val = _max_sidelobe(h)
    
    # Precompute FFT size
    n_fft = 1 << (2 * n - 1).bit_length()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_i = -1
        best_new_val = best_val
        
        for i in range(n):
            if time.time() > deadline:
                break
            # Try flipping bit i
            h[i] = -h[i]
            val = _max_sidelobe(h)
            h[i] = -h[i]  # revert
            
            if val < best_new_val:
                best_new_val = val
                best_i = i
        
        if best_i >= 0 and best_new_val < best_val:
            h[best_i] = -h[best_i]
            best_val = best_new_val
            improved = True
    
    return h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    np.random.seed(42)
    
    # Phase 1: Continuous relaxation using L-BFGS-B
    # Start from a random continuous vector
    x0 = np.random.randn(N)
    
    # We'll use multiple random restarts for the continuous phase
    best_h = None
    best_val = np.inf
    
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:  # Leave 20s for discrete refinement
            break
        
        x0 = np.random.randn(N)
        
        try:
            result = minimize(
                _continuous_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-4}
            )
        except Exception:
            continue
        
        # Convert continuous solution to discrete (+/-1)
        h_cont = np.tanh(result.x)
        h_disc = np.sign(h_cont)
        h_disc[h_disc == 0] = 1  # Handle any zeros
        
        val = _max_sidelobe(h_disc)
        
        if val < best_val:
            best_val = val
            best_h = h_disc.copy()
    
    if best_h is None:
        # Fallback: random discrete sequence
        best_h = np.random.choice([-1.0, 1.0], size=N).astype(np.float64)
        best_val = _max_sidelobe(best_h)
    
    # Phase 2: Discrete local search refinement
    if time.time() < deadline - 5:
        best_h, best_val = _local_search_discrete(best_h.copy(), deadline - 5)
    
    return best_h
# EVOLVE-BLOCK-END
