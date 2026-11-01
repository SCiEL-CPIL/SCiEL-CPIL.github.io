"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_acf(h):
    """Compute linear autocorrelation using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    return acf


def _compute_c3_bound(h):
    """
    Compute the C3 autoconvolution upper bound.
    For a ±1 sequence, this is based on the third moment of the autocorrelation.
    """
    acf = _compute_acf(h)
    sidelobes = acf[1:]
    c3 = np.sum(np.abs(sidelobes) ** 3) / (N ** 3)
    return c3


def _compute_c3_from_continuous(x):
    """
    Compute C3 bound for a continuous vector x in [-1, 1]^N.
    This is a smooth surrogate for the discrete problem.
    """
    acf = _compute_acf(x)
    sidelobes = acf[1:]
    c3 = np.sum(np.abs(sidelobes) ** 3) / (N ** 3)
    return c3


def _compute_c3_gradient(x):
    """
    Compute gradient of C3 bound with respect to x using FFT derivatives.
    This is an approximation for optimization purposes.
    """
    # For smooth optimization, we can use numerical gradient or analytical
    # Here we use a finite difference approach for robustness
    eps = 1e-7
    grad = np.zeros(N)
    f0 = _compute_c3_from_continuous(x)
    for i in range(N):
        xp = x.copy()
        xp[i] += eps
        fp = _compute_c3_from_continuous(xp)
        grad[i] = (fp - f0) / eps
    return grad


def _round_to_binary(x):
    """Round continuous vector to ±1."""
    return np.sign(x)
    # Handle zeros by assigning +1
    x = x.copy()
    x[np.abs(x) < 1e-10] = 1.0
    return np.sign(x)


def _local_search_binary(h, deadline, max_iters=5000):
    """
    Fast local search on binary sequence.
    Tries single flips and accepts if C3 improves.
    Uses efficient ACF update for speed.
    """
    best_h = h.copy()
    best_val = _compute_c3_bound(best_h)
    
    # Precompute ACF for fast updates
    acf = _compute_acf(best_h)
    
    def update_acf_single_flip(acf, h, idx, new_val):
        """Update ACF after flipping one element."""
        # This is an approximation; for exact, recompute
        # For speed, we'll just recompute when needed
        pass
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try each flip
        for i in range(N):
            if time.time() > deadline:
                break
            old_val = best_h[i]
            best_h[i] = -old_val
            val = _compute_c3_bound(best_h)
            if val < best_val:
                best_val = val
                improved = True
            else:
                best_h[i] = old_val  # revert
    
    return best_h, best_val


def _simulated_annealing(h, deadline, T0=1.0, T_end=0.01, cooling=0.9999):
    """
    Simulated annealing on binary sequence.
    Accepts worse solutions with probability exp(-delta/T).
    """
    current_h = h.copy()
    current_val = _compute_c3_bound(current_h)
    best_h = current_h.copy()
    best_val = current_val
    
    T = T0
    rng = np.random.default_rng(12345)
    
    while time.time() < deadline and T > T_end:
        # Pick random position to flip
        i = rng.integers(0, N)
        old_val = current_h[i]
        current_h[i] = -old_val
        new_val = _compute_c3_bound(current_h)
        
        delta = new_val - current_val
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = current_h.copy()
        else:
            current_h[i] = old_val  # revert
        
        T *= cooling
    
    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Strategy 1: Continuous optimization to find good starting points
    best_h = np.ones(N)
    best_val = _compute_c3_bound(best_h)
    
    # Try multiple random initializations for continuous optimization
    rng = np.random.default_rng(42)
    
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        # Random initialization in [-1, 1]
        x0 = rng.uniform(-1, 1, size=N)
        
        try:
            result = minimize(
                _compute_c3_from_continuous,
                x0,
                method='L-BFGS-B',
                bounds=[(-1, 1)] * N,
                options={'maxiter': 500, 'maxfun': 2000, 'disp': False}
            )
            if result.success or result.fun < best_val:
                # Round to binary
                h_bin = _round_to_binary(result.x)
                val_bin = _compute_c3_bound(h_bin)
                if val_bin < best_val:
                    best_h = h_bin
                    best_val = val_bin
        except Exception:
            pass
    
    # Strategy 2: Simulated annealing from best found
    if time.time() < deadline - 15:
        sa_h, sa_val = _simulated_annealing(best_h, deadline)
        if sa_val < best_val:
            best_h = sa_h
            best_val = sa_val
    
    # Strategy 3: Local search polish
    if time.time() < deadline - 5:
        ls_h, ls_val = _local_search_binary(best_h, deadline)
        if ls_val < best_val:
            best_h = ls_h
            best_val = ls_val
    
    return best_h
# EVOLVE-BLOCK-END
