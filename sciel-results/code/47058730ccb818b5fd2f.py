"""Erdos minimum overlap: Multi-harmonic spectral design + local polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
_start_time = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _deadline_ok(margin=2.0):
    return _elapsed() < _TIME_LIMIT - margin


def _h_from_coeffs(coeffs):
    """
    Build h from coefficients [A1, phi1, A2, phi2, ...].
    h_i = 0.5 + sum_k A_k * cos(2*pi*k*i/N + phi_k)
    """
    i = np.arange(N)
    h = np.full(N, 0.5)
    num_terms = len(coeffs) // 2
    for k in range(num_terms):
        A = coeffs[2 * k]
        phi = coeffs[2 * k + 1]
        h += A * np.cos(2.0 * np.pi * (k + 1) * i / N + phi)
    return h


def _overlap_smooth_max(h, alpha=50.0):
    """
    Compute smooth log-sum-exp of the overlap profile.
    Overlap at shift k: O_k = sum_i h_i (1 - h_{i-k}) = N/2 - autocorr_k.
    """
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = N / 2.0 - autocorr
    
    mmax = np.max(overlaps)
    # Avoid overflow
    exp_vals = np.exp(alpha * (overlaps - mmax))
    sum_exp = np.sum(exp_vals)
    val = (mmax + np.log(sum_exp)) / alpha
    return val


def _objective(coeffs):
    h = _h_from_coeffs(coeffs)
    # Soft constraint: penalize if h goes out of [0,1]
    penalty = 0.0
    if np.any(h < 0) or np.any(h > 1):
        penalty = 100.0 * (np.sum(np.clip(-h, 0, None)) + np.sum(np.clip(h - 1, 0, None)))
    return _overlap_smooth_max(h) + penalty


def _max_overlap(h):
    """Exact max circular overlap (raw value, lower is better)."""
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    return float(N / 2.0 - np.min(autocorr))


def _fix_sum(h):
    """Adjust h to sum exactly N/2, keeping values in [0,1]."""
    target = N / 2.0
    for _ in range(50):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h


def _local_search(h, max_sweeps=500):
    """Coordinate descent / local search to refine the full vector."""
    h = _fix_sum(h.copy())
    best = _max_overlap(h)
    deltas = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
    
    # Precompute FFT of h for fast overlap calculation? 
    # Actually, _max_overlap is O(N log N). 
    # For N=100, it's very fast.
    
    for _ in range(max_sweeps):
        if not _deadline_ok(10.0):
            break
        improved = False
        # Randomize order to avoid cycles
        indices = np.random.permutation(N)
        for i in indices:
            if not _deadline_ok(10.0):
                break
            # Try moving mass from i to j
            # To keep sum constant, we need to pick j.
            # Simple strategy: try all j? O(N^2) per sweep. 100*100 = 10k evals per sweep.
            # 500 sweeps * 10k = 5M evals. Too slow if each eval is 1ms.
            # Let's restrict j or use a smarter heuristic.
            
            # Heuristic: move mass to the index j where h[j] is lowest (if h[i] > 0.5) 
            # or highest (if h[i] < 0.5) to try to flatten.
            # Or just try a few random j's.
            
            if h[i] <= 0.001 or h[i] >= 0.999:
                continue
                
            # Try moving to a few random partners
            js = np.random.choice(N, 10, replace=False)
            for j in js:
                if i == j:
                    continue
                for d in deltas:
                    # Move d from i to j
                    if h[i] - d >= 0 and h[j] + d <= 1:
                        h2 = h.copy()
                        h2[i] -= d
                        h2[j] += d
                        h2 = _fix_sum(h2)
                        ov = _max_overlap(h2)
                        if ov < best - 1e-11:
                            h = h2
                            best = ov
                            improved = True
                            break
                    # Move d from j to i
                    if h[j] - d >= 0 and h[i] + d <= 1:
                        h2 = h.copy()
                        h2[j] -= d
                        h2[i] += d
                        h2 = _fix_sum(h2)
                        ov = _max_overlap(h2)
                        if ov < best - 1e-11:
                            h = h2
                            best = ov
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Try smaller deltas if no improvement
            break
    return h, best


def construct():
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Strategy: Optimize coefficients for a sum of cosines.
    # We'll try different numbers of harmonics.
    
    for num_harmonics in [1, 2, 3, 4]:
        if not _deadline_ok(15.0):
            break
            
        # Initial guess: small amplitudes
        x0 = np.zeros(2 * num_harmonics)
        # Add some randomization to escape local minima
        for restart in range(5):
            if not _deadline_ok(10.0):
                break
            
            x0 = np.random.uniform(-0.1, 0.1, size=2 * num_harmonics)
            # Ensure amplitudes are non-negative? No, phase handles sign.
            # But A can be negative. Let's just let optimizer handle it.
            
            try:
                res = minimize(
                    _objective, x0, method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-8}
                )
                h = _h_from_coeffs(res.x)
                h = _fix_sum(h)
                sc = _max_overlap(h)
                if sc < best_score - 1e-12:
                    best_score = sc
                    best_h = h.copy()
            except Exception:
                pass

    # Local polish on the best full vector
    if _deadline_ok(15.0):
        h, sc = _local_search(best_h, max_sweeps=1000)
        if sc < best_score - 1e-12:
            best_score = sc
            best_h = h.copy()
            
    best_h = _fix_sum(best_h)
    return best_h
