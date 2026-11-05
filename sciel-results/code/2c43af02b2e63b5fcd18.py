"""Find 24 ±1 coefficients minimizing sup-norm of P(z) on |z|=1, scaled by 1/sqrt(24)."""
import numpy as np
import time
import random

# Use a fixed seed for reproducibility but allow some randomness
random.seed(42)
np.random.seed(42)

N = 24
SQRT_N = np.sqrt(N)

def _eval_at(coeffs, theta):
    """Evaluate |P(e^{i theta})| for given coefficients and angle."""
    # Use direct evaluation for accuracy at specific points
    z = np.exp(1j * theta)
    # Horner's method
    val = coeffs[-1]
    for k in range(N - 2, -1, -1):
        val = val * z + coeffs[k]
    return abs(val)

def _supnorm_fast(coeffs, n_points=2048):
    """Fast sup-norm estimate using dense sampling."""
    thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    z = np.exp(1j * thetas)
    # Evaluate P(z) for all thetas using vectorized dot product
    # coeffs shape (N,), z shape (n_points,)
    # P(z_j) = sum_k c_k * z_j^k
    # Create matrix of z_j^k
    powers = np.outer(z, np.arange(N))  # (n_points, N)
    vals = powers @ coeffs  # (n_points,)
    return float(np.max(np.abs(vals)))

def _supnorm_refined(coeffs, n_coarse=2048, n_fine_per_peak=50, n_peaks=5):
    """
    Two-tier sup-norm: coarse sampling + golden-section refinement of top peaks.
    Returns the estimated maximum |P(z)| on |z|=1.
    """
    # Coarse sampling
    thetas_coarse = np.linspace(0, 2 * np.pi, n_coarse, endpoint=False)
    z_coarse = np.exp(1j * thetas_coarse)
    powers = np.outer(z_coarse, np.arange(N))
    vals = powers @ coeffs
    abs_vals = np.abs(vals)
    
    max_val = float(np.max(abs_vals))
    
    # Find top peaks
    idx = np.argsort(abs_vals)[-n_peaks:]
    idx = np.sort(idx)
    
    # Refine each peak using golden-section search
    golden = (np.sqrt(5) - 1) / 2  # ~0.618
    best_peak = max_val
    
    for i in idx:
        theta0 = thetas_coarse[i]
        # Search in a small window around the peak
        window = 2 * np.pi / n_coarse
        a = theta0 - window
        b = theta0 + window
        
        # Golden-section search for maximum of |P(e^{i theta})|
        c = b - golden * (b - a)
        d = a + golden * (b - a)
        fc = _eval_at(coeffs, c)
        fd = _eval_at(coeffs, d)
        
        for _ in range(30):  # ~30 iterations gives ~1e-9 accuracy
            if fc > fd:
                b = d
                d = c
                fd = fc
                c = b - golden * (b - a)
                fc = _eval_at(coeffs, c)
            else:
                a = c
                c = d
                fc = fd
                d = a + golden * (b - a)
                fd = _eval_at(coeffs, d)
        
        peak_val = max(fc, fd)
        if peak_val > best_peak:
            best_peak = peak_val
    
    return best_peak

def _local_search(coeffs, max_iter=200, time_limit=10.0):
    """
    Local search: flip single coefficients to reduce sup-norm.
    Returns improved coefficients.
    """
    start_time = time.time()
    current = coeffs.copy()
    current_norm = _supnorm_refined(current)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        best_flip = -1
        best_norm = current_norm
        
        for k in range(N):
            # Try flipping coefficient k
            trial = current.copy()
            trial[k] = -trial[k]
            # Quick check: only do refined if fast estimate is promising
            fast_est = _supnorm_fast(trial, n_points=512)
            if fast_est < current_norm * 0.999:  # Only refine if significantly better
                refined_est = _supnorm_refined(trial)
                if refined_est < best_norm:
                    best_norm = refined_est
                    best_flip = k
        
        if best_flip >= 0:
            current[best_flip] = -current[best_flip]
            current_norm = best_norm
        else:
            break
    
    return current, current_norm

def _two_flip_search(coeffs, max_iter=50, time_limit=15.0):
    """
    2-flip local search: try flipping pairs of coefficients.
    """
    start_time = time.time()
    current = coeffs.copy()
    current_norm = _supnorm_refined(current)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        best_pair = (-1, -1)
        best_norm = current_norm
        
        for i in range(N):
            for j in range(i + 1, N):
                if time.time() - start_time > time_limit:
                    break
                trial = current.copy()
                trial[i] = -trial[i]
                trial[j] = -trial[j]
                fast_est = _supnorm_fast(trial, n_points=256)
                if fast_est < current_norm * 0.999:
                    refined_est = _supnorm_refined(trial)
                    if refined_est < best_norm:
                        best_norm = refined_est
                        best_pair = (i, j)
        
        if best_pair != (-1, -1):
            i, j = best_pair
            current[i] = -current[i]
            current[j] = -current[j]
            current_norm = best_norm
        else:
            break
    
    return current, current_norm

def _golay_rudin_shapiro():
    """Generate Golay-Rudin-Shapiro sequence truncated to 24 coefficients."""
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    return np.array(p[:N], dtype=float)

def construct():
    """
    Find 24 ±1 coefficients minimizing max_{|z|=1} |P(z)| / sqrt(24).
    Returns list of 24 coefficients in {+1, -1}.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (150s limit, 35s buffer)
    
    # Start with Golay-Rudin-Shapiro sequence
    best_coeffs = _golay_rudin_shapiro()
    best_norm = _supnorm_refined(best_coeffs)
    
    # Multiple random restarts with local search
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate a random starting point
        if restart == 0:
            current = _golay_rudin_shapiro()
        else:
            # Start with GR and flip some random coefficients
            current = _golay_rudin_shapiro()
            n_flips = random.randint(3, 8)
            flip_indices = random.sample(range(N), n_flips)
            for idx in flip_indices:
                current[idx] = -current[idx]
        
        # 1-flip local search
        current, current_norm = _local_search(current, max_iter=100, time_limit=5.0)
        
        if current_norm < best_norm:
            best_coeffs = current.copy()
            best_norm = current_norm
        
        # 2-flip local search if time permits
        if time.time() < deadline - 20:
            current2, norm2 = _two_flip_search(current, max_iter=30, time_limit=10.0)
            if norm2 < best_norm:
                best_coeffs = current2.copy()
                best_norm = norm2
    
    # Final 1-flip polish
    if time.time() < deadline - 5:
        best_coeffs, best_norm = _local_search(best_coeffs, max_iter=50, time_limit=5.0)
    
    # Return as list of integers
    return [int(c) for c in best_coeffs]
