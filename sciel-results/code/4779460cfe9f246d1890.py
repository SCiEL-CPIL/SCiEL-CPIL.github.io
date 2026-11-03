import numpy as np
import time
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    dx = 1.0 / (2 * N)  # interval width: [-1/4, 1/4] has length 1/2, N intervals

    def autoconv_max_ratio(h):
        """Compute max(autoconv) / ||h||_2^2 for step function h on N equal intervals."""
        h = np.asarray(h, dtype=np.float64)
        if np.any(h < 0):
            h = np.maximum(h, 0)
        l2_sq = np.sum(h**2) * dx
        if l2_sq < 1e-15:
            return float('inf')
        
        # Use FFT for efficient autoconvolution
        n_fft = 2 * N
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        
        H = np.fft.fft(h_padded)
        autoconv_circ = np.fft.ifft(H * H).real
        
        # The linear autoconvolution has support from -(N-1)*dx to (N-1)*dx
        # With circular conv of size 2N, indices 0..N-1 correspond to lags 0..(N-1)
        # and indices N..2N-1 correspond to lags -(N-1)..-1
        # By symmetry, we can just take max over all 2N-1 valid entries
        
        autoconv_vals = autoconv_circ[:2*N-1] * dx**2
        max_autoconv = np.max(autoconv_vals)
        
        return max_autoconv / l2_sq

    def coordinate_descent_refine(h, max_iters=500, step_decay=0.95):
        """Refine h using coordinate descent to minimize autoconv_max_ratio."""
        h = np.asarray(h, dtype=np.float64).copy()
        current_score = autoconv_max_ratio(h)
        step_size = 0.1
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                # Try increasing and decreasing h[i]
                for direction in [1, -1]:
                    h_trial = h.copy()
                    h_trial[i] = max(0, h[i] + direction * step_size)
                    trial_score = autoconv_max_ratio(h_trial)
                    if trial_score < current_score:
                        h = h_trial
                        current_score = trial_score
                        improved = True
                        break
                if not improved and direction == -1:
                    step_size *= step_decay
            if not improved:
                break
        
        return h, current_score

    def fourier_to_heights(coeffs, num_harmonics=15):
        """Convert Fourier coefficients to spatial heights."""
        x = np.linspace(-1/4, 1/4, N)
        v = np.zeros(N)
        # coeffs[0] is the constant term
        v += coeffs[0]
        for m in range(1, num_harmonics + 1):
            # Frequency m on domain length 1/2 -> angular freq 4*pi*m
            v += coeffs[m] * np.cos(4 * np.pi * m * x)
        h = np.exp(v)
        # Normalize to avoid overflow/underflow and scale invariance issues
        # We normalize such that max(h) = 1
        max_h = np.max(h)
        if max_h > 0:
            h = h / max_h
        return h

    def fit_fourier_coeffs(h, num_harmonics=15):
        """Fit Fourier coefficients to log(h) to initialize optimization."""
        h = np.asarray(h, dtype=np.float64)
        h = np.maximum(h, 1e-10)
        log_h = np.log(h)
        x = np.linspace(-1/4, 1/4, N)
        
        A = np.zeros((N, num_harmonics + 1))
        A[:, 0] = 1.0
        for m in range(1, num_harmonics + 1):
            A[:, m] = np.cos(4 * np.pi * m * x)
            
        # Least squares solve A * coeffs = log_h
        coeffs, residuals, rank, s = np.linalg.lstsq(A, log_h, rcond=None)
        return coeffs

    def optimize_fourier(coeffs_init, num_harmonics=15, max_iter=100):
        """Optimize Fourier coefficients to minimize autoconv_max_ratio."""
        def objective(coeffs):
            h = fourier_to_heights(coeffs, num_harmonics)
            return autoconv_max_ratio(h)
        
        # Use L-BFGS-B with bounds to keep coefficients reasonable
        # Bounds: +/- 10 should be enough for exp to not explode after normalization
        bounds = [(-10, 10)] * (num_harmonics + 1)
        
        res = minimize(objective, coeffs_init, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8})
        return res.x, res.fun

    best_score = float('inf')
    best_h = None

    centers = np.arange(N) - N // 2  # centered indices: -300, -299, ..., 299

    # Generate diverse initial candidates
    candidates = []
    
    # 1. Gaussian peaks with various widths
    for sigma in [1, 2, 5, 10, 20, 50, 100]:
        h = np.exp(-centers**2 / (2 * sigma**2))
        candidates.append(h)
    
    # 2. Triangular peaks
    for width in [10, 50, 100, 200, 300]:
        h = np.maximum(0, 1 - np.abs(centers) / width)
        candidates.append(h)
    
    # 3. Box functions (constant on central region)
    for half_width in [10, 50, 100, 200, 300]:
        h = np.zeros(N)
        mask = np.abs(centers) <= half_width
        h[mask] = 1.0
        candidates.append(h)
    
    # 4. Gaussian + base
    for sigma in [5, 20, 50]:
        for base in [0.1, 0.3, 0.5]:
            h = base + (1 - base) * np.exp(-centers**2 / (2 * sigma**2))
            candidates.append(h)
    
    # 5. Inverted Gaussian (valley in center, high at edges)
    for sigma in [50, 100, 200]:
        h = 1.0 - np.exp(-centers**2 / (2 * sigma**2))
        h = np.maximum(h, 0)
        candidates.append(h)
    
    # 6. Random smooth functions
    np.random.seed(42)
    for trial in range(5):
        freqs = np.random.uniform(0.5, 5.0, size=5)
        phases = np.random.uniform(0, 2*np.pi, size=5)
        amps = np.random.uniform(0.5, 2.0, size=5)
        h = np.ones(N)
        for f, p, a in zip(freqs, phases, amps):
            h += a * np.cos(f * centers / N * 2 * np.pi + p)
        h = np.maximum(h, 0)
        if np.sum(h) > 1e-10:
            candidates.append(h)

    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        h = np.asarray(h, dtype=np.float64)
        if np.any(h < 0):
            h = np.maximum(h, 0)
        if np.sum(h) < 1e-10:
            continue
        score = autoconv_max_ratio(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()

    # Refine the best candidate with coordinate descent
    if best_h is not None and time.time() < deadline:
        refined_h, refined_score = coordinate_descent_refine(best_h, max_iters=300, step_decay=0.9)
        if refined_score < best_score:
            best_score = refined_score
            best_h = refined_h
    
    # Additional refinement passes with smaller steps
    if best_h is not None and time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            refined_h, refined_score = coordinate_descent_refine(best_h, max_iters=100, step_decay=0.9)
            if refined_score < best_score:
                best_score = refined_score
                best_h = refined_h
            else:
                break

    # Fourier-based optimization
    if best_h is not None and time.time() < deadline - 20:
        num_harmonics = 15
        # Fit initial coefficients from the current best spatial solution
        coeffs_init = fit_fourier_coeffs(best_h, num_harmonics)
        
        # Optimize
        try:
            coeffs_opt, score_opt = optimize_fourier(coeffs_init, num_harmonics, max_iter=50)
            h_fourier = fourier_to_heights(coeffs_opt, num_harmonics)
            
            # Evaluate the Fourier-optimized solution
            score_fourier = autoconv_max_ratio(h_fourier)
            
            if score_fourier < best_score:
                best_score = score_fourier
                best_h = h_fourier.copy()
                
                # Polish with coordinate descent
                if time.time() < deadline - 10:
                    refined_h, refined_score = coordinate_descent_refine(best_h, max_iters=200, step_decay=0.9)
                    if refined_score < best_score:
                        best_score = refined_score
                        best_h = refined_h
        except Exception as e:
            pass # If optimization fails, keep previous best

    if best_h is None:
        return [1.0] * N

    return list(best_h)

# EVOLVE-BLOCK-END
