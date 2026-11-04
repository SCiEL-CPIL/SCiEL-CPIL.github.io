"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy import optimize

N = 600
DELTA = 0.5 / N  # interval width

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    x = np.linspace(-0.25, 0.25, N, endpoint=False)

    def autoconv_max_ratio(h):
        """Compute max(autoconv) / ||h||_2^2 for step function h on N equal intervals."""
        h = np.asarray(h, dtype=np.float64)
        if h.size != N:
            h = np.resize(h, N)
        h = np.maximum(h, 0.0)
        
        n_fft = 1
        while n_fft < 2 * N:
            n_fft *= 2
            
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        
        H = np.fft.rfft(h_padded)
        autoconv_circ = np.fft.irfft(H * H, n_fft)
        
        # Linear autoconvolution is in the first 2N-1 entries
        autoconv_vals = autoconv_circ[:2*N-1] * DELTA
        max_autoconv = np.max(autoconv_vals)
        
        l2_sq = np.sum(h**2) * DELTA
        if l2_sq < 1e-15:
            return float('inf')
        
        return max_autoconv / l2_sq

    def fourier_to_h(params):
        """
        Convert Fourier coefficients to non-negative heights.
        params: [a0, b1, a1, b2, a2, ..., bK, aK]
        h(x) = a0 + sum_{k=1}^K (a_k * cos(2*pi*k*x/L) + b_k * sin(2*pi*k*x/L))
        where L = 0.5 (length of interval)
        """
        K = 20  # number of Fourier modes
        h = np.full(N, params[0])
        
        for k in range(1, K + 1):
            a_k = params[2*k - 1]
            b_k = params[2*k]
            h += a_k * np.cos(2 * np.pi * k * x / 0.5)
            h += b_k * np.sin(2 * np.pi * k * x / 0.5)
        
        return np.maximum(h, 0.0)

    def objective_fourier(params):
        h = fourier_to_h(params)
        return autoconv_max_ratio(h)

    def coordinate_descent_refine(h, max_iters=200, step_decay=0.95):
        """Refine h using coordinate descent to minimize autoconv_max_ratio."""
        h = np.asarray(h, dtype=np.float64).copy()
        current_score = autoconv_max_ratio(h)
        step_size = 0.05
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                # Try increasing and decreasing h[i]
                h_trial = h.copy()
                h_trial[i] = max(0, h[i] + step_size)
                trial_score = autoconv_max_ratio(h_trial)
                if trial_score < current_score:
                    h = h_trial
                    current_score = trial_score
                    improved = True
                    continue
                    
                h_trial[i] = max(0, h[i] - step_size)
                trial_score = autoconv_max_ratio(h_trial)
                if trial_score < current_score:
                    h = h_trial
                    current_score = trial_score
                    improved = True
            
            if not improved:
                step_size *= step_decay
                if step_size < 1e-4:
                    break
            
        return h, current_score

    best_score = float('inf')
    best_h = None

    # 1. Optimize in Fourier space with L-BFGS-B
    # This allows for more complex shapes than simple envelope modulation
    K = 20  # number of Fourier modes
    n_params = 1 + 2 * K  # a0 + K pairs of (a_k, b_k)
    
    # Initial guess: constant function
    x0 = np.zeros(n_params)
    x0[0] = 1.0  # a0 = 1
    
    # Bounds: allow negative coefficients but clip to non-negative after conversion
    bounds = [(-5.0, 5.0)] * n_params
    
    try:
        res = optimize.minimize(
            objective_fourier, x0, method='L-BFGS-B',
            bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8}
        )
        if res.fun < best_score:
            best_score = res.fun
            best_h = fourier_to_h(res.x).copy()
    except Exception:
        pass

    # 2. Fallback to diverse candidates if Fourier optimization fails or is slow
    if best_h is None or time.time() < deadline - 30:
        centers = np.arange(N) - N // 2
        candidates = []
        
        # Gaussian peaks
        for sigma in [1, 2, 5, 10, 20, 50, 100]:
            h = np.exp(-centers**2 / (2 * sigma**2))
            candidates.append(h)
        
        # Triangular peaks
        for width in [10, 50, 100, 200, 300]:
            h = np.maximum(0, 1 - np.abs(centers) / width)
            candidates.append(h)
            
        # Box functions
        for half_width in [10, 50, 100, 200, 300]:
            h = np.zeros(N)
            mask = np.abs(centers) <= half_width
            h[mask] = 1.0
            candidates.append(h)

        # Gaussian + base
        for sigma in [5, 20, 50]:
            for base in [0.1, 0.3, 0.5]:
                h = base + (1 - base) * np.exp(-centers**2 / (2 * sigma**2))
                candidates.append(h)

        for h in candidates:
            if time.time() > deadline - 20:
                break
            h = np.asarray(h, dtype=np.float64)
            h = np.maximum(h, 0)
            if np.sum(h) < 1e-10:
                continue
            score = autoconv_max_ratio(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()

    # 3. Refine the best candidate with coordinate descent
    if best_h is not None and time.time() < deadline - 5:
        refined_h, refined_score = coordinate_descent_refine(best_h, max_iters=100, step_decay=0.9)
        if refined_score < best_score:
            best_score = refined_score
            best_h = refined_h

    if best_h is None:
        return [1.0] * N

    return list(best_h)

# EVOLVE-BLOCK-END
