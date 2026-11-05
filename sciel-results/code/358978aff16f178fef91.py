"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Minimize value = abs(2*N*max(convolve(h,h)) / (sum(h))^2)
    using Zadoff-Chu sequences modulated by chirps and windowed.
    """
    best_h = np.ones(N)
    best_val = 2.0  # seed value
    
    def evaluate(h):
        if not np.all(np.isfinite(h)):
            return float('inf')
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        conv = np.convolve(h, h)
        m = np.max(conv)
        val = abs(2 * N * m / (s * s))
        return val

    t = np.arange(N, dtype=np.float64)
    
    def get_h(params):
        mu, alpha, win_param, dc = params
        # Zadoff-Chu like sequence
        zc = np.exp(-1j * np.pi * mu * t * (t + 1) / N)
        # Chirp modulation
        chirp = np.exp(1j * np.pi * alpha * t * t / N)
        signal = zc * chirp
        
        # Windowing: Gaussian or Cosine
        if win_param < 1.0:
            n_norm = (t - (N-1)/2.0) / ((N-1)/2.0)
            hann = 0.5 * (1 + np.cos(np.pi * n_norm))
            window = (1 - win_param) * 1.0 + win_param * hann
        else:
            sigma = win_param - 1.0 + 0.1
            n_norm = (t - (N-1)/2.0) / ((N-1)/2.0)
            window = np.exp(-0.5 * (n_norm / (sigma/2.0))**2)
        
        h_real = np.real(signal) * window + dc
        return h_real

    best_candidate = None
    best_candidate_val = float('inf')
    best_params = None
    
    # Fixed seed for reproducibility
    np.random.seed(42)
    
    # Coarse grid search
    mu_vals = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
    alpha_vals = np.linspace(-0.5, 0.5, 11)
    win_vals = [0.0, 0.5, 1.0, 1.5, 2.0]
    dc_vals = [0.0, 0.1, 0.5, 1.0]
    
    for mu in mu_vals:
        for alpha in alpha_vals:
            for w in win_vals:
                for d in dc_vals:
                    h_test = get_h([mu, alpha, w, d])
                    val = evaluate(h_test)
                    if val < best_candidate_val:
                        best_candidate_val = val
                        best_candidate = h_test.copy()
                        best_params = [mu, alpha, w, d]
    
    # Refine best candidates
    if best_candidate is not None and best_params is not None:
        # Try local optimizations
        for _ in range(5):
            # Add small noise to start point
            start = np.array(best_params) + np.random.normal(0, 0.05, 4)
            try:
                res = minimize(
                    lambda p: evaluate(get_h(p)),
                    start,
                    method='Powell',
                    options={'maxiter': 200, 'maxfev': 1000}
                )
                if res.success and res.fun < best_candidate_val:
                    best_candidate_val = res.fun
                    best_candidate = get_h(res.x).copy()
                    best_params = res.x
            except Exception:
                pass

    if best_candidate is not None and best_candidate_val < best_val:
        return best_candidate
    return best_h
# EVOLVE-BLOCK-END
