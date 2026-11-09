"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def ratio(h):
        """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
        h = np.asarray(h, dtype=float)
        if np.any(h < 0):
            return -1.0
        l1 = np.sum(h)
        if l1 < 1e-15:
            return -1.0
        # Autoconvolution via FFT
        h_pad = np.zeros(2 * N - 1)
        h_pad[:N] = h
        conv = np.real(np.fft.ifft(np.fft.fft(h_pad) ** 2))
        peak = np.max(np.abs(conv))
        if peak < 1e-15:
            return -1.0
        l2_sq = np.sum(h ** 2)
        return l2_sq / (l1 * peak)
    
    def neg_ratio(h):
        return -ratio(h)
    
    # Start with a few initial guesses
    best_h = np.ones(N) / N  # uniform, ratio = 1.0
    best_val = ratio(best_h)
    
    # Try some structured initial guesses
    candidates = [
        np.ones(N) / N,
        np.sin(np.linspace(0, np.pi, N)) / N,  # triangular-ish
        np.cos(np.linspace(-np.pi, np.pi, N)) ** 2 / N,
        np.exp(-((np.arange(N) - N/2) ** 2) / (2 * (N/4) ** 2)),
        np.ones(N) * (1 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N)),
    ]
    for c in candidates:
        c = np.clip(c, 0, None)
        if np.sum(c) > 0:
            c = c / np.sum(c)
            v = ratio(c)
            if v > best_val:
                best_val = v
                best_h = c.copy()
    
    # Powell optimization from best initial guess
    # Normalize to L1=1 to reduce dimensionality (scale doesn't matter for ratio)
    x0 = best_h.copy()
    
    n_iter = 0
    while time.time() < deadline:
        result = minimize(
            neg_ratio, x0, method='Powell',
            options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-8, 'disp': False}
        )
        n_iter += 1
        h_new = np.clip(result.x, 0, None)
        if np.sum(h_new) > 1e-15:
            h_new = h_new / np.sum(h_new)
            v = ratio(h_new)
            if v > best_val:
                best_val = v
                best_h = h_new.copy()
        
        # Check if we should stop
        if result.x is None or not np.all(np.isfinite(result.x)):
            break
        
        # If improvement is tiny, try a perturbation
        if best_val > 1.0 and (best_val - 1.0) < 1e-6:
            # Try random perturbations around best
            improved = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                h_pert = best_h + 0.05 * np.random.randn(N)
                h_pert = np.clip(h_pert, 0, None)
                if np.sum(h_pert) > 1e-15:
                    h_pert = h_pert / np.sum(h_pert)
                    v = ratio(h_pert)
                    if v > best_val:
                        best_val = v
                        best_h = h_pert.copy()
                        improved = True
            if not improved:
                break
        else:
            # Continue optimizing from the new point
            x0 = result.x.copy()
    
    # Final polish: Nelder-Mead with tighter tolerance
    if time.time() < deadline - 5:
        result = minimize(
            neg_ratio, best_h, method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False}
        )
        h_new = np.clip(result.x, 0, None)
        if np.sum(h_new) > 1e-15:
            h_new = h_new / np.sum(h_new)
            v = ratio(h_new)
            if v > best_val:
                best_val = v
                best_h = h_new.copy()
    
    # Return as list of nonnegative heights
    return [float(x) for x in best_h]

# EVOLVE-BLOCK-END
