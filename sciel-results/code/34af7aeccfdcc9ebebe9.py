"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600
DELTA = 0.5 / N  # width of each interval

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def autoconv(h):
        """Compute discrete autoconvolution via FFT."""
        n = len(h)
        size = 2 * n
        f = np.zeros(size)
        f[:n] = h
        F = np.fft.fft(f)
        return np.fft.ifft(F * np.conj(F)).real

    def objective_exact(h):
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        autocorr = autoconv(h)
        max_conv = np.max(autocorr)
        return max_conv / (DELTA * s * s)

    def soft_max(arr, beta):
        arr = np.asarray(arr, dtype=float)
        m = np.max(arr)
        if m < 1e-10:
            return m
        return m + (1.0 / beta) * np.log(np.sum(np.exp(beta * (arr - m))))

    def objective_smooth(h, beta):
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        autocorr = autoconv(h)
        max_conv_smooth = soft_max(autocorr, beta)
        return max_conv_smooth / (DELTA * s * s)

    def fista_smooth(h0, beta, max_iter=200, tol=1e-8):
        """FISTA (accelerated projected gradient) for smooth objective with nonnegativity."""
        h = np.maximum(h0.copy(), 0)
        z = h.copy()
        t = 1.0
        # Step size: use Lipschitz constant estimate
        # For soft-max objective, gradient involves autocorrelation
        # Use a conservative step size
        L = 10.0  # initial Lipschitz estimate
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            # Compute gradient
            hn = np.maximum(h, 0)
            s = np.sum(hn)
            if s < 1e-10:
                break
            autocorr = autoconv(hn)
            m = np.max(autocorr)
            exps = np.exp(beta * (autocorr - m))
            probs = exps / np.sum(exps)
            # Gradient of soft_max w.r.t. autocorr
            # autocorr[k] = sum_i h[i] h[i-k] (for lag k)
            # d(soft_max)/d(autocorr[k]) = probs[k] * beta
            # d(autocorr[k])/d(h[j]) involves contributions from all lags
            # This is complex; use numerical gradient or approximate
            # Instead, use subgradient-like approach: push down the peak lags
            # Simpler: use the fact that we want to reduce the max autocorr
            # Gradient direction: reduce h where it contributes to peak lags
            # Approximate gradient via finite differences is too slow
            # Use: grad_obj ≈ -probs-weighted autocorr contribution
            # For now, use a simple heuristic: reduce h at positions that 
            # contribute to the highest autocorrelation lags
            # Actually, let's just use L-BFGS-B with better settings
            break
        return h

    best_h = np.ones(N)
    best_score = objective_exact(best_h)
    
    bounds = [(0, None)] * N
    np.random.seed(42)

    # Phase 1: L-BFGS-B with smooth surrogate, adaptive beta
    betas = [20.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline - 30:
            break
        
        for restart in range(4):
            if time.time() > deadline - 30:
                break
            
            if restart == 0:
                x0 = np.ones(N)
            elif restart == 1:
                # Gaussian-like shape
                x = np.linspace(-1, 1, N)
                x0 = np.exp(-x**2 * 2.0)
            elif restart == 2:
                # Triangle shape
                x = np.linspace(-1, 1, N)
                x0 = np.maximum(0, 1 - np.abs(x))
            else:
                # Random with smoothing
                x0 = np.random.exponential(1.0, N)
                # Smooth it
                kernel = np.ones(5) / 5
                x0 = np.convolve(x0, kernel, mode='same')
                x0 = np.maximum(x0, 0)
            
            try:
                obj = lambda h: objective_smooth(h, beta)
                result = minimize(obj, x0, method='L-BFGS-B', 
                                 bounds=bounds, 
                                 options={'maxiter': 100, 'maxfun': 300,
                                          'ftol': 1e-10, 'gtol': 1e-8})
                
                if time.time() > deadline - 30:
                    break
                
                h_candidate = np.maximum(result.x, 0)
                exact_score = objective_exact(h_candidate)
                if exact_score < best_score:
                    best_score = exact_score
                    best_h = h_candidate
            except Exception:
                continue
    
    # Phase 2: Local refinement using active set identification
    if time.time() < deadline - 10:
        h_work = best_h.copy()
        s = np.sum(h_work)
        if s > 1e-10:
            autocorr = autoconv(h_work)
            max_val = np.max(autocorr)
            # Find lags that are near the maximum (active set)
            active_mask = autocorr > 0.9 * max_val
            active_lags = np.where(active_mask)[0]
            
            # Try to reduce the peak by adjusting heights
            # Use a few iterations of: compute gradient, take step
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                h_work = np.maximum(h_work, 0)
                s = np.sum(h_work)
                if s < 1e-10:
                    break
                autocorr = autoconv(h_work)
                max_val = np.max(autocorr)
                # Subgradient: direction to decrease the objective
                # The objective is max_conv / (DELTA * s^2)
                # To decrease it, we want to decrease max_conv or increase s
                # Decreasing max_conv: reduce h at positions contributing to peak lags
                # For each peak lag k, autocorr[k] = sum_i h[i]*h[i-k]
                # Gradient w.r.t h[j] for lag k: sum over i where i-k=j or i=j+k
                # Simplified: use the autocorrelation structure
                # Approximate gradient: for each j, grad_j = sum_k probs[k] * (h[j-k] + h[j+k])
                # where probs are softmax weights over autocorr
                m = np.max(autocorr)
                exps = np.exp(50 * (autocorr - m))
                probs = exps / np.sum(exps)
                
                # Compute approximate gradient
                grad = np.zeros(N)
                for k in range(N):
                    if probs[k] > 1e-6:
                        # autocorr[k] = sum_i h[i] h[i-k]
                        # d/dh[j] = sum_i [delta(i,j)*h[i-k] + delta(i-k,j)*h[i]]
                        # = h[j-k] (if j-k in range) + h[j+k] (if j+k in range)
                        for j in range(N):
                            contrib = 0
                            if j - k >= 0 and j - k < N:
                                contrib += h_work[j - k]
                            if j + k >= 0 and j + k < N:
                                contrib += h_work[j + k]
                            grad[j] += probs[k] * contrib
                
                # Normalize and take step
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-10:
                    break
                grad = grad / grad_norm
                # Step size: decrease h where gradient is positive
                step_size = 0.01
                h_new = h_work - step_size * grad
                h_new = np.maximum(h_new, 0)
                # Rescale to maintain similar L1 norm
                s_new = np.sum(h_new)
                if s_new > 1e-10:
                    h_new = h_new * (s / s_new)
                
                new_score = objective_exact(h_new)
                if new_score < best_score:
                    best_score = new_score
                    best_h = h_new
                    h_work = h_new
                else:
                    step_size *= 0.5
    
    # Phase 3: Final polish with L-BFGS-B on exact objective
    if time.time() < deadline - 5:
        try:
            result = minimize(objective_exact, best_h, method='L-BFGS-B', 
                             bounds=bounds, 
                             options={'maxiter': 100, 'maxfun': 200,
                                      'ftol': 1e-12, 'gtol': 1e-10})
            h_candidate = np.maximum(result.x, 0)
            exact_score = objective_exact(h_candidate)
            if exact_score < best_score:
                best_score = exact_score
                best_h = h_candidate
        except Exception:
            pass
    
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
