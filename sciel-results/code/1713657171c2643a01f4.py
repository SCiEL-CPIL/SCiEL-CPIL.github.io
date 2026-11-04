"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr(h):
    """Compute circular autocorrelation of h at all lags."""
    # autocorr[k] = sum_i h[i] * h[(i+k) % N]
    # Use FFT for efficiency
    H = np.fft.fft(h)
    power = H * np.conj(H)
    ac = np.fft.ifft(power).real
    return ac

def _cross_corr_max(h):
    """Compute max of cross-correlation of h with (1-h) at all shifts."""
    ac = _autocorr(h)
    # cross_corr[k] = N/2 - ac[k]
    cc = N / 2.0 - ac
    return np.max(cc), cc

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s
    
    # Parameterize h[i] = 0.5 + sum_{j=1}^{M} a_j * cos(2*pi*j*i/N + phi_j)
    # This ensures mean = 0.5, so sum = N/2
    # We optimize a_j and phi_j to minimize max cross-correlation
    
    M = 20  # number of harmonics
    i = np.arange(N)
    
    # Initialize with small random perturbations
    np.random.seed(42)
    a = np.random.randn(M) * 0.1
    phi = np.random.randn(M) * np.pi
    
    def h_from_params(a, phi):
        h = np.full(N, 0.5)
        for j in range(M):
            h += a[j] * np.cos(2 * np.pi * j * i / N + phi[j])
        # Clip to [0, 1]
        h = np.clip(h, 0.0, 1.0)
        # Re-normalize to ensure sum = N/2
        # Actually, clipping breaks the sum constraint. Let's use a different approach.
        return h
    
    def h_from_params_safe(a, phi):
        """Generate h with guaranteed [0,1] bounds and sum = N/2."""
        h = np.full(N, 0.5)
        for j in range(M):
            h += a[j] * np.cos(2 * np.pi * j * i / N + phi[j])
        # Project to [0,1] while maintaining sum
        # Simple approach: clip and adjust
        h = np.clip(h, 0.0, 1.0)
        # Adjust to fix sum
        current_sum = np.sum(h)
        target_sum = N / 2.0
        if current_sum > 0 and current_sum < N:
            # Scale toward target
            # Find elements not at bounds
            mask = (h > 0.01) & (h < 0.99)
            if np.sum(mask) > 0:
                h[mask] += (target_sum - current_sum) / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
        return h
    
    def objective(params):
        a = params[:M]
        phi = params[M:]
        h = h_from_params_safe(a, phi)
        # Verify sum
        if abs(np.sum(h) - N/2.0) > 1e-6:
            return 1e10
        ac = _autocorr(h)
        cc = N / 2.0 - ac
        # Use soft-max to make differentiable
        beta = 20.0
        soft_max = np.log(np.sum(np.exp(beta * cc))) / beta
        return soft_max
    
    # Gradient-free optimization using Nelder-Mead or similar
    # Since we have 40 parameters, let's use a simpler approach
    # Use coordinate descent with soft-max
    
    params = np.concatenate([a, phi])
    best_params = params.copy()
    best_score = objective(params)
    
    # Use scipy if available for optimization
    try:
        from scipy.optimize import minimize
        # Use L-BFGS-B with bounds
        bounds = [(-0.5, 0.5)] * M + [(-np.pi, np.pi)] * M
        
        result = minimize(
            objective,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun
    except:
        pass
    
    # Try multiple restarts if time permits
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        np.random.seed(42 + r * 1000)
        a_r = np.random.randn(M) * 0.15
        phi_r = np.random.randn(M) * np.pi
        params_r = np.concatenate([a_r, phi_r])
        
        try:
            from scipy.optimize import minimize
            bounds = [(-0.5, 0.5)] * M + [(-np.pi, np.pi)] * M
            result = minimize(
                objective,
                params_r,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except:
            pass
    
    # Final refinement: try to directly optimize h with projection
    # Use the best harmonic solution as starting point
    a_best = best_params[:M]
    phi_best = best_params[M:]
    h_best = h_from_params_safe(a_best, phi_best)
    
    # Local refinement using projected gradient on h directly
    # Objective: minimize max_k (N/2 - autocorr[k])
    # = maximize min_k autocorr[k]
    
    def h_objective(h):
        ac = _autocorr(h)
        cc = N / 2.0 - ac
        beta = 50.0
        return np.log(np.sum(np.exp(beta * cc))) / beta
    
    def project_h(h):
        """Project h to [0,1]^N with sum = N/2."""
        h = np.clip(h, 0.0, 1.0)
        for _ in range(20):
            s = np.sum(h)
            if abs(s - N/2.0) < 1e-10:
                break
            diff = (N/2.0 - s) / N
            mask = (h > 0.001) & (h < 0.999)
            if np.sum(mask) == 0:
                break
            h[mask] += diff
            h = np.clip(h, 0.0, 1.0)
        return h
    
    # Gradient of h_objective w.r.t. h
    # d/dh_i [soft_max(cc)] where cc[k] = N/2 - ac[k]
    # ac[k] = sum_j h[j] * h[(j+k)%N]
    # dac/dh_i = sum_k h[(i+k)%N] for the term involving h[i] as first factor
    #           + sum_k h[(i-k)%N] for h[i] as second factor (by symmetry, same)
    # Actually ac[k] = sum_j h[j]*h[(j+k)%N]
    # d ac[k]/d h[i] = h[(i+k)%N] + h[(i-k)%N] (circular)
    # So d cc[k]/d h[i] = -h[(i+k)%N] - h[(i-k)%N]
    # soft_max derivative: softmax(cc) * d cc[k]/d h[i]
    
    def h_grad(h):
        ac = _autocorr(h)
        cc = N / 2.0 - ac
        beta = 50.0
        # softmax weights
        cc_shifted = cc - np.max(cc)
        w = np.exp(beta * cc_shifted)
        w = w / np.sum(w)
        # grad wrt h[i] = -sum_k w[k] * (h[(i+k)%N] + h[(i-k)%N])
        # This is -2 * (w * h) circular convolution, but let's compute directly
        # Actually: sum_k w[k]*h[(i+k)%N] is (w * h) circular conv at i
        # And sum_k w[k]*h[(i-k)%N] is (w * h) circular conv at i as well (by symmetry of circular conv with real signals)
        # So grad = -2 * ifft(fft(w) * fft(h)).real
        W = np.fft.fft(w)
        H = np.fft.fft(h)
        grad = -2.0 * np.fft.ifft(W * H).real
        return grad
    
    # Gradient descent with projection
    h_curr = h_best.copy()
    lr = 0.01
    for it in range(200):
        if time.time() > deadline - 5:
            break
        g = h_grad(h_curr)
        h_new = project_h(h_curr - lr * g)
        obj_new = h_objective(h_new)
        obj_old = h_objective(h_curr)
        if obj_new < obj_old:
            h_curr = h_new
            # Adaptive learning rate
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.7
            if lr < 1e-6:
                break
    
    # Final check and return
    final_h = project_h(h_curr)
    # Ensure exact sum
    s = np.sum(final_h)
    if abs(s - N/2.0) > 1e-9:
        # Tiny adjustment
        mask = (final_h > 0.001) & (final_h < 0.999)
        if np.sum(mask) > 0:
            final_h[mask] += (N/2.0 - s) / np.sum(mask)
            final_h = np.clip(final_h, 0.0, 1.0)
    
    return final_h

# EVOLVE-BLOCK-END
