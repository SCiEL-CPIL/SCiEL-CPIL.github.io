"""Optimized program for the second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 115.0  # seconds, safety margin before 120s kill

def _autoconvolution_peak(h):
    """Compute the supremum norm of the autoconvolution of h."""
    conv = np.convolve(h, h, mode='full')
    return np.max(conv)

def _l2_norm_sq(h):
    """Squared L2 norm."""
    return np.sum(h**2)

def _l1_norm(h):
    """L1 norm."""
    return np.sum(h)

def _ratio(h):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
    l2sq = _l2_norm_sq(h)
    l1 = _l1_norm(h)
    peak = _autoconvolution_peak(h)
    if l1 == 0 or peak == 0:
        return 0.0
    return l2sq / (l1 * peak)

def _smooth_max(x, sigma=1.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    x_shifted = x - x_max
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_shifted / sigma))) * sigma
    return log_sum_exp

def _objective_complex(g_flat):
    """
    Objective to minimize: negative of the ratio.
    g_flat is a real vector of length 2*N representing complex g.
    h = |g|^2.
    """
    g = g_flat.reshape(N, 2)
    h = g[:, 0]**2 + g[:, 1]**2
    
    # Normalize h to have L1 norm 1 to ensure scale invariance and stability
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h = h / l1
    
    l2sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    peak = _smooth_max(conv, sigma=0.01)
    
    if peak == 0:
        return 0.0
    
    return -l2sq / peak  # Since l1 is 1, ratio is l2sq / peak

def _optimize_complex(g0, time_limit):
    """Run L-BFGS-B optimization on complex parameters."""
    start_time = time.time()
    
    # No bounds, as g can be any real number
    result = minimize(
        _objective_complex,
        g0,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
    )
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        # Return h corresponding to initial g0 if timeout
        g = g0.reshape(N, 2)
        h = g[:, 0]**2 + g[:, 1]**2
        return h
    
    g_opt = result.x.reshape(N, 2)
    h_opt = g_opt[:, 0]**2 + g_opt[:, 1]**2
    return h_opt

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Generate diverse initializations for h, then convert to g
    initializations_h = []
    
    # 1. Uniform
    initializations_h.append(np.ones(N))
    
    # 2. Gaussian-like
    x = np.linspace(-1, 1, N)
    initializations_h.append(np.exp(-x**2 / 0.1))
    initializations_h.append(np.exp(-x**2 / 0.5))
    initializations_h.append(np.exp(-x**2 / 2.0))
    
    # 3. Bimodal
    initializations_h.append(np.exp(-((x - 0.4)**2) / 0.1) + np.exp(-((x + 0.4)**2) / 0.1))
    initializations_h.append(np.exp(-((x - 0.6)**2) / 0.05) + np.exp(-((x + 0.6)**2) / 0.05))
    
    # 4. Triangle / tent
    initializations_h.append(1 - np.abs(x))
    
    # 5. Cosine window
    initializations_h.append(0.5 + 0.5 * np.cos(np.pi * x))
    
    # 6. Random smooth
    for seed in [1, 2, 3, 4, 5]:
        rng_local = np.random.RandomState(seed)
        h_rand = np.abs(rng_local.randn(N))
        kernel = np.ones(5) / 5
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        initializations_h.append(h_smooth)
    
    # 7. Peak in center
    h_center = np.zeros(N)
    h_center[N//2 - 2:N//2 + 3] = 1.0
    initializations_h.append(h_center)
    
    # 8. Peak at edges
    h_edge = np.zeros(N)
    h_edge[0:3] = 1.0
    h_edge[-3:] = 1.0
    initializations_h.append(h_edge)
    
    # 9. Alternating
    h_alt = np.ones(N)
    h_alt[::2] = 0.5
    initializations_h.append(h_alt)
    
    # 10. Linear ramp
    initializations_h.append(np.linspace(0.1, 1.0, N))
    initializations_h.append(np.linspace(1.0, 0.1, N))
    
    # 11. DFT-based
    for seed in [10, 20, 30]:
        rng_local = np.random.RandomState(seed)
        z = rng_local.randn(N) + 1j * rng_local.randn(N)
        H = np.fft.fft(z)
        h_dft = np.abs(H)**2
        h_dft = h_dft / np.max(h_dft)
        initializations_h.append(h_dft)
        
    # 12. More structured patterns
    initializations_h.append(np.sin(np.pi * (x + 1) / 2)**2)
    initializations_h.append(np.cos(np.pi * x / 2)**2)

    time_per_opt = 8.0  # seconds per optimization attempt

    for i, h0 in enumerate(initializations_h):
        elapsed = time.time() - start_time
        if elapsed + time_per_opt > _DEADLINE:
            break
        
        # Convert h0 to g0
        # h0 = |g0|^2 => g0 = sqrt(h0) * exp(i * phase)
        # We can just use real g0 = sqrt(h0), imag g0 = 0
        g0_real = np.sqrt(h0)
        g0_imag = np.zeros(N)
        g0_flat = np.concatenate([g0_real, g0_imag])
        
        h_opt = _optimize_complex(g0_flat, time_per_opt)
        
        score = _ratio(h_opt)
        if score > best_score:
            best_score = score
            best_h = h_opt
    
    # Final polish: run a few more iterations from the best solution
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 10:
        # Try small perturbations around the best
        for pert_seed in [100, 200, 300]:
            elapsed = time.time() - start_time
            if elapsed + 5 > _DEADLINE:
                break
            rng_local = np.random.RandomState(pert_seed)
            # Perturb in g-space
            g_best_real = np.sqrt(best_h)
            g_best_imag = np.zeros(N)
            
            pert_real = g_best_real + rng_local.randn(N) * 0.05
            pert_imag = rng_local.randn(N) * 0.05
            
            g_pert_flat = np.concatenate([pert_real, pert_imag])
            
            h_opt = _optimize_complex(g_pert_flat, 5.0)
            score = _ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt
    
    return list(best_h)
