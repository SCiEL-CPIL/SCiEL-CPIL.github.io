"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Initialize with smooth Gaussian bump family.
2. Simulated annealing with discrete interval flips to escape local minima.
3. Continuous sensitivity-based refinement for final polish.

Key improvements:
- Discrete simulated annealing leveraging the 600-bin structure
- FFT-based autoconvolution for O(N log N) per evaluation
- Adaptive temperature schedule with restarts
- Continuous refinement as final polish
"""

import numpy as np
from scipy.signal import fftconvolve
import time

N = 600
TIME_LIMIT = 138  # safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = fftconvolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def fast_conv_max(f):
    """Fast computation of max autoconvolution value (unnormalized)."""
    conv = fftconvolve(f, f, mode='full')
    return np.max(conv)

def gaussian_basis(x, centers, widths, heights):
    """Generate a smooth nonnegative function from Gaussian bump components."""
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        result += h * np.exp(-0.5 * ((x - c) / (w / 2.0)) ** 2)
    return result

def simulate_annealing(f_init, start_time, max_time, n_restarts=3):
    """
    Simulated annealing with discrete interval flips.
    Each step flips one bin to a random value, accepting based on Metropolis criterion.
    """
    best_f = f_init.copy()
    best_val = compute_score(f_init)
    
    for restart in range(n_restarts):
        if time.time() - start_time > max_time:
            break
        
        current_f = best_f.copy() if restart > 0 else f_init.copy()
        current_val = best_val if restart > 0 else compute_score(f_init)
        
        # Temperature schedule
        T0 = 0.5  # initial temperature
        T_min = 0.001
        n_steps = 5000
        cooling_rate = (T_min / T0) ** (1.0 / n_steps)
        
        T = T0
        rng = np.random.RandomState(42 + restart * 1000)
        
        # Precompute sum for normalization
        s = np.sum(current_f)
        
        for step in range(n_steps):
            if time.time() - start_time > max_time:
                break
            
            T *= cooling_rate
            
            # Choose a random bin to modify
            idx = rng.randint(N)
            old_val = current_f[idx]
            
            # New value: random in [0, 2*mean] or set to 0
            mean_val = s / N
            if rng.random() < 0.3:
                new_val = 0.0
            else:
                new_val = rng.uniform(0, 2.0 * mean_val + 0.1)
            
            # Apply change
            current_f[idx] = new_val
            new_s = s - old_val + new_val
            
            if new_s < 1e-12:
                current_f[idx] = old_val
                s = old_s
                continue
            
            # Compute new score
            conv = fftconvolve(current_f, current_f, mode='full')
            new_val_score = np.max(conv) / (new_s * new_s)
            
            # Metropolis acceptance
            delta = new_val_score - current_val
            if delta < 0 or rng.random() < np.exp(-delta / (T + 1e-15)):
                # Accept
                current_val = new_val_score
                s = new_s
                if new_val_score < best_val:
                    best_val = new_val_score
                    best_f = current_f.copy()
            else:
                # Reject
                current_f[idx] = old_val
        
        # After each restart, do a quick local refinement
        if time.time() - start_time < max_time - 5:
            refined, refined_val = _quick_refine(best_f, start_time, max_time - 3)
            if refined_val < best_val:
                best_f = refined
                best_val = refined_val
    
    return best_f, best_val

def _quick_refine(f, start_time, max_time, max_iter=50):
    """Quick local refinement using sensitivity-based approach."""
    current_f = f.copy()
    current_val = compute_score(f)
    best_f = f.copy()
    best_val = current_val
    alpha = 0.05
    
    for _ in range(max_iter):
        if time.time() - start_time > max_time:
            break
        
        s = np.sum(current_f)
        if s < 1e-12:
            break
        current_f = current_f * (N / s)
        
        conv = fftconvolve(current_f, current_f, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
        
        threshold = max_c * 0.95
        peak_indices = np.where(conv >= threshold)[0]
        
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                sensitivity[j_min:j_max+1] += current_f[m - j_max:m - j_min + 1]
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_f * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fftconvolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_f = h_new
            alpha = min(alpha * 1.03, 0.3)
        else:
            alpha *= 0.9
            if alpha < 1e-8:
                break
    
    return best_f, best_val

def local_refinement(h, start_time, max_time, alpha_init=0.1, threshold_frac=0.9):
    """Local refinement using sensitivity-based peak reduction with FFT."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fftconvolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute sensitivity using FFT-based approach
        # For each bin j, sensitivity = sum over peak lags m of h[m-j]
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                sensitivity[j_min:j_max+1] += current_h[m - j_max:m - j_min + 1]
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fftconvolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.04, 0.5)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def local_refinement(h, start_time, max_time, alpha_init=0.1, threshold_frac=0.9):
    """Local refinement using sensitivity-based peak reduction with FFT."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fftconvolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute sensitivity using FFT-based approach
        # For each bin j, sensitivity = sum over peak lags m of h[m-j]
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                sensitivity[j_min:j_max+1] += current_h[m - j_max:m - j_min + 1]
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fftconvolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.04, 0.5)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """Final high-precision polish pass with very small step sizes."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.03
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fftconvolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        threshold = max_c * 0.995
        peak_indices = np.where(conv >= threshold)[0]
        
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                sensitivity[j_min:j_max+1] += current_h[m - j_max:m - j_min + 1]
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fftconvolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.2)
        else:
            alpha *= 0.9
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of Gaussian bump components for initialization
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    for n_c in [2, 3, 4, 5, 6]:
        if time.time() - start_time > TIME_LIMIT - 50:
            break
        
        rng = np.random.RandomState(42 + n_c)
        
        for restart in range(6):
            if time.time() - start_time > TIME_LIMIT - 45:
                break
            
            # Structured initialization
            if restart == 0:
                # Symmetric initialization
                centers = np.linspace(-0.15, 0.15, n_c)
                widths = np.full(n_c, 0.12)
                heights = np.full(n_c, 1.0)
            elif restart == 1:
                # Asymmetric with varying widths
                centers = rng.uniform(-0.2, 0.2, size=n_c)
                widths = rng.uniform(0.04, 0.2, size=n_c)
                heights = rng.uniform(0.5, 2.0, size=n_c)
            elif restart == 2:
                # Concentrated at center
                centers = np.zeros(n_c) + rng.uniform(-0.05, 0.05, size=n_c)
                widths = rng.uniform(0.03, 0.1, size=n_c)
                heights = rng.uniform(0.8, 2.5, size=n_c)
            elif restart == 3:
                # Wide spread
                centers = rng.uniform(-0.22, 0.22, size=n_c)
                widths = rng.uniform(0.1, 0.3, size=n_c)
                heights = rng.uniform(0.3, 1.5, size=n_c)
            else:
                centers = rng.uniform(-0.2, 0.2, size=n_c)
                widths = rng.uniform(0.02, 0.25, size=n_c)
                heights = rng.uniform(0.3, 3.0, size=n_c)
            
            f = gaussian_basis(x, centers, widths, heights)
            f = np.maximum(f, 0.0)
            
            # Normalize
            s = np.sum(f)
            if s < 1e-12:
                continue
            f = f * (N / s)
            
            score = compute_score(f)
            if score < best_score:
                best_f = f.copy()
                best_score = score
    
    # Simulated annealing with discrete flips (main optimization)
    if time.time() - start_time < TIME_LIMIT - 40:
        sa_f, sa_score = simulate_annealing(
            best_f, start_time, TIME_LIMIT - 20, n_restarts=3
        )
        if sa_score < best_score:
            best_f = sa_f
            best_score = sa_score
    
    # Local refinement: sensitivity-based peak reduction
    if time.time() - start_time < TIME_LIMIT - 15:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 8
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Final polish pass
    if time.time() - start_time < TIME_LIMIT - 5:
        polished_f, polished_score = polish_pass(
            best_f, start_time, TIME_LIMIT - 2
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
