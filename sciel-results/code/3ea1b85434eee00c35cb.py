import numpy as np
import time
from scipy.optimize import differential_evolution

N = 600
TIME_LIMIT = 145  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def fft_autoconv(f):
    """Compute autoconvolution via FFT for speed."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)[:2 * N - 1]
    return conv

def pulse_objective(params):
    """
    Objective function for differential evolution.
    params: [pos_0, width_0, height_0, pos_1, width_1, height_1, ...]
    pos in [-0.25, 0.25], width in [0.001, 0.25], height in [0.0, 10.0]
    """
    if len(params) % 3 != 0:
        return 1e10
    
    n_pulses = len(params) // 3
    f = np.zeros(N)
    
    # Step size in domain
    step = 0.5 / N  # Total length is 0.5, from -0.25 to 0.25
    
    for i in range(n_pulses):
        pos = params[3*i]
        width = params[3*i+1]
        height = params[3*i+2]
        
        # Convert position and width to indices
        # pos is center of pulse
        left = (pos - width/2 + 0.25) / step
        right = (pos + width/2 + 0.25) / step
        
        idx_left = int(np.floor(left))
        idx_right = int(np.ceil(right))
        
        # Clip to valid range
        idx_left = max(0, idx_left)
        idx_right = min(N, idx_right)
        
        if idx_right > idx_left:
            f[idx_left:idx_right] += height
    
    # Check non-negativity (should be guaranteed by height >= 0)
    if np.any(f < -1e-15):
        return 1e10
    
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    
    conv = fft_autoconv(f)
    c_max = np.max(conv)
    return c_max / (s * s)

def construct():
    start_time = time.time()
    
    # Start with a constant function as baseline
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Differential evolution with multiple pulse configurations
    # Try different numbers of pulses
    pulse_counts = [2, 3, 4, 5]
    
    for n_pulses in pulse_counts:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        # Define bounds for each parameter
        # pos: [-0.25, 0.25], width: [0.005, 0.25], height: [0.0, 10.0]
        bounds = []
        for _ in range(n_pulses):
            bounds.append((-0.25, 0.25))  # position
            bounds.append((0.005, 0.25))   # width
            bounds.append((0.0, 10.0))     # height
        
        # Run differential evolution
        try:
            result = differential_evolution(
                pulse_objective,
                bounds=bounds,
                maxiter=50,
                popsize=10,
                tol=1e-4,
                seed=42,
                init='latinhypercube',
                polish=True,
                maxfev=2000
            )
            
            if result.fun < best_score:
                # Reconstruct f from optimal parameters
                params = result.x
                n_p = len(params) // 3
                f_new = np.zeros(N)
                step = 0.5 / N
                
                for i in range(n_p):
                    pos = params[3*i]
                    width = params[3*i+1]
                    height = params[3*i+2]
                    
                    left = (pos - width/2 + 0.25) / step
                    right = (pos + width/2 + 0.25) / step
                    
                    idx_left = max(0, int(np.floor(left)))
                    idx_right = min(N, int(np.ceil(right)))
                    
                    if idx_right > idx_left:
                        f_new[idx_left:idx_right] += height
                
                new_score = compute_score(f_new)
                if new_score < best_score:
                    best_f = f_new
                    best_score = new_score
                    
        except Exception:
            pass
    
    # Local refinement using coordinate descent on the discrete grid
    if time.time() - start_time < TIME_LIMIT - 10:
        current_f = best_f.copy()
        current_score = best_score
        alpha = 0.05
        
        for step in range(200):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            conv = fft_autoconv(current_f)
            s = np.sum(current_f)
            c_max = np.max(conv)
            current_score = c_max / (s * s)
            
            # Find peak lags
            threshold = c_max * 0.9
            peak_indices = np.where(conv >= threshold)[0]
            
            # Sensitivity: d(conv[m])/d(f[j]) = f[m-j]
            sensitivity = np.zeros(N)
            for m in peak_indices:
                j_min = max(0, m - (N - 1))
                j_max = min(N - 1, m)
                for j in range(j_min, j_max + 1):
                    sensitivity[j] += current_f[m - j]
            
            # Normalize
            sens_mean = np.mean(sensitivity)
            sens_std = np.std(sensitivity) + 1e-12
            sens_norm = (sensitivity - sens_mean) / sens_std
            
            # Update
            f_new = current_f * (1.0 - alpha * sens_norm)
            f_new = np.maximum(f_new, 0.0)
            
            # Normalize to keep sum constant
            s_new = np.sum(f_new)
            if s_new < 1e-12:
                break
            f_new = f_new * (s / s_new)
            
            new_score = compute_score(f_new)
            if new_score < current_score:
                current_f = f_new
                current_score = new_score
                alpha = min(alpha * 1.05, 0.2)
            else:
                alpha *= 0.8
                if alpha < 1e-6:
                    break
        
        if current_score < best_score:
            best_f = current_f
            best_score = current_score
    
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
