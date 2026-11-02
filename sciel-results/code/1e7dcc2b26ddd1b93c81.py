import numpy as np
import time
from scipy.optimize import minimize

N = 600
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s

K = 32
M_STEP = N * K  # 19200
M_FFT = 1 << 15  # 32768

def _build_fine_grid(heights):
    """Build fine-sampled function from step heights."""
    return np.repeat(heights, K)

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n
    
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = _build_fine_grid(heights)
    
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * w
    
    if integral_f <= 0:
        return np.inf
    
    score = max_conv / (integral_f ** 2)
    return score

def _pulse_objective(params, n_pulses):
    """Objective for pulse-based representation.
    params: [amp1, pos1, width1, amp2, pos2, width2, ...]
    Each pulse is a rectangle of given amplitude, center position, and width.
    Position and width are in [0, 1] (fraction of the interval).
    """
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return 1e10
    
    heights = np.zeros(N, dtype=np.float64)
    for i in range(n_pulses):
        amp = params[3*i]
        pos = params[3*i + 1]  # center in [0,1]
        width = params[3*i + 2]  # width in [0,1]
        
        if amp < 0 or width <= 0:
            continue
        
        # Convert to interval indices
        center_idx = pos * N
        half_width_idx = width * N / 2.0
        
        start_idx = int(np.floor(center_idx - half_width_idx))
        end_idx = int(np.ceil(center_idx + half_width_idx))
        
        start_idx = max(0, start_idx)
        end_idx = min(N, end_idx)
        
        if start_idx < end_idx:
            heights[start_idx:end_idx] += amp
    
    if np.sum(heights) <= 0:
        return 1e10
    
    # Compute score
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = np.repeat(heights, K)
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * (0.5 / N)
    
    if integral_f <= 1e-10:
        return 1e10
    
    return max_conv / (integral_f ** 2)

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return [1.0] * N
    
    # Start with uniform (known to give score 1.0)
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Strategy: Pulse-based representation with Nelder-Mead optimization
    # Try different numbers of pulses
    rng = np.random.default_rng(42)
    
    for n_pulses in [1, 2, 3, 4, 5, 6, 7]:
        if time.time() - START_TIME > TIME_LIMIT - 30:
            break
        
        # Multiple random restarts for each n_pulses
        n_restarts = max(1, int((TIME_LIMIT - 30 - (time.time() - START_TIME)) / (n_pulses * 3)))
        n_restarts = min(n_restarts, 10)
        
        for restart in range(n_restarts):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            # Random initial parameters
            params = np.zeros(3 * n_pulses)
            for i in range(n_pulses):
                params[3*i] = rng.uniform(0.5, 2.0)  # amplitude
                params[3*i + 1] = rng.uniform(0.1, 0.9)  # position
                params[3*i + 2] = rng.uniform(0.05, 0.4)  # width
            
            # Bounds
            bounds = []
            for i in range(n_pulses):
                bounds.append((0.0, 5.0))  # amplitude
                bounds.append((0.0, 1.0))  # position
                bounds.append((0.01, 0.8))  # width
            
            # Nelder-Mead optimization
            try:
                result = minimize(
                    _pulse_objective,
                    params,
                    args=(n_pulses,),
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
                )
                
                if result.fun < best_score:
                    # Convert back to heights
                    final_params = result.x
                    heights = np.zeros(N, dtype=np.float64)
                    for i in range(n_pulses):
                        amp = final_params[3*i]
                        pos = final_params[3*i + 1]
                        width = final_params[3*i + 2]
                        
                        if amp < 0 or width <= 0:
                            continue
                        
                        center_idx = pos * N
                        half_width_idx = width * N / 2.0
                        start_idx = max(0, int(np.floor(center_idx - half_width_idx)))
                        end_idx = min(N, int(np.ceil(center_idx + half_width_idx)))
                        
                        if start_idx < end_idx:
                            heights[start_idx:end_idx] += amp
                    
                    best_score = result.fun
                    best_h = heights.copy()
            except Exception:
                pass
    
    # Final refinement: local search around best
    if time.time() - START_TIME < TIME_LIMIT - 5:
        rng2 = np.random.default_rng(123)
        for _ in range(30):
            if time.time() - START_TIME > TIME_LIMIT - 3:
                break
            perturbation = rng2.normal(0, 0.02, N)
            h_trial = best_h + perturbation
            h_trial = np.maximum(h_trial, 0.0)
            score = compute_score(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
    
    best_h = np.maximum(best_h, 0.0)
    return [max(0.0, float(v)) for v in best_h]

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
