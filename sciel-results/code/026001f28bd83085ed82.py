import numpy as np
import time
from scipy.optimize import minimize

N = 600
START_TIME = time.time()
TIME_LIMIT = 145

K = 32
M_STEP = N * K
M_FFT = 1 << 15

def compute_score(heights):
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    n = len(heights)
    w = 0.5 / n
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = np.repeat(heights, K)
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * w
    if integral_f <= 0:
        return np.inf
    return max_conv / (integral_f ** 2)

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return [1.0] * N
    
    best_h = np.ones(N, dtype=np.float64)
    best_score = compute_score(best_h)
    
    # Try two-rectangle designs
    x = np.linspace(0, 1, N)
    for gap_frac in np.linspace(0.05, 0.4, 15):
        for height in [0.5, 1.0, 1.5, 2.0]:
            h = np.zeros(N)
            gap = int(gap_frac * N)
            left_end = (N - gap) // 2
            right_start = left_end + gap
            if left_end > 0:
                h[:left_end] = height
            if right_start < N:
                h[right_start:] = height
            score = compute_score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Use scipy to optimize
    def objective(h):
        return compute_score(h)
    
    def objective_grad(h):
        h = np.maximum(h, 0.0)
        n = len(h)
        w = 0.5 / n
        f_fine = np.zeros(M_FFT, dtype=np.float64)
        f_fine[:M_STEP] = np.repeat(h, K)
        F = np.fft.rfft(f_fine)
        conv = np.fft.irfft(F * F, n=M_FFT)
        max_conv = np.max(conv)
        integral_f = np.sum(h) * w
        if integral_f <= 1e-10:
            return 1e10, np.zeros_like(h)
        score = max_conv / (integral_f ** 2)
        peak_idx = np.argmax(conv)
        G = np.roll(f_fine, peak_idx)
        G_step = G[:M_STEP]
        G_reshaped = G_step.reshape(N, K)
        grad_conv = np.sum(G_reshaped, axis=1)
        dI_dh = np.full(n, w)
        grad_score = grad_conv / (integral_f ** 2) - 2 * max_conv * dI_dh / (integral_f ** 3)
        return score, grad_score
    
    # Start from best found
    h0 = best_h.copy()
    
    # Use L-BFGS-B with bounds
    bounds = [(0.0, None)] * N
    
    # First, try a few random starts
    rng = np.random.default_rng(42)
    candidates = [best_h.copy()]
    for _ in range(5):
        h_rand = np.abs(rng.normal(1.0, 0.3, N))
        h_rand = np.maximum(h_rand, 0.0)
        candidates.append(h_rand)
    
    for h_start in candidates:
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        try:
            result = minimize(objective, h_start, method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
            if result.fun < best_score:
                best_score = result.fun
                best_h = np.maximum(result.x, 0.0)
        except Exception:
            pass
    
    # Final local refinement
    if time.time() - START_TIME < TIME_LIMIT - 5:
        for _ in range(20):
            if time.time() - START_TIME > TIME_LIMIT - 3:
                break
            perturbation = rng.normal(0, 0.005, N)
            h_trial = np.maximum(best_h + perturbation, 0.0)
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
