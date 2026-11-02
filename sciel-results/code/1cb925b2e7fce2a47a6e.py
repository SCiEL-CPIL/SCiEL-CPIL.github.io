import numpy as np
import time
from scipy import optimize

N = 600
_CONV_LEN = 2 * N - 1
_FFT_SIZE = 1

def _fft_convolve(h):
    n = _FFT_SIZE
    f = np.fft.rfft(h, n=n)
    g = np.fft.irfft(f * f, n=n)[:_CONV_LEN]
    return g

def compute_score(h):
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    max_conv = np.max(conv)
    return max_conv / (s * s)

def smooth_score(h, beta=100.0):
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    # Log-sum-exp for max
    # max(conv) approx (1/beta) * log(sum(exp(beta * conv)))
    # To avoid overflow, subtract max
    max_c = np.max(conv)
    log_sum_exp = max_c + (1.0/beta) * np.log(np.sum(np.exp(beta * (conv - max_c))))
    return log_sum_exp / (s * s)

def smooth_score_grad(h, beta=100.0):
    # Gradient of smooth_score wrt h
    # This is complex to derive analytically with FFT.
    # We can use numerical gradient if needed, but L-BFGS-B can handle numerical gradients.
    # However, numerical gradient for N=600 is 600 evaluations of smooth_score per step.
    # That might be too slow.
    # Let's try to use a lower beta or just use the raw max and hope L-BFGS-B handles the non-differentiability?
    # Or use SLSQP with numerical gradient?
    # Actually, the "max" function is non-smooth, but the log-sum-exp is smooth.
    # We can just use smooth_score as the objective and let L-BFGS-B compute numerical gradients.
    # With N=600, one gradient step takes ~600 FFTs.
    # 100 steps * 600 FFTs = 60000 FFTs.
    # Each FFT is fast. 60000 * (600 log 600) ops is manageable.
    
    pass

def construct():
    global _FFT_SIZE
    start_time = time.time()
    deadline = start_time + 110.0

    # Set FFT size
    _FFT_SIZE = 1
    while _FFT_SIZE < _CONV_LEN:
        _FFT_SIZE *= 2

    w = 0.5 / N
    x = np.linspace(-0.25 + w/2, 0.25 - w/2, N)

    # Candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N) / N)
    
    # 2. Gaussian
    sigma = 0.1
    h_gauss = np.exp(-x**2 / (2 * sigma**2))
    h_gauss /= np.sum(h_gauss)
    candidates.append(h_gauss)
    
    # 3. U-shaped (x^2)
    h_u = x**2
    h_u /= np.sum(h_u)
    candidates.append(h_u)
    
    # 4. Cosine Edge (sin^2)
    h_sin = np.sin(np.pi * x / 0.25)**2
    h_sin /= np.sum(h_sin)
    candidates.append(h_sin)

    best_h = None
    best_score = float('inf')

    for h_init in candidates:
        # Optimize using L-BFGS-B
        # We optimize v, h = softmax(v) ?
        # Or just optimize h directly with bounds [0, inf) and normalize inside objective?
        # If we normalize inside objective, the gradient is messed up by the normalization.
        # Better: h = exp(v). Objective: smooth_score(h) / sum(h)^2 ?
        # No, score is scale invariant.
        # Let's optimize h directly with constraint sum(h)=1?
        # L-BFGS-B doesn't handle equality constraints well.
        # SLSQP does.
        
        # Let's use SLSQP with constraint sum(h)=1 and h>=0.
        # But SLSQP is slow for N=600.
        
        # Alternative: Optimize in log-space v, h = exp(v).
        # Objective: smooth_score(exp(v)).
        # The scale invariance means the objective is constant along the ray h -> c*h.
        # This creates a flat valley in the loss landscape.
        # To fix this, we can add a small penalty for sum(h) != 1.
        # Or just fix one variable?
        
        # Let's try optimizing h directly with L-BFGS-B and bounds [0, 1].
        # And add a penalty term: penalty * (sum(h) - 1)^2.
        # This breaks scale invariance slightly, allowing the optimizer to settle on a scale.
        
        def objective(h):
            s = np.sum(h)
            if s == 0: return 1.0
            score = smooth_score(h, beta=50.0)
            # Penalty to keep sum near 1
            pen = 100.0 * (s - 1.0)**2
            return score + pen
        
        # Initial guess
        h0 = h_init.copy()
        
        # Bounds
        bounds = [(0, 1.0)] * N
        
        try:
            res = optimize.minimize(objective, h0, method='L-BFGS-B', bounds=bounds, 
                                    options={'maxiter': 20, 'ftol': 1e-6})
            h_opt = res.x
            # Normalize
            h_opt /= np.sum(h_opt)
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
        except Exception:
            pass

        if time.time() > deadline - 10:
            break

    # If no optimization succeeded, use the best initial candidate
    if best_h is None:
        for h in candidates:
            s = compute_score(h)
            if s < best_score:
                best_score = s
                best_h = h

    return best_h.tolist()
