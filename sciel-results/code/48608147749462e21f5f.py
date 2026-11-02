import time
import numpy as np
from scipy.optimize import minimize

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    # Use FFT for efficient convolution
    n_fft = 2 * N - 1
    H = np.fft.fft(h, n=n_fft)
    conv = np.real(np.fft.ifft(H * H))[:2*N-1]
    max_conv = np.max(conv)
    return max_conv / (s * s)

def smooth_max_surrogate(conv, beta=100.0):
    """
    Smooth approximation of max(conv) using log-sum-exp.
    Returns: (1/beta) * log(sum(exp(beta * conv)))
    """
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return (max_val + log_sum / beta)

def objective(params, beta=100.0):
    """
    Objective function for optimization.
    params: flattened array of h values (non-negative enforced via projection)
    """
    h = np.maximum(params, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return 1e10
    n_fft = 2 * N - 1
    H = np.fft.fft(h, n=n_fft)
    conv = np.real(np.fft.ifft(H * H))[:2*N-1]
    smoothed_peak = smooth_max_surrogate(conv, beta)
    return smoothed_peak / (s * s)

def objective_grad(params, beta=100.0):
    """
    Gradient of the objective function using FFT for efficiency.
    """
    h = np.maximum(params, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return np.zeros_like(params)
    
    n_fft = 2 * N - 1
    H = np.fft.fft(h, n=n_fft)
    conv = np.real(np.fft.ifft(H * H))[:2*N-1]
    
    # Compute softmax weights for the max surrogate
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    # Compute gradient using FFT
    # grad_h = 2 * correlate(weights, h)
    W = np.fft.fft(weights, n=n_fft)
    
    # Correlation via FFT
    grad_h = 2.0 * np.real(np.fft.ifft(W * np.conj(H)))[:N]
    
    # Gradient of objective: f = smooth_max / s^2
    smooth_max_val = smooth_max_surrogate(conv, beta)
    grad_obj = grad_h / (s * s) - 2 * smooth_max_val / (s * s * s)
    
    # Zero out gradient where h is at boundary (h=0) to respect constraint
    grad_obj[h <= 1e-12] = 0.0
    
    return grad_obj

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_h = np.ones(N) / N
    best_score = compute_score(best_h)

    np.random.seed(42)
    
    # Multiple initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N) / N)
    
    # 2. Gaussian
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-x**2 / (2 * 0.5**2))
    gauss /= np.sum(gauss)
    initializations.append(gauss)
    
    # 3. Linear ramp
    lin = np.linspace(0.1, 1.0, N)
    lin /= np.sum(lin)
    initializations.append(lin)
    
    # 4. Sine
    sine = np.sin(np.pi * (x + 1) / 2) ** 2
    sine /= np.sum(sine)
    initializations.append(sine)
    
    # 5. Step function (left half vs right half)
    step = np.zeros(N)
    step[:N//2] = 1.0
    step /= np.sum(step)
    initializations.append(step)
    
    # 6. Random
    rand_h = np.random.rand(N)
    rand_h /= np.sum(rand_h)
    initializations.append(rand_h)
    
    # 7. Triangular
    tri = np.abs(np.linspace(-1, 1, N))
    tri = 1.0 - tri
    tri /= np.sum(tri)
    initializations.append(tri)
    
    # 8. Two-block: high-low-high pattern
    two_block = np.zeros(N)
    two_block[:N//3] = 1.0
    two_block[2*N//3:] = 1.0
    two_block /= np.sum(two_block)
    initializations.append(two_block)
    
    # 9. Cosine bump
    cos_bump = np.cos(np.pi * x) ** 2
    cos_bump = np.maximum(cos_bump, 0)
    cos_bump /= np.sum(cos_bump)
    initializations.append(cos_bump)

    beta = 100.0
    bounds = [(0, None)] * N
    
    for h0 in initializations:
        if time.time() > deadline - 25:
            break
            
        # Optimize with adaptive beta
        try:
            # Start with lower beta for smoother landscape
            res = minimize(
                objective, 
                h0, 
                method='L-BFGS-B', 
                jac=objective_grad, 
                args=(50.0,),
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            
            # Refine with higher beta
            if time.time() < deadline - 20:
                res2 = minimize(
                    objective, 
                    h_opt, 
                    method='L-BFGS-B', 
                    jac=objective_grad, 
                    args=(100.0,),
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                h_opt = res2.x
                h_opt = np.maximum(h_opt, 0)
            
            score_opt = compute_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # Final polish with tighter tolerance and higher beta
    if time.time() < deadline - 10:
        try:
            res = minimize(
                objective, 
                best_h, 
                method='L-BFGS-B', 
                jac=objective_grad, 
                args=(200.0,),
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Coordinate descent refinement
    if time.time() < deadline - 5:
        h_cd = best_h.copy()
        for iteration in range(50):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(0, N, 3):  # Step by 3 for speed
                if time.time() > deadline - 3:
                    break
                old_val = h_cd[i]
                # Try small perturbations
                for delta in [-0.01, 0.01]:
                    trial = h_cd.copy()
                    trial[i] = max(0, old_val + delta)
                    trial /= np.sum(trial)
                    trial_score = compute_score(trial)
                    if trial_score < best_score - 1e-10:
                        h_cd = trial
                        best_score = trial_score
                        best_h = trial.copy()
                        improved = True
                        break
                if not improved and old_val > 0:
                    # Try setting to zero
                    trial = h_cd.copy()
                    trial[i] = 0
                    if np.sum(trial) > 0:
                        trial /= np.sum(trial)
                        trial_score = compute_score(trial)
                        if trial_score < best_score - 1e-10:
                            h_cd = trial
                            best_score = trial_score
                            best_h = trial.copy()
                            improved = True

    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
# EVOLVE-BLOCK-END
