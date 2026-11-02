import time
import numpy as np
from scipy.optimize import minimize

N = 600

def compute_score(h):
    s = np.sum(h)
    if s == 0:
        return 0.0
    n_fft = 2 * N - 1
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    max_conv = np.max(conv)
    return max_conv / (s * s)

def smooth_max_surrogate(conv, beta=100.0):
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return (max_val + log_sum / beta)

def objective_and_grad(params, beta=100.0):
    h = np.maximum(params, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return 1e10, np.zeros_like(params)
    
    n_fft = 2 * N - 1
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    W = np.fft.rfft(weights, n=n_fft)
    grad_h = 2.0 * np.fft.irfft(W * np.conj(H), n=n_fft)[:N]
    
    smooth_max_val = smooth_max_surrogate(conv, beta)
    obj_val = smooth_max_val / (s * s)
    grad_obj = grad_h / (s * s) - 2 * smooth_max_val / (s * s * s)
    
    return obj_val, grad_obj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)
    
    np.random.seed(42)
    
    initializations = []
    
    # Uniform
    initializations.append(np.ones(N) / N)
    
    # Gaussian (various widths)
    x = np.linspace(-1, 1, N)
    for sigma in [0.3, 0.5, 0.7]:
        gauss = np.exp(-x**2 / (2 * sigma**2))
        gauss /= np.sum(gauss)
        initializations.append(gauss.copy())
    
    # Cosine window
    cos_win = np.cos(np.pi * x / 2)**2
    cos_win /= np.sum(cos_win)
    initializations.append(cos_win.copy())
    
    # Two-block (flat middle, zero ends)
    two_block = np.zeros(N)
    two_block[N//4 : 3*N//4] = 1.0
    two_block /= np.sum(two_block)
    initializations.append(two_block.copy())
    
    # Sine raised to power
    for p in [1, 2, 4]:
        sine = np.sin(np.pi * (x + 1) / 2) ** p
        sine /= np.sum(sine)
        initializations.append(sine.copy())
    
    # Triangular
    tri = 1.0 - np.abs(x)
    tri = np.maximum(tri, 0)
    tri /= np.sum(tri)
    initializations.append(tri.copy())
    
    # Random with smoothing
    rand_h = np.random.rand(N)
    # Apply moving average for smoothing
    kernel = np.ones(5) / 5
    rand_h = np.convolve(rand_h, kernel, mode='same')
    rand_h = np.maximum(rand_h, 0)
    rand_h /= np.sum(rand_h)
    initializations.append(rand_h.copy())
    
    # Plateau (flat top, steep edges)
    plateau = np.zeros(N)
    plateau[N//6 : 5*N//6] = 1.0
    # Smooth the edges
    edge = np.ones(10) / 10
    plateau = np.convolve(plateau, edge, mode='same')
    plateau = np.maximum(plateau, 0)
    plateau /= np.sum(plateau)
    initializations.append(plateau.copy())
    
    # Beta-like distribution (asymmetric)
    beta_dist = np.exp(-x * 0.5) * np.exp(-x**2 / (2 * 0.4**2))
    beta_dist = np.maximum(beta_dist, 0)
    beta_dist /= np.sum(beta_dist)
    initializations.append(beta_dist.copy())
    
    # W-shaped
    w_shape = np.abs(x - 0.3) + np.abs(x + 0.3)
    w_shape = np.exp(-w_shape * 3)
    w_shape /= np.sum(w_shape)
    initializations.append(w_shape.copy())
    
    beta_schedule = [50.0, 100.0, 200.0, 500.0]
    
    for h0 in initializations:
        if time.time() > deadline - 25:
            break
            
        current_h = h0.copy()
        current_score = compute_score(current_h)
        
        for beta in beta_schedule:
            if time.time() > deadline - 20:
                break
            
            try:
                res = minimize(
                    objective_and_grad, 
                    current_h, 
                    method='L-BFGS-B', 
                    jac=True,
                    args=(beta,),
                    bounds=[(0, None)] * N,
                    options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-9}
                )
                
                h_opt = np.maximum(res.x, 0)
                if np.sum(h_opt) < 1e-10:
                    continue
                score_opt = compute_score(h_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
                    current_h = h_opt.copy()
                elif score_opt < current_score + 1e-6:
                    current_h = h_opt.copy()
                    current_score = score_opt
                    
            except Exception:
                pass
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        try:
            res = minimize(
                objective_and_grad, 
                best_h, 
                method='L-BFGS-B', 
                jac=True,
                args=(1000.0,),
                bounds=[(0, None)] * N,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            h_opt = np.maximum(res.x, 0)
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Coordinate descent refinement on the best solution
    if time.time() < deadline - 5:
        h_cd = best_h.copy()
        for iteration in range(30):
            if time.time() > deadline - 3:
                break
            for i in range(0, N, 2):
                if time.time() > deadline - 3:
                    break
                old_val = h_cd[i]
                for delta in [-0.005, 0.005, -0.002, 0.002]:
                    trial = h_cd.copy()
                    trial[i] = max(0, old_val + delta)
                    if np.sum(trial) < 1e-10:
                        continue
                    trial_score = compute_score(trial)
                    if trial_score < best_score - 1e-12:
                        h_cd = trial
                        best_score = trial_score
                        best_h = trial.copy()
                        break
    
    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
