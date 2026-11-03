"""
Strategy: Hybrid Spectral and Direct Optimization
Combines the DCT spectral basis approach (current) with direct coordinate optimization and
analytical gradient refinement (inspiration).

Key ideas:
1. Use DCT basis to find good global structures quickly (low-dimensional optimization)
2. Then refine in full 600D space using L-BFGS-B with analytical gradient
3. Use smooth max surrogate (log-sum-exp) for differentiable objective
4. Multiple initializations from both spectral and hand-crafted shapes
5. Final polish with tighter tolerances
"""
import time
import numpy as np
from scipy import optimize
from scipy.fft import dct, idct

N = 600
M = 20  # Number of DCT modes

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    size = 2 * N
    f = np.fft.rfft(h, size)
    conv = np.fft.irfft(f * f, size)[:2*N-1]
    max_conv = np.max(conv)
    return max_conv / (s * s)

def _dct_to_h(coeffs):
    """
    Convert DCT coefficients to step function heights.
    coeffs: array of length M.
    Returns array of length N.
    """
    full_coeffs = np.zeros(N)
    full_coeffs[:M] = coeffs
    h = idct(full_coeffs, type=2, norm='ortho')
    h = np.maximum(h, 0)
    return h

def smooth_max_surrogate(conv, beta=100.0):
    """
    Smooth approximation of max(conv) using log-sum-exp.
    """
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta

def objective_full(h, beta=100.0):
    """
    Objective function for full 600D optimization.
    h: array of length N (non-negative enforced via projection).
    """
    h = np.maximum(h, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return 1e10
    size = 2 * N
    f = np.fft.rfft(h, size)
    conv = np.fft.irfft(f * f, size)[:2*N-1]
    smoothed_peak = smooth_max_surrogate(conv, beta)
    return smoothed_peak / (s * s)

def objective_grad_full(h, beta=100.0):
    """
    Gradient of the objective function using FFT for efficiency.
    """
    h = np.maximum(h, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return np.zeros_like(h)
    
    size = 2 * N
    f = np.fft.rfft(h, size)
    conv = np.fft.irfft(f * f, size)[:2*N-1]
    
    # Compute softmax weights for the max surrogate
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    # Compute gradient using FFT
    # grad_h = 2 * correlate(weights, h)
    W = np.fft.rfft(weights, size)
    grad_h = 2.0 * np.fft.irfft(W * f, size)[:N]
    
    # Gradient of objective: f = smooth_max / s^2
    smooth_max_val = smooth_max_surrogate(conv, beta)
    grad_obj = grad_h / (s * s) - 2 * smooth_max_val / (s * s * s)
    
    # Zero out gradient where h is at boundary (h=0) to respect constraint
    grad_obj[h <= 1e-12] = 0.0
    
    return grad_obj

def objective_dct(ac_coeffs, fixed_dc, beta=10.0):
    """
    Objective function for DCT coefficient optimization.
    ac_coeffs: array of length M-1 (AC components).
    fixed_dc: DC component (fixed).
    """
    coeffs = np.zeros(M)
    coeffs[0] = fixed_dc
    coeffs[1:] = ac_coeffs
    
    h = _dct_to_h(coeffs)
    s = np.sum(h)
    if s < 1e-6:
        return 1e10
        
    size = 2 * N
    f = np.fft.rfft(h, size)
    conv = np.fft.irfft(f * f, size)[:2*N-1]
    
    c_max = np.max(conv)
    if c_max < 0:
        c_max = 0
    conv_shifted = conv - c_max
    lse = c_max + beta * np.log(np.sum(np.exp(conv_shifted / beta)))
    
    return lse / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def check_time():
        return time.time() > deadline

    # Initialize best solution with uniform
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)
    
    # Phase 1: DCT spectral optimization to find good global structure
    h_uniform = np.ones(N) / N
    coeffs_uniform = dct(h_uniform, type=2, norm='ortho')
    fixed_dc = coeffs_uniform[0]
    init_ac = np.zeros(M - 1)
    
    # Optimize DCT coefficients with annealing beta
    betas_dct = [50.0, 20.0, 10.0, 5.0]
    current_ac = init_ac.copy()
    
    for beta in betas_dct:
        if check_time():
            break
            
        def obj_dct(x, b=beta):
            if check_time():
                return 1e10
            return objective_dct(x, fixed_dc, b)
        
        try:
            res = optimize.minimize(
                obj_dct,
                current_ac,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            current_ac = res.x
        except Exception:
            pass
        
        # Evaluate actual score with the found coeffs
        coeffs = np.zeros(M)
        coeffs[0] = fixed_dc
        coeffs[1:] = current_ac
        h_cand = _dct_to_h(coeffs)
        h_cand = h_cand / np.sum(h_cand)
        
        if np.sum(h_cand) > 1e-6:
            score_cand = compute_score(h_cand)
            if score_cand < best_score:
                best_score = score_cand
                best_h = h_cand.copy()

    # Phase 2: Full 600D refinement from multiple initializations
    np.random.seed(42)
    
    # Build list of initializations
    initializations = []
    
    # 1. Best from DCT phase
    initializations.append(best_h.copy())
    
    # 2. Uniform
    initializations.append(np.ones(N) / N)
    
    # 3. Gaussian
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-x**2 / (2 * 0.5**2))
    gauss = np.maximum(gauss, 0)
    gauss /= np.sum(gauss)
    initializations.append(gauss)
    
    # 4. Sine
    sine = np.sin(np.pi * (x + 1) / 2) ** 2
    sine /= np.sum(sine)
    initializations.append(sine)
    
    # 5. Triangular
    tri = 1.0 - np.abs(x)
    tri = np.maximum(tri, 0)
    tri /= np.sum(tri)
    initializations.append(tri)
    
    # 6. Two-block
    two_block = np.zeros(N)
    two_block[:N//3] = 1.0
    two_block[2*N//3:] = 1.0
    two_block /= np.sum(two_block)
    initializations.append(two_block)
    
    # 7. Cosine bump
    cos_bump = np.cos(np.pi * x) ** 2
    cos_bump = np.maximum(cos_bump, 0)
    cos_bump /= np.sum(cos_bump)
    initializations.append(cos_bump)
    
    # 8. Random
    rand_h = np.random.rand(N)
    rand_h /= np.sum(rand_h)
    initializations.append(rand_h)
    
    bounds = [(0, None)] * N
    
    for h0 in initializations:
        if check_time() or time.time() > deadline - 25:
            break
            
        try:
            # Start with lower beta for smoother landscape
            res = optimize.minimize(
                objective_full, 
                h0, 
                method='L-BFGS-B', 
                jac=objective_grad_full, 
                args=(50.0,),
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            h_opt = h_opt / np.sum(h_opt)
            
            # Refine with higher beta
            if time.time() < deadline - 20:
                res2 = optimize.minimize(
                    objective_full, 
                    h_opt, 
                    method='L-BFGS-B', 
                    jac=objective_grad_full, 
                    args=(100.0,),
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                h_opt = res2.x
                h_opt = np.maximum(h_opt, 0)
                h_opt = h_opt / np.sum(h_opt)
            
            score_opt = compute_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # Phase 3: Final polish with tighter tolerance and higher beta
    if time.time() < deadline - 10:
        try:
            res = optimize.minimize(
                objective_full, 
                best_h, 
                method='L-BFGS-B', 
                jac=objective_grad_full, 
                args=(200.0,),
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            h_opt = h_opt / np.sum(h_opt)
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 4: Coordinate descent refinement
    if time.time() < deadline - 5:
        h_cd = best_h.copy()
        step_size = 0.01
        
        for iteration in range(100):
            if time.time() > deadline - 3:
                break
            improved = False
            
            # Focus on indices near the peak of autoconvolution
            size = 2 * N
            f = np.fft.rfft(h_cd, size)
            conv = np.fft.irfft(f * f, size)[:2*N-1]
            peak_idx = np.argmax(conv)
            
            # Try adjusting coordinates, prioritizing those near the peak
            # and those with high values
            indices = np.argsort(-h_cd)
            
            for i in indices:
                if time.time() > deadline - 3:
                    break
                
                old_val = h_cd[i]
                
                # Try increasing
                trial = h_cd.copy()
                trial[i] = old_val + step_size
                trial /= np.sum(trial)
                trial_score = compute_score(trial)
                if trial_score < best_score - 1e-12:
                    h_cd = trial
                    best_score = trial_score
                    best_h = trial.copy()
                    improved = True
                    continue
                
                # Try decreasing
                trial = h_cd.copy()
                trial[i] = max(0, old_val - step_size)
                if np.sum(trial) < 1e-10:
                    continue
                trial /= np.sum(trial)
                trial_score = compute_score(trial)
                if trial_score < best_score - 1e-12:
                    h_cd = trial
                    best_score = trial_score
                    best_h = trial.copy()
                    improved = True
            
            if not improved:
                break

    # Phase 5: Final aggressive polish with very high beta and tight tolerances
    if time.time() < deadline - 2:
        try:
            res = optimize.minimize(
                objective_full, 
                best_h, 
                method='L-BFGS-B', 
                jac=objective_grad_full, 
                args=(500.0,),
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            h_opt = h_opt / np.sum(h_opt)
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    return best_h.tolist()
