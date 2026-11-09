import numpy as np
import time
from scipy.optimize import minimize

N = 100

def calculate_max_overlap(h):
    """Calculate maximum linear overlap of h with shifted 1-h."""
    max_ov = 0.0
    for k in range(N):
        if k == 0:
            ov = np.sum(h * (1 - h))
        else:
            ov = np.sum(h[k:] * (1 - h[:-k]))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def soft_max_overlaps(h, beta=50.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    overlaps = np.zeros(N)
    for k in range(N):
        if k == 0:
            overlaps[k] = np.sum(h * (1 - h))
        else:
            overlaps[k] = np.sum(h[k:] * (1 - h[:-k]))
    # Log-sum-exp for numerical stability
    m = np.max(overlaps)
    log_sum = m + np.log(np.sum(np.exp(beta * (overlaps - m))))
    return log_sum / beta

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Base solution
    h_base = np.full(N, 0.5)
    base_score = calculate_max_overlap(h_base)
    
    best_h = h_base
    best_score = base_score
    
    # Try multiple basis functions: Gaussian, sinusoidal, cosine taper
    def make_haussian(params):
        sigma, amp = params
        x = np.arange(N, dtype=float)
        center = (N - 1) / 2.0
        eps = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        eps -= np.mean(eps)
        eps_max = np.max(np.abs(eps))
        if eps_max < 1e-12:
            return None
        eps = eps / eps_max
        h = 0.5 + amp * eps
        if np.min(h) < -1e-9 or np.max(h) > 1 + 1e-9:
            return None
        # Correct sum
        diff = N / 2.0 - np.sum(h)
        if abs(diff) > 1e-12:
            idx = np.argmin(np.abs(h - 0.5))
            h[idx] += diff
            if np.min(h) < -1e-9 or np.max(h) > 1 + 1e-9:
                return None
        return h
    
    def make_sinusoidal(params):
        freq, amp, phase = params
        x = np.arange(N, dtype=float)
        eps = amp * np.sin(2 * np.pi * freq * x / N + phase)
        eps -= np.mean(eps)
        eps_max = np.max(np.abs(eps))
        if eps_max < 1e-12:
            return None
        eps = eps / eps_max
        h = 0.5 + amp * eps
        if np.min(h) < -1e-9 or np.max(h) > 1 + 1e-9:
            return None
        diff = N / 2.0 - np.sum(h)
        if abs(diff) > 1e-12:
            idx = np.argmin(np.abs(h - 0.5))
            h[idx] += diff
            if np.min(h) < -1e-9 or np.max(h) > 1 + 1e-9:
                return None
        return h
    
    def make_cosine_taper(params):
        # Cosine taper: high in middle, low at edges
        width, amp = params
        x = np.arange(N, dtype=float)
        center = (N - 1) / 2.0
        # Cosine bump with width parameter
        t = np.abs(x - center) / (width + 1e-9)
        eps = np.where(t < 1.0, np.cos(np.pi * t / 2.0), 0.0)
        eps -= np.mean(eps)
        eps_max = np.max(np.abs(eps))
        if eps_max < 1e-12:
            return None
        eps = eps / eps_max
        h = 0.5 + amp * eps
        if np.min(h) < -1e-9 or np.max(h) > 1 + 1e-9:
            return None
        diff = N / 2.0 - np.sum(h)
        if abs(diff) > 1e-12:
            idx = np.argmin(np.abs(h - 0.5))
            h[idx] += diff
            if np.min(h) < -1e-9 or np.max(h) > 1 + 1e-9:
                return None
        return h
    
    # Grid search for each basis type
    candidates = []
    
    # Gaussian search
    for sigma in np.linspace(5, 30, 12):
        for amp in np.linspace(0.0, 0.48, 10):
            h = make_haussian([sigma, amp])
            if h is not None:
                score = calculate_max_overlap(h)
                candidates.append((score, h, 'gaussian', [sigma, amp]))
    
    # Sinusoidal search
    for freq in np.linspace(0.5, 10, 20):
        for amp in np.linspace(0.0, 0.45, 8):
            for phase in [0, np.pi/2, np.pi]:
                h = make_sinusoidal([freq, amp, phase])
                if h is not None:
                    score = calculate_max_overlap(h)
                    candidates.append((score, h, 'sinusoidal', [freq, amp, phase]))
    
    # Cosine taper search
    for width in np.linspace(5, 40, 15):
        for amp in np.linspace(0.0, 0.48, 10):
            h = make_cosine_taper([width, amp])
            if h is not None:
                score = calculate_max_overlap(h)
                candidates.append((score, h, 'cosine', [width, amp]))
    
    if not candidates:
        return best_h
    
    # Sort by score and take top few for local optimization
    candidates.sort(key=lambda x: x[0])
    
    # Local optimization on best candidates
    for score, h, basis, params in candidates[:5]:
        if time.time() > deadline - 30:
            break
        
        try:
            if basis == 'gaussian':
                def obj_gaussian(p):
                    h_c = make_haussian(p)
                    if h_c is None:
                        return 100.0
                    return calculate_max_overlap(h_c)
                
                bounds = [(2, 50), (0, 0.49)]
                res = minimize(obj_gaussian, params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
                if res.success:
                    h_opt = make_haussian(res.x)
                    if h_opt is not None:
                        s_opt = calculate_max_overlap(h_opt)
                        if s_opt < best_score:
                            best_score = s_opt
                            best_h = h_opt.copy()
            
            elif basis == 'sinusoidal':
                def obj_sinusoidal(p):
                    h_c = make_sinusoidal(p)
                    if h_c is None:
                        return 100.0
                    return calculate_max_overlap(h_c)
                
                bounds = [(0.5, 15), (0, 0.48), (0, 2*np.pi)]
                res = minimize(obj_sinusoidal, params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
                if res.success:
                    h_opt = make_sinusoidal(res.x)
                    if h_opt is not None:
                        s_opt = calculate_max_overlap(h_opt)
                        if s_opt < best_score:
                            best_score = s_opt
                            best_h = h_opt.copy()
            
            elif basis == 'cosine':
                def obj_cosine(p):
                    h_c = make_cosine_taper(p)
                    if h_c is None:
                        return 100.0
                    return calculate_max_overlap(h_c)
                
                bounds = [(3, 50), (0, 0.49)]
                res = minimize(obj_cosine, params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
                if res.success:
                    h_opt = make_cosine_taper(res.x)
                    if h_opt is not None:
                        s_opt = calculate_max_overlap(h_opt)
                        if s_opt < best_score:
                            best_score = s_opt
                            best_h = h_opt.copy()
                            
        except Exception:
            pass
    
    return best_h
