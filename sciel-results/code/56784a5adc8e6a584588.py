import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0

# Precompute fine grid mapping for fast ratio
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    
    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX
    max_conv = np.max(conv)
    l2_sq = np.sum(f_fine ** 2) * DX
    if l2_sq < 1e-15:
        return float('inf')
    return max_conv / l2_sq

def param_to_h(p):
    """
    Map parameter vector p to step heights h.
    p[0] : global scale (log)
    p[1:] : Fourier coefficients (sin/cos pairs up to some order)
    """
    n_fourier = len(p) - 1
    # Number of Fourier pairs
    n_pairs = n_fourier // 2
    n_sin = n_fourier % 2
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    
    # Base constant term
    h = np.ones(N) * np.exp(p[0])
    
    # Add Fourier terms
    idx = 1
    for k in range(1, n_pairs + 1):
        a = p[idx]
        b = p[idx + 1]
        idx += 2
        h += a * np.cos(2 * np.pi * k * x / 0.5) + b * np.sin(2 * np.pi * k * x / 0.5)
    
    if n_sin:
        a = p[idx]
        h += a * np.cos(2 * np.pi * (n_pairs + 1) * x / 0.5)
    
    # Clip to nonnegative
    h = np.clip(h, 0.0, None)
    
    # Ensure not all zero
    if h.sum() < 1e-15:
        h += 1e-10
    
    return h

def objective(p):
    h = param_to_h(p)
    return ratio_from_h(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Precompute some initial candidates
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)
    
    # Try a few structured initial points
    np.random.seed(42)
    
    # Initial parameter guesses
    initial_guesses = []
    
    # 1. Constant
    initial_guesses.append([np.log(1.0)])
    
    # 2. With low-frequency Fourier terms
    for n_pairs in [2, 3, 4, 5, 6, 8, 10]:
        # Random small coefficients
        p = [np.log(np.random.uniform(0.5, 2.0))]
        for _ in range(n_pairs):
            p.append(np.random.uniform(-0.5, 0.5))
            p.append(np.random.uniform(-0.5, 0.5))
        initial_guesses.append(p)
    
    # 3. Asymmetric bias (linear term approximated by Fourier)
    # Linear ramp: x from -0.25 to 0.25 -> use cos/sin to approximate
    for n_pairs in [3, 5, 8]:
        p = [np.log(1.0)]
        # Emphasize first few harmonics with alternating signs for asymmetry
        for k in range(1, n_pairs + 1):
            a = np.random.uniform(-0.3, 0.3)
            b = np.random.uniform(-0.3, 0.3)
            p.append(a)
            p.append(b)
        initial_guesses.append(p)
    
    # 4. Gaussian-like bump (approximated by Fourier)
    for n_pairs in [5, 8, 12]:
        p = [np.log(0.8)]
        # Gaussian has decaying Fourier coefficients
        for k in range(1, n_pairs + 1):
            decay = np.exp(-0.1 * k)
            a = np.random.uniform(-0.4, 0.4) * decay
            b = np.random.uniform(-0.4, 0.4) * decay
            p.append(a)
            p.append(b)
        initial_guesses.append(p)
    
    # Evaluate all initial guesses
    for p in initial_guesses:
        if time.time() > deadline:
            break
        h = param_to_h(p)
        val = ratio_from_h(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Now do local optimization from the best initial point and a few others
    # Sort initial guesses by their objective value
    scored_guesses = []
    for p in initial_guesses:
        h = param_to_h(p)
        val = ratio_from_h(h)
        scored_guesses.append((val, p))
    
    scored_guesses.sort(key=lambda x: x[0])
    
    # Take top few for optimization
    n_optimize = min(8, len(scored_guesses))
    
    for i in range(n_optimize):
        if time.time() > deadline - 10.0:
            break
        
        val_init, p_init = scored_guesses[i]
        
        # Try different optimization methods
        methods_to_try = ['L-BFGS-B', 'SLSQP', 'Nelder-Mead']
        
        for method in methods_to_try:
            if time.time() > deadline - 5.0:
                break
            
            try:
                if method == 'Nelder-Mead':
                    res = minimize(
                        objective, 
                        p_init, 
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                    )
                else:
                    res = minimize(
                        objective, 
                        p_init, 
                        method=method,
                        bounds=[(-5, 5)] + [(-2, 2)] * (len(p_init) - 1),
                        options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                
                if res.success or res.fun < best_val:
                    h_opt = param_to_h(res.x)
                    val_opt = ratio_from_h(h_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
                        
            except Exception:
                continue
    
    # Final refinement: try a few more random restarts if time allows
    if time.time() < deadline - 20.0:
        n_restarts = 5
        for i in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Random parameter vector with moderate complexity
            n_pairs = np.random.randint(4, 15)
            p = [np.log(np.random.uniform(0.3, 2.0))]
            for _ in range(n_pairs):
                p.append(np.random.uniform(-0.8, 0.8))
                p.append(np.random.uniform(-0.8, 0.8))
            
            try:
                res = minimize(
                    objective, 
                    p, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] + [(-3, 3)] * (len(p) - 1),
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                
                if res.fun < best_val:
                    h_opt = param_to_h(res.x)
                    val_opt = ratio_from_h(h_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
                        
            except Exception:
                continue
    
    return list(best_h)
