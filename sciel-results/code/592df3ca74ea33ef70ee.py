import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
import sys

N = 600
TIME_LIMIT = 135  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def smooth_max(values, temperature=0.01):
    """Smooth approximation of the maximum."""
    m = np.max(values)
    return m + temperature * np.log(np.sum(np.exp((values - m) / temperature)))

def cosine_basis(k, N):
    """
    Generate cosine basis functions on N intervals.
    k is the frequency index (1, 2, ...).
    """
    x = np.linspace(0, N, N)  # 0 to N-1 essentially, or 0 to 1
    # Let's use 0 to 1 for standard cosine
    x = np.linspace(0, 1, N)
    return np.cos(np.pi * k * x)

def parametric_to_step(coeffs, base_height=1.0):
    """
    Construct a step function from cosine coefficients.
    f(x) = base_height + sum(c_k * cos(pi * k * x))
    Ensure non-negativity.
    """
    x = np.linspace(0, 1, N)
    f = np.full(N, base_height)
    for k, c in enumerate(coeffs, start=1):
        f += c * np.cos(np.pi * k * x)
    # Enforce non-negativity
    f = np.maximum(f, 0.0)
    return f

def objective(coeffs, base_height=1.0, temperature=0.01):
    """
    Objective function to minimize: smooth max of autoconvolution.
    coeffs: array of cosine coefficients.
    """
    f = parametric_to_step(coeffs, base_height)
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    conv = np.convolve(f, f, mode='full')
    # Normalize by mass squared
    conv_norm = conv / (s * s)
    return smooth_max(conv_norm, temperature)

def fft_autocorr(f):
    """Fast autocorrelation using FFT."""
    n = len(f)
    # Pad to next power of 2 for efficiency? 2*N-1 is 1199. 2048 is good.
    n_fft = 2048
    F = np.fft.rfft(f, n=n_fft)
    C = np.fft.irfft(F * np.conj(F), n=n_fft)
    # The first N elements correspond to lags 0 to N-1
    # The second half corresponds to negative lags
    # Since f is real, C is symmetric. We only need lags 0 to N-1.
    return C[:n]

def compute_sensitivity_fft(f, coeffs, base_height=1.0, temperature=0.01):
    """
    Compute gradient of the smooth max objective with respect to coeffs.
    This is complex. Let's rely on numerical gradients or a simpler approach.
    For now, we will use a local refinement strategy that doesn't require explicit gradients
    or use scipy's minimize with numerical gradients.
    """
    pass

def local_refinement(f, max_iters=100, time_limit=10.0):
    """
    Refine the step function f using coordinate descent or similar.
    We want to minimize the peak of the autoconvolution.
    """
    start_time = time.time()
    f = f.copy()
    current_score = compute_score(f)
    
    # We can try to reduce the bins that contribute most to the peak.
    # The peak is likely at lag 0.
    # conv[0] = sum(f^2).
    # To minimize sum(f^2) subject to sum(f)=constant, we want f to be constant.
    # But we are constrained to be non-negative.
    # If we allow the mass to vary, we can scale f.
    # The score is max(conv) / (sum(f))^2.
    # Scaling f by alpha scales conv by alpha^2 and sum by alpha.
    # Score is invariant to scaling!
    # So we can fix sum(f) = 1 without loss of generality.
    
    # Let's fix sum(f) = 1.
    s = np.sum(f)
    if s > 0:
        f = f / s
    current_score = compute_score(f)
    
    # Coordinate descent on f_i
    # Objective: max_k conv_k.
    # This is non-smooth.
    # We can use a smooth approximation or just try to reduce the max.
    
    # Strategy: 
    # 1. Identify the lag k* with the maximum convolution value.
    # 2. Try to reduce conv[k*] by adjusting f.
    # conv[k] = sum_i f[i] f[i+k].
    # This is a quadratic form.
    
    # A simpler heuristic: 
    # If the function is not flat, flatten it.
    # But we already know constant is the local min for sum(f^2).
    # The peak might be at a non-zero lag?
    # For non-negative functions, the peak of the autoconvolution is always at lag 0?
    # Let's check.
    # f = [1, 0, 1]. conv = [2, 0, 2, 0, 1]. Peak is 2 (at lag 0 and 2).
    # f = [1, 1, 0]. conv = [1, 2, 1, 1, 0]. Peak is 2 (at lag 1).
    # So the peak can be at non-zero lags.
    
    # To minimize the peak, we need to balance the convolution values.
    # This is a minimax problem.
    
    # Let's use scipy.optimize.minimize with a smooth max objective.
    # We need to define the objective in terms of f directly.
    # But f must be non-negative.
    # We can optimize log(f) or use a transformation.
    # Or just use bounds.
    
    # However, N=600 is large for scipy.optimize.
    # Let's try a few iterations of a custom gradient-free method.
    
    # We'll use a simple "peak shaving" approach.
    # 1. Compute conv.
    # 2. Find max lag k*.
    # 3. Try to reduce conv[k*] by slightly reducing f[i] where f[i] * f[i+k*] is large.
    # 4. Redistribute the mass to other bins.
    
    # This is getting complicated. Let's stick to the parametric approach which is more robust.
    # The parametric approach restricts the search space to smooth functions, which are likely to be good.
    
    return f

def polish_pass(f, time_limit=5.0):
    """
    Final polishing of the function.
    """
    start_time = time.time()
    f = f.copy()
    
    # Try to average out small variations
    # This might increase the peak, so be careful.
    
    # Let's just return f for now, as the parametric optimization should be sufficient.
    return f

def construct():
    start_time = time.time()
    
    # Strategy:
    # 1. Parametric optimization using cosine basis.
    # 2. The function is f(x) = 1 + sum(c_k cos(k pi x)).
    # 3. We want to minimize the smooth max of the normalized autoconvolution.
    # 4. We start with c_k = 0 (constant function) and try to see if we can improve.
    #    Wait, we proved constant is the global min for sum(f^2).
    #    But the peak can be at other lags.
    #    If we make the function slightly non-constant, we might reduce the peak at lag 0
    #    at the cost of increasing peaks at other lags.
    #    If the "penalty" for increasing other lags is less than the reduction at lag 0,
    #    we might improve the overall max.
    #    This is possible if the constant function has a "sharp" peak at lag 0
    #    and the function can be shaped to spread that peak out.
    
    # Let's try to optimize the coefficients.
    # We will use a limited number of cosine terms.
    num_coeffs = 5
    
    # Initial guess: all zeros (constant function)
    x0 = np.zeros(num_coeffs)
    
    # Bounds for coefficients
    # We want to ensure f >= 0.
    # f(x) = 1 + sum c_k cos(k pi x).
    # Min f(x) >= 0.
    # A safe bound is |c_k| < 1/num_coeffs? No, that's too restrictive.
    # Let's use bounds [-0.5, 0.5] and check non-negativity in the objective.
    # Or better, use a transformation that guarantees non-negativity.
    # But let's stick to bounds for simplicity.
    
    bounds = [(-0.5, 0.5)] * num_coeffs
    
    # Objective function wrapper
    def obj_wrapper(coeffs):
        return objective(coeffs, base_height=1.0, temperature=0.01)
    
    # Optimize
    best_coeffs = x0
    best_val = obj_wrapper(x0)
    
    # Try a few optimization runs
    methods = ['L-BFGS-B', 'Nelder-Mead']
    
    for method in methods:
        if time.time() - start_time > TIME_LIMIT * 0.5:
            break
            
        try:
            res = minimize(obj_wrapper, x0, method=method, bounds=bounds, options={'maxiter': 1000})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except Exception:
            pass
            
    # Construct the function from the best coefficients
    f = parametric_to_step(best_coeffs, base_height=1.0)
    
    # Refine the function directly
    # We can try to optimize the individual bins using a local search.
    # But let's first see if the parametric result is good.
    
    # Let's also try a "random restart" strategy with the parametric model.
    # Generate random coefficients and optimize from there.
    
    num_restarts = 5
    for _ in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT * 0.8:
            break
            
        x0_rand = np.random.uniform(-0.2, 0.2, num_coeffs)
        
        try:
            res = minimize(obj_wrapper, x0_rand, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except Exception:
            pass
            
    f = parametric_to_step(best_coeffs, base_height=1.0)
    
    # Final check and normalization
    # The score is invariant to scaling, so we can normalize f to sum to 1.
    s = np.sum(f)
    if s > 0:
        f = f / s
        
    # One last polish: try to reduce the peak by averaging
    # This is a heuristic and might not work.
    # Let's just return the result.
    
    return [float(x) for x in f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
