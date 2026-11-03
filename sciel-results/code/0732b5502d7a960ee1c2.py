import numpy as np
from scipy.optimize import minimize
import time

def compute_score(f):
    """
    Compute the score for the given step function f (array of nonnegative values).
    The score is the normalized maximum of the circular autoconvolution.
    score = max_k c_k / mean_k c_k = n * max(c_k) / (sum f)^2
    """
    n = len(f)
    if n == 0 or np.sum(f) == 0:
        return 0.0
    # Circular autoconvolution via FFT
    F = np.fft.fft(f)
    P = np.abs(F) ** 2
    c = np.fft.ifft(P).real
    # Numerical cleanup
    c = np.clip(c, 0, None)
    max_c = np.max(c)
    mean_c = np.mean(c)  # equals (sum f)^2 / n
    if mean_c == 0:
        return 0.0
    return n * max_c / (np.sum(f) ** 2)

def cosine_basis(n, K):
    """
    Create a cosine basis matrix for the first K harmonics.
    The basis includes the constant term and cos(2 pi k t) for k=1..K-1.
    t_i = (i + 0.5) / n, i=0..n-1
    """
    t = (np.arange(n) + 0.5) / n
    B = np.zeros((n, K))
    B[:, 0] = 1.0
    for k in range(1, K):
        B[:, k] = np.cos(2 * np.pi * k * t)
    return B

def parametric_to_step(params, B, floor_val=0.0):
    """
    Convert parametric coefficients to a step function using softplus to ensure non-negativity.
    f_i = softplus(B @ params)
    """
    z = B @ params
    # Softplus: log(1 + exp(z))
    # To avoid overflow, use np.logaddexp(0, z)
    f = np.logaddexp(0.0, z)
    # Ensure a small floor to avoid exact zeros which might cause issues in some contexts, though not strictly necessary
    if floor_val > 0:
        f = np.maximum(f, floor_val)
    return f

def objective(params, B, floor_val=0.0):
    f = parametric_to_step(params, B, floor_val)
    return compute_score(f)

def smooth_max(x, alpha=1.0):
    """Smooth maximum approximation."""
    return np.log(np.sum(np.exp(alpha * x))) / alpha

def objective_smooth(params, B, floor_val=0.0, alpha=100.0):
    """
    Objective using smooth max for better gradient behavior.
    Returns the smooth max of c / mean(c) * n.
    """
    f = parametric_to_step(params, B, floor_val)
    n = len(f)
    F = np.fft.fft(f)
    P = np.abs(F) ** 2
    c = np.fft.ifft(P).real
    c = np.clip(c, 0, None)
    mean_c = np.sum(c) / n
    if mean_c == 0:
        return 0.0
    # Normalize c to sum to 1? No, we want max(c) / mean(c).
    # max(c) / mean(c) = max(c / mean(c)).
    c_norm = c / mean_c
    return n * smooth_max(c_norm, alpha) / n  # Wait, smooth_max approximates max.
    # Actually, we want to minimize max(c) / mean(c).
    # max(c) / mean(c) is the ratio.
    # Let's just compute the ratio directly in the gradient if possible, or use the smooth max of the normalized peaks.
    # If we minimize max(c / mean(c)), since mean(c) is fixed by sum(f), minimizing max(c) is equivalent.
    # But we want the ratio.
    # Let's return the smooth max of c, and divide by mean(c) outside? No, that's not smooth.
    # Let's return smooth_max(c, alpha) / mean(c).
    return smooth_max(c, alpha) / mean_c

def local_refinement(f0, max_iters=500, tol=1e-10):
    """
    Local refinement using gradient descent on the parametric representation.
    """
    n = len(f0)
    K = 50  # Number of harmonics
    B = cosine_basis(n, K)
    
    # Initialize params to match f0 as closely as possible
    # Minimize ||B @ p - f0||^2
    # p_init = pinv(B) @ f0
    try:
        # We want B @ p such that softplus(B @ p) ~ f0
        # B @ p ~ log(f0 - 1) if f0 > 1? No, softplus(z) = log(1+e^z).
        # If f0 is large, log(1+e^z) ~ z. So z ~ f0.
        # If f0 is small, log(1+e^z) ~ e^z. So e^z ~ f0, z ~ log(f0).
        # This is tricky. Let's just use least squares on f0.
        p_init = np.linalg.lstsq(B, f0, rcond=None)[0]
    except np.linalg.LinAlgError:
        p_init = np.zeros(K)
    
    # Optimize
    best_params = p_init
    best_obj = objective_smooth(p_init, B, 0.0, 100.0)
    
    # Use L-BFGS-B
    res = minimize(objective_smooth, p_init, args=(B, 0.0, 100.0), method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
    
    if res.success and res.fun < best_obj:
        best_params = res.x
        best_obj = res.fun
    
    f_opt = parametric_to_step(best_params, B, 0.0)
    
    # Coordinate descent polish
    # Try to improve by adjusting individual bins slightly?
    # No, keep it simple.
    
    return f_opt

def construct():
    n = 600
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial guess: Constant function
    f_best = np.ones(n)
    score_best = compute_score(f_best)
    
    # Multi-start optimization
    n_starts = 10
    
    for i in range(n_starts):
        if time.time() > deadline - 10:
            break
            
        # Random parametric initialization
        K = 30
        B = cosine_basis(n, K)
        
        # Random weights
        params = np.random.normal(0, 0.5, K)
        
        f_curr = parametric_to_step(params, B, 0.0)
        score_curr = compute_score(f_curr)
        
        if score_curr < score_best:
            f_best = f_curr.copy()
            score_best = score_curr
        
        # Local refinement
        f_opt = local_refinement(f_curr)
        score_opt = compute_score(f_opt)
        
        if score_opt < score_best:
            f_best = f_opt.copy()
            score_best = score_opt
    
    # Final polish on the best found
    f_final = local_refinement(f_best)
    score_final = compute_score(f_final)
    if score_final < score_best:
        f_best = f_final
        score_best = score_final
    
    # Ensure non-negativity
    f_best = np.maximum(f_best, 0.0)
    
    return list(f_best)

if __name__ == "__main__":
    # Quick test
    f = construct()
    print(f"Score: {compute_score(np.array(f))}")
