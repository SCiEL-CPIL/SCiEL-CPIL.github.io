import numpy as np
import time
from scipy.optimize import minimize

N = 600
dx = 1.0 / (2 * N)

def compute_raw_score(f):
    """
    Compute the normalized maximum of the autoconvolution.
    f: array of nonnegative step heights.
    Score = max(autoconv) / (sum(f))^2
    """
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-12):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-12:
        return np.inf
    
    # Compute autoconvolution using FFT
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)[:2*N-1]
    
    return np.max(conv) / (s * s)

def smooth_max(x, beta):
    """
    Smooth approximation of max(x) using log-sum-exp.
    """
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    x_shifted = x - x_max
    exp_x = np.exp(beta * x_shifted)
    log_sum = np.log(np.sum(exp_x))
    return x_max + log_sum / beta

def objective_and_grad(f, beta=10.0):
    """
    Returns the objective value (smooth max of normalized conv) and its gradient.
    """
    f = np.asarray(f, dtype=np.float64)
    s = np.sum(f)
    if s < 1e-12:
        return 1e10, np.zeros_like(f)
    
    # Normalize to sum N for the gradient step calculation
    f_norm = f * (N / s)
    
    n_fft = 2 * N
    F = np.fft.rfft(f_norm, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)[:2*N-1]
    
    max_c = np.max(conv)
    if max_c < 1e-15:
        return 0.0, np.zeros_like(f)
        
    beta_safe = min(beta, 100.0)
    shifted = conv - max_c
    exp_shifted = np.exp(beta_safe * shifted)
    sum_exp = np.sum(exp_shifted)
    if sum_exp < 1e-15:
        return 0.0, np.zeros_like(f)
    weights = exp_shifted / sum_exp
    
    W = np.fft.rfft(weights, n=n_fft)
    F_full = np.fft.rfft(f_norm, n=n_fft)
    grad_conv = 2.0 * np.fft.irfft(W * np.conj(F_full), n=n_fft)[:N]
    
    grad_f = grad_conv * (N / s)
    
    val = max_c / (s * s)
    
    return val, grad_f

def spectral_gradient_descent(f_init, start_time, max_time, beta_start=1.0, beta_end=50.0):
    """
    Gradient descent in the frequency/time domain with non-negativity projection.
    Includes backtracking line search.
    """
    f = f_init.copy()
    s = np.sum(f)
    if s < 1e-12:
        return f, compute_raw_score(f)
    f = f * (N / s)
    
    best_f = f.copy()
    best_val = compute_raw_score(f)
    
    lr = 0.1
    min_lr = 1e-10
    max_lr = 1.0
    patience = 0
    max_patience = 20
    
    n_steps = 0
    while time.time() - start_time < max_time:
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / max_time)
        beta = beta_start + (beta_end - beta_start) * progress
        
        val, grad = objective_and_grad(f, beta=beta)
        
        if val < best_val:
            best_val = val
            best_f = f.copy()
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                lr *= 0.5
                if lr < min_lr:
                    break
                patience = 0
        
        # Backtracking line search
        f_new = f
        lr_try = lr
        val_new = val
        improved = False
        
        for _ in range(10):
            candidate = f - lr_try * grad
            candidate = np.maximum(candidate, 0.0)
            s_cand = np.sum(candidate)
            if s_cand < 1e-12:
                lr_try *= 0.5
                continue
            candidate = candidate * (N / s_cand)
            
            val_cand, _ = objective_and_grad(candidate, beta=beta)
            
            if val_cand < val:
                f_new = candidate
                val_new = val_cand
                improved = True
                break
            else:
                lr_try *= 0.5
        
        if improved:
            f = f_new
            lr = min(lr * 1.1, max_lr)
        else:
            lr = max(lr * 0.5, min_lr)
            if lr < min_lr:
                break
            
        n_steps += 1
        if n_steps > 5000:
             break

    return best_f, best_val

def polish_with_lbfgsb(f_init, start_time, max_time, beta=100.0):
    """
    Polish the result using L-BFGS-B.
    """
    if time.time() - start_time > max_time - 5:
        return f_init, compute_raw_score(f_init)
    
    # L-BFGS-B requires the objective to be differentiable and bounded below.
    # We use the smooth max with high beta.
    # We must ensure non-negativity via bounds.
    
    def lbfgsb_obj(x):
        val, grad = objective_and_grad(x, beta=beta)
        return val, grad

    # Initial value
    f = f_init.copy()
    s = np.sum(f)
    if s < 1e-12:
        return f, compute_raw_score(f)
    f = f * (N / s)
    
    best_val = compute_raw_score(f)
    best_f = f.copy()
    
    try:
        res = minimize(
            lbfgsb_obj, 
            f, 
            method='L-BFGS-B', 
            bounds=[(0, None)]*N,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        if res.success or res.fun < best_val:
            f_res = res.x
            f_res = np.maximum(f_res, 0.0)
            s_res = np.sum(f_res)
            if s_res > 1e-12:
                f_res = f_res * (N / s_res)
                val_res = compute_raw_score(f_res)
                if val_res < best_val:
                    best_val = val_res
                    best_f = f_res
    except Exception as e:
        pass
        
    return best_f, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    
    best_f = None
    best_val = np.inf
    
    # Initializations
    np.random.seed(42)
    initializations = [
        np.exp(-0.5 * (x / 0.1)**2),
        np.exp(-0.5 * (x / 0.05)**2),
        np.exp(-0.5 * (x / 0.2)**2),
        np.ones(N) * 0.5 + 0.5 * np.cos(2 * np.pi * x / 0.5),
        # Random positive noise
        np.random.rand(N) + 0.1,
        np.random.rand(N) + 0.1,
        # Sparse random
        np.random.rand(N) > 0.9,
    ]
    
    # Time budget per initialization
    # We have ~140s. Let's do 5 restarts with GD, then polish the best one.
    # Or do GD for each, keep best, then polish best.
    
    for f_init in initializations:
        if time.time() > deadline - 20: # Leave 20s for polish
            break
            
        f_res, val_res = spectral_gradient_descent(f_init, start_time, deadline - start_time - 20)
        
        f_res = np.maximum(f_res, 0.0)
        s = np.sum(f_res)
        if s < 1e-12:
            continue
        # Normalize for comparison
        f_res_norm = f_res / s
        val_final = compute_raw_score(f_res_norm)
        
        if val_final < best_val:
            best_val = val_final
            best_f = f_res_norm.copy()
    
    if best_f is None:
        best_f = np.ones(N)
    
    # Polish the best candidate
    if time.time() < deadline - 5:
        best_f, best_val = polish_with_lbfgsb(best_f, start_time, deadline)
        
    # Final check
    final_val = compute_raw_score(best_f)
    if final_val < best_val:
        best_val = final_val
        
    return list(best_f)

if __name__ == "__main__":
    f = construct()
    print(f"Length: {len(f)}")
    print(f"Score: {compute_raw_score(np.array(f))}")
