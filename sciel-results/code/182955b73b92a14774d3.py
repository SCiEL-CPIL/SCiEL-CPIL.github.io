import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Use FFT for linear autoconvolution
    M = 1
    while M < 2 * n - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    # Normalization factor derived in original code
    # score = max_conv * n / (total^2 * w)
    score = max_conv * n / (total * total * w)
    return score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def time_left():
        return deadline - time.time()

    # Pre-compute FFT size
    M = 1
    while M < 2 * N - 1:
        M *= 2

    def softmax(x):
        """Numerically stable softmax."""
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)

    def objective_and_grad(x, alpha):
        """
        Compute the smooth max of the autoconvolution of softmax(x)
        and its gradient with respect to x.
        """
        p = softmax(x)
        
        # Compute autoconvolution
        f = np.zeros(M, dtype=np.float64)
        f[:N] = p
        F = np.fft.rfft(f)
        conv = np.fft.irfft(F * F, n=M)
        
        # Smooth max
        max_conv = np.max(conv)
        # Scale to avoid overflow in exp
        conv_scaled = alpha * (conv - max_conv)
        exp_conv = np.exp(conv_scaled)
        sum_exp = np.sum(exp_conv)
        smooth_max_val = max_conv + np.log(sum_exp) / alpha
        
        # Weights for the gradient
        weights = exp_conv / sum_exp
        
        # Gradient of smooth_max w.r.t conv
        # d(smooth_max)/d(conv_k) = weights[k]
        
        # Gradient of conv w.r.t p
        # conv_k = sum_i p_i p_{k-i}
        # d(conv_k)/d(p_j) = 2 * p_{k-j} if 0 <= k-j < N
        
        # Gradient w.r.t p_j
        # grad_p[j] = sum_k weights[k] * 2 * p_{k-j}
        # This is a correlation of weights and p
        W = np.fft.rfft(weights)
        P = np.fft.rfft(p)
        # corr[j] = sum_k weights[k] p_{k-j}
        # In FFT: ifft(fft(weights) * conj(fft(p))) gives circular correlation
        # Since p is zero-padded to M >= 2N-1, circular corr in M points 
        # matches linear corr for j in [0, N-1] if weights are also effectively 
        # supported in [0, 2N-2].
        corr = np.fft.irfft(W * np.conj(P), n=M)
        grad_p = 2.0 * corr[:N]
        
        # Chain rule: p = softmax(x)
        # p_j = e^{x_j} / S
        # d p_j / d x_i = p_j (delta_{ij} - p_i)
        # grad_x[i] = sum_j grad_p[j] * d p_j / d x_i
        #           = sum_j grad_p[j] * p_j (delta_{ij} - p_i)
        #           = grad_p[i] * p_i - p_i * sum_j grad_p[j] * p_j
        
        dot_term = np.dot(grad_p, p)
        grad_x = p * (grad_p - dot_term)
        
        return smooth_max_val, grad_x

    # Strategy:
    # 1. Generate diverse initial guesses for x (log-heights)
    # 2. Run L-BFGS-B for each, with increasing alpha (homotopy)
    # 3. Track best raw score

    rng = np.random.default_rng(42)
    
    best_p = np.ones(N) / N
    best_raw_score = compute_score(best_p)
    
    # Initial guesses
    inits = []
    
    # 1. Uniform (x = 0)
    inits.append(np.zeros(N))
    
    # 2. Gaussian-like bumps in log-space
    # We want p to be Gaussian. x = log(p).
    # p(x) ~ exp(- (x-c)^2 / 2sigma^2)
    # x_i = - (i-c)^2 / 2sigma^2 + const
    x_grid = np.arange(N)
    center = N / 2.0
    for sigma in [50, 100, 150, 200, 300]:
        x_init = -((x_grid - center) ** 2) / (2 * sigma * sigma)
        inits.append(x_init)
        
    # 3. U-shape (High at edges, low in middle)
    # p ~ 1 + a * (2*x/N - 1)^2
    # x = log(1 + a * (...))
    for a in [1, 2, 5, 10]:
        t = 2.0 * x_grid / N - 1.0
        p_shape = 1.0 + a * (t ** 2)
        x_init = np.log(p_shape)
        inits.append(x_init)
        
    # 4. Random smooth functions
    for _ in range(5):
        # Sum of cosines
        x_init = np.zeros(N)
        for k in [1, 2, 3, 5]:
            amp = rng.uniform(0.5, 2.0)
            phase = rng.uniform(0, 2*np.pi)
            x_init += amp * np.cos(2 * np.pi * k * x_grid / N + phase)
        inits.append(x_init)

    # Optimization loop
    # We use a homotopy approach: start with low alpha (smoother), increase alpha
    alphas = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    for x_init in inits:
        if time_left() < 20:
            break
            
        # Normalize x_init to avoid huge values
        x_init = x_init - np.mean(x_init)
        
        # Run optimization for this initial guess
        x_opt = x_init.copy()
        
        for alpha in alphas:
            if time_left() < 10:
                break
                
            # Define objective for scipy
            def obj(x_curr):
                val, _ = objective_and_grad(x_curr, alpha)
                return val
            
            def grad(x_curr):
                _, g = objective_and_grad(x_curr, alpha)
                return g
            
            try:
                res = minimize(
                    obj, 
                    x_opt, 
                    jac=grad, 
                    method='L-BFGS-B', 
                    options={'maxiter': 50, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                x_opt = res.x
            except Exception:
                # If optimization fails, keep previous x_opt
                pass
        
        # Evaluate final result
        p_opt = softmax(x_opt)
        raw_score = compute_score(p_opt)
        
        if raw_score < best_raw_score:
            best_raw_score = raw_score
            best_p = p_opt

    # Final Polish with high alpha on the best solution
    if time_left() > 15:
        x_best = np.log(best_p + 1e-300)
        x_best = x_best - np.mean(x_best)
        
        # Run a few more steps with very high alpha
        for alpha in [1000.0, 2000.0]:
            if time_left() < 5:
                break
            
            def obj(x_curr):
                val, _ = objective_and_grad(x_curr, alpha)
                return val
            
            def grad(x_curr):
                _, g = objective_and_grad(x_curr, alpha)
                return g
            
            try:
                res = minimize(
                    obj, 
                    x_best, 
                    jac=grad, 
                    method='L-BFGS-B', 
                    options={'maxiter': 50, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                x_best = res.x
            except Exception:
                pass
        
        p_final = softmax(x_best)
        raw_score_final = compute_score(p_final)
        
        if raw_score_final < best_raw_score:
            best_raw_score = raw_score_final
            best_p = p_final

    # Ensure non-negativity and normalize to match interface (heights)
    # The interface expects "step heights". The score is scale invariant.
    # We return the probabilities p_i.
    best_p = np.maximum(best_p, 0.0)
    
    return list(best_p)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
