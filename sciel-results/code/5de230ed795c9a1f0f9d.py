"""Second autocorrelation inequality - optimize 50 nonnegative heights."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import time

N = 50
DEADLINE = 120.0  # seconds
START_TIME = time.time()

def time_remaining():
    return DEADLINE - (time.time() - START_TIME)

def autoconvolution_peak(h):
    """Compute the supremum norm of the autoconvolution of h.
    
    The autoconvolution at lag k is sum_{i} h_i * h_{i+k} for k >= 0.
    The supremum over all lags (positive and negative) is the max over k>=0 of sum_i h_i * h_{i+k}.
    """
    # h is a 1D array of length N
    # Autoconvolution (full): c[k] = sum_i h[i] * h[i+k] for appropriate ranges
    # We need the max over all lags
    # Use FFT for efficiency
    n = len(h)
    # Zero-pad to avoid circular convolution issues
    size = 2 * n - 1
    H = np.fft.rfft(h, n=size)
    C = np.fft.irfft(H * np.conj(H), n=size)
    # C has length size, representing autoconvolution
    # The peak is the max absolute value (but h >= 0 so all non-negative)
    return np.max(C)

def l2_norm_sq(h):
    return np.sum(h ** 2)

def l1_norm(h):
    return np.sum(np.abs(h))

def compute_score(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h * h||_inf)
    
    This is the value we want to maximize. The score is this value.
    """
    l1 = l1_norm(h)
    if l1 < 1e-15:
        return 0.0
    peak = autoconvolution_peak(h)
    if peak < 1e-15:
        return 0.0
    return l2_norm_sq(h) / (l1 * peak)

def objective_neg_ratio(h, t):
    """Epigraph formulation: minimize t such that t >= 1/ratio(h)
    
    Actually, let's think about this differently.
    We want to maximize R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    
    Equivalently, minimize 1/R(h) = ||h||_1 * ||h*h||_inf / ||h||_2^2
    
    Let f(h) = ||h||_1 * ||h*h||_inf / ||h||_2^2
    We want to minimize f(h) subject to h >= 0.
    
    Using epigraph: minimize t
    s.t. t >= ||h||_1 * ||h*h||_inf / ||h||_2^2
         h >= 0
    
    This is still nonsmooth due to max. Let's use a soft approximation.
    """
    pass

def soft_max(x, temperature=0.01):
    """Soft approximation of max using log-sum-exp."""
    if len(x) == 0:
        return 0.0
    m = np.max(x)
    if m < 1e-30:
        return 0.0
    return m + temperature * np.log(np.sum(np.exp((x - m) / temperature)))

def soft_autoconv_peak(h, temperature=0.01):
    """Soft approximation of the autoconvolution peak."""
    n = len(h)
    size = 2 * n - 1
    H = np.fft.rfft(h, n=size)
    C = np.fft.irfft(H * np.conj(H), n=size)
    return soft_max(C, temperature)

def objective_with_epigraph(h, t, temperature=0.01):
    """Minimize t subject to t >= l1 * peak / l2_sq.
    
    For SLSQP, we can use: minimize t + penalty * max(0, l1*peak/l2_sq - t)
    Or better: use constraints.
    
    Let's define:
    - Variables: h (N vars) and t (1 var)
    - Objective: t
    - Constraint: l1(h) * peak(h) / l2_sq(h) - t <= 0
    - Constraint: h_i >= 0 for all i
    """
    l1 = np.sum(h)
    l2_sq = np.sum(h ** 2)
    if l2_sq < 1e-15:
        return 1e10
    peak = soft_autoconv_peak(h, temperature)
    return t

def constraint_ratio(x, temperature=0.01):
    """Constraint: l1 * peak / l2_sq - t <= 0"""
    h = x[:-1]
    t = x[-1]
    l1 = np.sum(h)
    l2_sq = np.sum(h ** 2)
    if l2_sq < 1e-15:
        return -1e10  # feasible
    peak = soft_autoconv_peak(h, temperature)
    return l1 * peak / l2_sq - t

def run_slsqp(h_init, temperature=0.01, maxiter=500):
    """Run SLSQP optimization with epigraph formulation."""
    N_local = len(h_init)
    x0 = np.concatenate([h_init, [compute_score(h_init) + 1.0]])
    
    bounds = [(0, None)] * N_local + [(0, None)]
    
    cons = [
        {'type': 'ineq', 'fun': lambda x: constraint_ratio(x, temperature)},
        # h >= 0 is handled by bounds
    ]
    
    def obj(x):
        return x[-1]
    
    def obj_grad(x):
        grad = np.zeros(len(x))
        grad[-1] = 1.0
        return grad
    
    result = minimize(
        obj, x0, method='SLSQP',
        jac=obj_grad,
        bounds=bounds,
        constraints=cons,
        options={'maxiter': maxiter, 'ftol': 1e-12}
    )
    
    if result.success:
        h_opt = result.x[:-1]
        return h_opt, compute_score(h_opt)
    else:
        return h_init, compute_score(h_init)

def run_gradient_ascent(h_init, steps=200, lr=0.01):
    """Simple gradient ascent on the ratio with smoothing."""
    h = h_init.copy()
    best_h = h.copy()
    best_score = compute_score(h)
    
    for i in range(steps):
        if time_remaining() < 5.0:
            break
        
        # Compute gradient of log(ratio) or ratio directly
        # R(h) = l2_sq / (l1 * peak)
        # dR/dh_i = (2*h_i * l1*peak - l2_sq * (dh_i/l1 * peak + l1 * dpeak/dh_i)) / (l1*peak)^2
        # This is complex. Let's use numerical gradient.
        
        eps = 1e-7
        grad = np.zeros(N)
        for j in range(N):
            h_plus = h.copy()
            h_plus[j] += eps
            h_minus = h.copy()
            h_minus[j] = max(0, h[j] - eps)
            
            R_plus = compute_score(h_plus)
            R_minus = compute_score(h_minus)
            
            if h[j] > eps:
                grad[j] = (R_plus - R_minus) / (2 * eps)
            else:
                grad[j] = (R_plus - R_minus) / eps
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-15:
            grad = grad / gnorm
        
        # Update with projection to non-negative
        h_new = h + lr * grad
        h_new = np.maximum(0, h_new)
        
        # Normalize to keep scale reasonable (ratio is scale-invariant)
        l1_new = np.sum(h_new)
        if l1_new > 1e-15:
            h_new = h_new / l1_new * np.sum(h)
        
        score_new = compute_score(h_new)
        if score_new > best_score:
            best_score = score_new
            best_h = h_new.copy()
        
        h = h_new
    
    return best_h, best_score

def random_start(seed):
    """Generate a random nonnegative vector."""
    rng = np.random.RandomState(seed)
    h = rng.random(N)
    # Add some structure
    h = np.maximum(0, h + 0.1 * np.sin(np.linspace(0, 2 * np.pi, N)))
    h = h / np.sum(h)
    return h

def construct():
    """Construct the optimal heights."""
    best_h = np.ones(N)
    best_score = compute_score(best_h)
    
    # Try different initializations
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    
    for seed in seeds:
        if time_remaining() < 10.0:
            break
        
        h_init = random_start(seed)
        
        # Try SLSQP with epigraph
        try:
            h_opt, score_opt = run_slsqp(h_init, temperature=0.01, maxiter=300)
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
        
        # Try gradient ascent
        if time_remaining() < 10.0:
            break
        try:
            h_ga, score_ga = run_gradient_ascent(h_opt if 'h_opt' in dir() else h_init, steps=100, lr=0.05)
            if score_ga > best_score:
                best_score = score_ga
                best_h = h_ga.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time_remaining() > 5.0:
        try:
            h_polish, score_polish = run_slsqp(best_h, temperature=0.005, maxiter=500)
            if score_polish > best_score:
                best_score = score_polish
                best_h = h_polish.copy()
        except Exception:
            pass
    
    # Ensure non-negative and return as list
    best_h = np.maximum(0, best_h)
    return list(best_h)

# Test on import
if __name__ == "__main__":
    h = construct()
    print(f"Score: {compute_score(np.array(h))}")
