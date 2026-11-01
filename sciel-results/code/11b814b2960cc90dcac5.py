import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 150.0  # Explicit deadline guard with safety margin

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The metric is: max_k |conv(h, h)[k]| / (sum_i |h_i|)^2
    where conv is the discrete autoconvolution.
    """
    h = np.asarray(signed_step, dtype=np.float64)
    conv = np.convolve(h, h, mode='full')
    sum_abs_h = np.sum(np.abs(h))
    
    if sum_abs_h < 1e-12:
        return float('inf')
        
    return np.max(np.abs(conv)) / (sum_abs_h ** 2)

def _soft_max_abs_conv(x, T=1.0):
    """
    Smooth approximation of max|conv(tanh(x))| using log-sum-exp.
    """
    h = np.tanh(x)
    conv = np.convolve(h, h, mode='full')
    eps = 1e-12
    abs_conv = np.sqrt(conv**2 + eps)
    max_abs = np.max(abs_conv)
    shifted = (abs_conv - max_abs) / T
    log_sum_exp = np.log(np.sum(np.exp(shifted))) + max_abs / T
    return T * log_sum_exp

def _objective_and_grad(x):
    """
    Objective: soft_max_abs_conv / (sum|tanh(x)|)^2
    Returns (value, gradient)
    """
    h = np.tanh(x)
    conv = np.convolve(h, h, mode='full')
    eps = 1e-12
    abs_conv = np.sqrt(conv**2 + eps)
    sum_abs_h = np.sum(np.sqrt(h**2 + eps))
    sum_abs_h_sq = sum_abs_h ** 2
    
    # Soft max with temperature
    T = 1.0
    max_abs = np.max(abs_conv)
    shifted = (abs_conv - max_abs) / T
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum_exp = np.log(sum_exp) + max_abs / T
    soft_max = T * log_sum_exp
    
    # Objective value
    value = soft_max / sum_abs_h_sq
    
    # Gradient via finite differences (robust for non-smooth max)
    # Use central differences with small step
    dx = 1e-5
    grad = np.zeros_like(x)
    for i in range(N):
        x_plus = x.copy()
        x_plus[i] += dx
        x_minus = x.copy()
        x_minus[i] -= dx
        
        h_plus = np.tanh(x_plus)
        conv_plus = np.convolve(h_plus, h_plus, mode='full')
        abs_conv_plus = np.sqrt(conv_plus**2 + eps)
        sum_abs_plus = np.sum(np.sqrt(h_plus**2 + eps))
        max_plus = np.max(abs_conv_plus)
        shifted_plus = (abs_conv_plus - max_plus) / T
        soft_max_plus = T * (np.log(np.sum(np.exp(shifted_plus))) + max_plus / T)
        val_plus = soft_max_plus / (sum_abs_plus ** 2)
        
        h_minus = np.tanh(x_minus)
        conv_minus = np.convolve(h_minus, h_minus, mode='full')
        abs_conv_minus = np.sqrt(conv_minus**2 + eps)
        sum_abs_minus = np.sum(np.sqrt(h_minus**2 + eps))
        max_minus = np.max(abs_conv_minus)
        shifted_minus = (abs_conv_minus - max_minus) / T
        soft_max_minus = T * (np.log(np.sum(np.exp(shifted_minus))) + max_minus / T)
        val_minus = soft_max_minus / (sum_abs_minus ** 2)
        
        grad[i] = (val_plus - val_minus) / (2 * dx)
    
    return value, grad

def _local_search(h, start_time, max_iters=5000):
    """
    Fast discrete local search refinement on binary {-1, 1} steps.
    """
    h = np.sign(h)
    h[h == 0] = 1
    best_c3 = _compute_c3(h)
    np.random.seed(42)
    no_improve = 0
    patience = 2000
    
    for _ in range(max_iters):
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        idx = np.random.randint(N)
        old_val = h[idx]
        h[idx] = -old_val
        new_c3 = _compute_c3(h)
        
        if new_c3 < best_c3:
            best_c3 = new_c3
            no_improve = 0
        else:
            h[idx] = old_val
            no_improve += 1
        
        if no_improve >= patience:
            break
    
    return h, best_c3

def construct():
    start_time = time.time()
    
    # Phase 1: Continuous relaxation via L-BFGS-B
    # Multiple diverse initializations to navigate non-convex landscape
    best_h = None
    best_val = float('inf')
    
    np.random.seed(42)
    
    # Initialization 1: All ones (seed)
    x0_ones = np.full(N, 3.0)
    
    # Initialization 2: Alternating signs
    x0_alt = np.zeros(N)
    x0_alt[::2] = 3.0
    x0_alt[1::2] = -3.0
    
    # Initialization 3: Random
    x0_rand = np.random.randn(N) * 2.0
    
    # Initialization 4: Blocky random (coarse structure)
    x0_block = np.repeat(np.random.choice([2.0, -2.0], size=N//20), 20)
    
    initializations = [x0_ones, x0_alt, x0_rand, x0_block]
    
    for x0 in initializations:
        if time.time() - start_time > DEADLINE - 30.0:
            break
        
        # Use L-BFGS-B with smooth objective
        # We'll use a simpler objective for speed: direct soft max without gradient
        # since finite-difference gradient is too slow for 400 vars
        def objective_simple(x):
            h = np.tanh(x)
            conv = np.convolve(h, h, mode='full')
            eps = 1e-12
            abs_conv = np.sqrt(conv**2 + eps)
            sum_abs_h = np.sum(np.sqrt(h**2 + eps))
            T = 1.0
            max_abs = np.max(abs_conv)
            shifted = (abs_conv - max_abs) / T
            soft_max = T * (np.log(np.sum(np.exp(shifted))) + max_abs / T)
            return soft_max / (sum_abs_h ** 2)
        
        try:
            res = minimize(objective_simple, x0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            h_opt = np.tanh(res.x)
            val_opt = objective_simple(res.x)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt
        except Exception:
            continue
    
    # Phase 2: Discrete refinement
    if best_h is None:
        best_h = np.ones(N)
    
    # Snap to binary and refine
    h_binary = np.sign(best_h)
    h_binary[h_binary == 0] = 1
    
    h_final, c3_final = _local_search(h_binary, start_time)
    
    # Compare with seed
    seed_c3 = _compute_c3(np.ones(N))
    if c3_final > seed_c3:
        h_final = np.ones(N)
    
    return h_final
