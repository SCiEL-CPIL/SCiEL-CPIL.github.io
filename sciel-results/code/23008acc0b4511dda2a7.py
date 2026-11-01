"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_acf(h):
    """Compute linear autocorrelation using FFT."""
    # We need aperiodic autocorrelation.
    # Zero-pad to 2*N to avoid circular convolution effects.
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real
    # The aperiodic autocorrelation R[k] for k=0..N-1 is in acf[0:N]
    # The negative lags are in acf[N+1:] (reversed)
    # We only care about positive lags k=1..N-1
    return acf

def _max_sidelobe_val(acf):
    """Compute max absolute value of sidelobes."""
    # Sidelobes are acf[1:N]
    return np.max(np.abs(acf[1:N]))

def _soft_max_sidelobe_val(acf, beta):
    """Smooth approximation of max |acf[1:N]|."""
    sidelobes = np.abs(acf[1:N])
    max_val = np.max(sidelobes)
    if max_val < 1e-10:
        return max_val
    shifted = beta * (sidelobes - max_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta

def _soft_max_objective_and_grad(x, beta):
    """
    Objective: Soft max of absolute sidelobes.
    Returns (value, gradient).
    x is the continuous variable vector.
    """
    acf = _compute_acf(x)
    sidelobes = acf[1:N]
    abs_sidelobes = np.abs(sidelobes)
    
    max_val = np.max(abs_sidelobes)
    if max_val < 1e-10:
        return max_val, np.zeros(N)
        
    shifted = beta * (abs_sidelobes - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj_val = max_val + np.log(sum_exp) / beta
    
    # Gradient calculation
    # d(obj)/dx_i = sum_k w_k * sign(acf[k]) * d(acf[k])/dx_i
    # w_k = beta * exp_shifted[k-1] / sum_exp
    # d(acf[k])/dx_i = x[i+k] + x[i-k] (with bounds)
    
    weights = beta * exp_shifted / sum_exp
    grad = np.zeros(N)
    
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k < 1e-15:
            continue
        
        sign_k = np.sign(acf[k])
        c_k = sign_k * w_k
        if c_k == 0:
            continue
        
        # Update gradient contributions
        # acf[k] = sum_{i=0}^{N-1-k} x[i] x[i+k]
        # d(acf[k])/dx[i] = x[i+k] for i in [0, N-1-k)
        # d(acf[k])/dx[i] = x[i-k] for i in [k, N)
        
        # Vectorized update
        # For i in 0..N-1-k: grad[i] += c_k * x[i+k]
        grad[0:N-k] += c_k * x[k:N]
        # For i in k..N-1: grad[i] += c_k * x[i-k]
        grad[k:N] += c_k * x[0:N-k]
        
    return obj_val, grad

def _local_search_max(h, deadline, rng, max_rounds=50):
    """
    Fast local search on binary sequence: flip bits that improve max sidelobe.
    """
    acf = _compute_acf(h)
    current_val = _max_sidelobe_val(acf)
    best_h = h.copy()
    best_val = current_val
    
    indices = np.arange(N)
    improved = True
    rounds = 0
    
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                return best_h, best_val
            
            # Try flipping h[i]
            h[i] = -h[i]
            new_acf = _compute_acf(h)
            new_val = _max_sidelobe_val(new_acf)
            
            if new_val < current_val:
                current_val = new_val
                acf = new_acf
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                h[i] = -h[i]  # Revert
    
    return best_h, best_val

def _simulated_annealing(h, deadline, rng, T0=0.01, cooling=0.999):
    """
    Simulated Annealing to escape local minima.
    """
    acf = _compute_acf(h)
    current_val = _max_sidelobe_val(acf)
    best_h = h.copy()
    best_val = current_val
    T = T0
    
    while time.time() < deadline:
        i = rng.integers(N)
        old_h_i = h[i]
        h[i] = -old_h_i
        
        new_acf = _compute_acf(h)
        new_val = _max_sidelobe_val(new_acf)
        
        delta = new_val - current_val
        
        if delta < 0 or (T > 1e-9 and rng.random() < np.exp(-delta / T)):
            # Accept
            current_val = new_val
            acf = new_acf
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Revert
            h[i] = old_h_i
            
        T *= cooling
        if T < 1e-9:
            break
            
    return best_h, best_val

def construct():
    start_time = time.time()
    # Evaluator limit is 150s. Use 140s as deadline.
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_val = np.inf
    
    # Phase 1: Continuous Relaxation
    # Use multiple restarts with different betas
    betas = [50.0, 100.0, 200.0]
    num_restarts = 5
    
    for i in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
            
        beta = betas[i % len(betas)]
        
        # Random binary initialization
        x0 = rng.choice([-1.0, 1.0], size=N)
        
        try:
            # Define objective and gradient wrapper
            def obj_grad(x, beta=beta):
                return _soft_max_objective_and_grad(x, beta)
            
            result = minimize(
                obj_grad,
                x0,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            # Round to binary
            h_bin = np.sign(result.x)
            h_bin[h_bin == 0] = 1.0
            
            acf = _compute_acf(h_bin)
            val = _max_sidelobe_val(acf)
            
            if val < best_val:
                best_val = val
                best_h = h_bin.copy()
                
        except Exception:
            continue
    
    # If continuous phase didn't find anything good, use random
    if best_h is None:
        best_h = rng.choice([-1.0, 1.0], size=N)
        acf = _compute_acf(best_h)
        best_val = _max_sidelobe_val(acf)

    # Phase 2: Discrete Local Search
    if time.time() < deadline - 10.0:
        res_h, res_val = _local_search_max(best_h.copy(), deadline - 10.0, rng, max_rounds=100)
        if res_val < best_val:
            best_val = res_val
            best_h = res_h

    # Phase 3: Simulated Annealing
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Set T0 relative to current best to allow meaningful jumps
        T0 = max(0.001, best_val * 0.1)
        res_h, res_val = _simulated_annealing(best_h.copy(), deadline - 2.0, rng, T0=T0)
        if res_val < best_val:
            best_val = res_val
            best_h = res_h
            
    # Final quick polish
    if time.time() < deadline - 1.0:
        res_h, res_val = _local_search_max(best_h.copy(), deadline - 1.0, rng, max_rounds=20)
        if res_val < best_val:
            best_h = res_h

    return best_h

# EVOLVE-BLOCK-END
