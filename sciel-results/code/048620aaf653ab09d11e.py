"""Model-and-optimize: Subgradient descent with multi-restart and simulated annealing."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 148

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 2)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x):
    """Compute objective value only (fast, no gradient)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    return m / (s * s)

def _subgradient_step(x, step_size, rng):
    """
    Perform one subgradient descent step.
    Uses the subgradient of the max|ac| term.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    s = np.sum(x)
    s2 = s * s
    
    if s2 < 1e-10:
        return x, 1e10

    # Find the dominant lag(s)
    # To make it more robust, we can average the gradient over the top few peaks
    # or just use the single max. Let's use the single max for now, 
    # but add a small random component to break symmetry if multiple peaks are close.
    max_val = np.max(abs_ac)
    
    # Get indices of peaks close to the max (within 1%)
    threshold = max_val * 0.99
    peak_indices = np.where(abs_ac >= threshold)[0]
    
    if len(peak_indices) == 0:
        peak_indices = [np.argmax(abs_ac)]
    
    # Compute subgradient for these peaks
    # grad_ac_k = sign(ac_k) * delta_k
    # grad_x = sum_k weights_k * sign(ac_k) * cross_corr(x, delta_k)
    # cross_corr(x, delta_k) is effectively shifting x.
    # ac_k = sum_i x_i x_{i+k}
    # d(ac_k)/d(x_j) = x_{j+k} + x_{j-k} (with 0 padding)
    
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
        
    # We can compute the gradient for all peaks at once using FFT?
    # No, it's easier to just compute the gradient for the selected peaks.
    # Since N=400, and we might have a few peaks, this is fast.
    
    grad_x = np.zeros(N)
    for k in peak_indices:
        sign_k = np.sign(ac[k])
        if sign_k == 0:
            sign_k = 1.0
            
        # d(ac_k)/d(x_j)
        # x_{j+k} is x shifted left by k
        # x_{j-k} is x shifted right by k
        
        # Using slicing:
        # x[j+k] for j in [0, N-1] -> x[k : N+k] (padded with 0)
        # x[j-k] for j in [0, N-1] -> x[-k : N-k] (padded with 0)
        
        # Let's use np.roll or explicit slicing to be safe and fast
        # Shifted x for j+k:
        # Non-zero only if 0 <= j+k < N => 0 <= j < N-k
        x_shifted_plus = np.zeros(N)
        if k < N:
            x_shifted_plus[:N-k] = x[k:]
            
        # Shifted x for j-k:
        # Non-zero only if 0 <= j-k < N => k <= j < N
        x_shifted_minus = np.zeros(N)
        if k < N:
            x_shifted_minus[k:] = x[:N-k]
            
        grad_x += sign_k * (x_shifted_plus + x_shifted_minus)
        
    # Objective gradient approximation:
    # f(x) = M(x) / s^2
    # df/dx = (grad_M * s^2 - M * 2*s*1) / s^4 = grad_M/s^2 - 2*M/(s^3)
    
    M = max_val
    grad_obj = grad_x / s2 - 2.0 * M * np.ones(N) / (s2 * s)
    
    # Normalize gradient to control step size effectively
    grad_norm = np.linalg.norm(grad_obj)
    if grad_norm > 1e-10:
        x_trial = x - step_size * grad_obj / grad_norm
    else:
        x_trial = x + rng.normal(0, step_size, size=N)
        
    return x_trial, _objective_only(x_trial)

def _subgradient_optimize(x0, rng, max_iters=5000, init_step=0.1, decay=0.995, min_step=0.001):
    """
    Subgradient descent with adaptive step size and restarts.
    """
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    step_size = init_step
    no_improve = 0
    current_x = x0.copy()
    
    for it in range(max_iters):
        if _check_deadline():
            break
            
        x_trial, val_trial = _subgradient_step(current_x, step_size, rng)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial.copy()
            current_x = x_trial.copy()
            no_improve = 0
            step_size = min(0.2, step_size * 1.01) # Increase step slightly if improving
        else:
            # If no improvement, keep current_x but maybe perturb it
            # Simulated Annealing style: accept worse with some probability
            # Or just restart step size
            no_improve += 1
            step_size = max(min_step, step_size * decay)
            
            if no_improve > 100:
                # Restart from best with perturbation
                current_x = best_x + rng.normal(0, 0.1, size=N)
                step_size = init_step * 0.5
                no_improve = 0
                
    return best_x, best_val

def _random_restart_search(rng, num_restarts=20):
    """
    Generate random initial points and optimize each.
    """
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    for i in range(num_restarts):
        if _check_deadline():
            break
            
        # Generate diverse random starts
        # Mix of smooth and noisy
        if i % 3 == 0:
            # Smooth low freq
            t = np.linspace(0, 1, N)
            x0 = 1.0 + 0.5 * np.sin(2 * np.pi * rng.uniform(1, 5) * t + rng.uniform(0, 2*np.pi))
        elif i % 3 == 1:
            # Noisy
            x0 = 1.0 + 0.3 * rng.normal(0, 1, size=N)
        else:
            # Spectral init
            L = 2 * N - 1
            L_fft = 1
            while L_fft < L:
                L_fft *= 2
            freqs = np.fft.rfftfreq(L_fft)
            mag = np.exp(-10 * np.abs(freqs) * L_fft)
            phase = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
            phase[0] = 0
            spectrum = mag * np.exp(1j * phase)
            spectrum[0] = mag[0]
            x0 = np.fft.irfft(spectrum, n=L_fft)[:N]
            
        s = np.sum(x0)
        if s < 0:
            x0 = -x0
            
        # Optimize this start
        x_opt, val_opt = _subgradient_optimize(x0, rng, max_iters=2000, init_step=0.15)
        
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt.copy()
            
    return best_x, best_val

def construct():
    rng = np.random.default_rng(42)
    
    # Start with a simple baseline
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    # Run multi-restart subgradient search
    # We allocate time dynamically
    # First pass: few restarts, short optimization
    # Second pass: more restarts, longer optimization if time permits
    
    # Pass 1: 10 restarts, 1000 iters each
    x1, v1 = _random_restart_search(rng, num_restarts=10)
    if v1 < best_val:
        best_val = v1
        best_x = x1.copy()
        
    if _check_deadline():
        return best_x
        
    # Pass 2: 5 restarts from best + 5 new, 2000 iters each
    # We can also just run _subgradient_optimize on the current best with more iterations
    x2, v2 = _subgradient_optimize(best_x, rng, max_iters=3000, init_step=0.1)
    if v2 < best_val:
        best_val = v2
        best_x = x2.copy()
        
    if _check_deadline():
        return best_x
        
    # Pass 3: Fine-tuning with smaller steps
    x3, v3 = _subgradient_optimize(best_x, rng, max_iters=2000, init_step=0.05, decay=0.998, min_step=0.0005)
    if v3 < best_val:
        best_val = v3
        best_x = x3.copy()
        
    return best_x
# EVOLVE-BLOCK-END
