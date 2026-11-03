import numpy as np
import time
from scipy.optimize import minimize_scalar

def _autoconv(f):
    """Compute autoconvolution (convolution f*f) using FFT."""
    n = len(f)
    F_full = np.fft.fft(f, n=2*n)
    conv_full = np.fft.ifft(F_full * F_full)[:2*n-1].real
    return conv_full

def _compute_score(f):
    """Compute the normalized max of autoconvolution."""
    n = len(f)
    f = np.asarray(f, dtype=np.float64)
    if np.all(f == 0):
        return 0.0
    conv = _autoconv(f)
    max_conv = np.max(conv)
    sum_f = np.sum(f)
    norm = sum_f * sum_f
    if norm == 0:
        return 0.0
    return max_conv / norm

def construct():
    n = 600
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    def time_remaining():
        return deadline - time.time()
    
    # Structural change: Represent f as |g| where g is complex with random phases
    # Optimize the phase angles to flatten the autoconvolution peak
    
    rng = np.random.default_rng(42)
    
    # Initialize with a smooth magnitude profile (Gaussian-like)
    x = np.linspace(-0.25, 0.25, n, endpoint=False)
    mag = np.exp(-x**2 / (2 * 0.08**2))
    mag = np.maximum(mag, 1e-6)  # avoid zeros
    
    # Random initial phases
    phases = rng.uniform(0, 2*np.pi, n)
    
    def phases_to_f(phases, mag):
        """Convert phases to real nonnegative step function."""
        g = mag * np.exp(1j * phases)
        f = np.abs(g)
        return f
    
    def compute_from_phases(phases, mag):
        """Compute score from phases."""
        f = phases_to_f(phases, mag)
        return _compute_score(f)
    
    # Initialize best
    best_phases = phases.copy()
    best_score = compute_from_phases(best_phases, mag)
    
    # Gradient descent on phase angles
    # The objective is max(conv(f,f)) / (sum(f))^2 where f = |g|
    # We can use smooth max surrogate for gradient computation
    
    def smooth_max(conv, temperature=0.005):
        """Soft maximum approximation."""
        max_val = np.max(conv)
        return max_val + temperature * np.log(np.sum(np.exp((conv - max_val) / temperature)))
    
    def objective_and_grad(phases, mag, temperature=0.005):
        """Compute objective and gradient with respect to phases."""
        f = phases_to_f(phases, mag)
        if np.sum(f) < 1e-10:
            return 1.0, np.zeros_like(phases)
        
        conv = _autoconv(f)
        smax = smooth_max(conv, temperature)
        obj = smax / (np.sum(f) ** 2)
        
        # Gradient computation using finite differences (simplified)
        # For phase j, df/dphase_j = -mag[j] * sin(phase_j)
        # We need d(conv)/df and then chain rule
        
        # Simplified: use numerical gradient on a subset
        grad = np.zeros_like(phases)
        eps = 0.01
        for j in range(0, n, 3):  # sample every 3rd for speed
            if time_remaining() < 10:
                break
            test_phases = phases.copy()
            test_phases[j] += eps
            f_test = phases_to_f(test_phases, mag)
            conv_test = _autoconv(f_test)
            smax_test = smooth_max(conv_test, temperature)
            obj_test = smax_test / (np.sum(f_test) ** 2)
            grad[j] = (obj_test - obj) / eps
        
        # Fill in unsampled gradients by interpolation
        if n > 3:
            sampled_idx = np.arange(0, n, 3)
            unsampled_idx = np.setdiff1d(np.arange(n), sampled_idx)
            if len(unsampled_idx) > 0 and len(sampled_idx) > 1:
                grad[unsampled_idx] = np.interp(unsampled_idx, sampled_idx, grad[sampled_idx])
        
        return obj, grad
    
    # Phase 1: Gradient descent on phases
    lr = 0.05
    max_iters = 500
    patience = 30
    no_improve_count = 0
    
    for it in range(max_iters):
        if time_remaining() < 20.0:
            break
        
        obj, grad = objective_and_grad(best_phases, mag, temperature=0.005)
        
        # Update phases
        new_phases = best_phases - lr * grad
        new_phases = np.mod(new_phases, 2*np.pi)
        
        new_score = compute_from_phases(new_phases, mag)
        
        if new_score < best_score - 1e-10:
            best_phases = new_phases
            best_score = new_score
            no_improve_count = 0
            # Decay learning rate slowly
            lr *= 0.999
        else:
            no_improve_count += 1
            # Reduce learning rate when stuck
            lr *= 0.95
        
        if no_improve_count >= patience:
            # Random restart with small perturbation
            best_phases = best_phases + rng.normal(0, 0.1, n)
            best_phases = np.mod(best_phases, 2*np.pi)
            best_score = compute_from_phases(best_phases, mag)
            no_improve_count = 0
            lr = 0.03
    
    # Phase 2: Refine magnitude profile while keeping optimized phases
    # Try different magnitude profiles
    
    current_f = phases_to_f(best_phases, mag)
    current_score = _compute_score(current_f)
    
    if time_remaining() > 30.0:
        # Try to optimize the magnitude profile
        # Use coordinate descent on log-magnitude
        
        log_mag = np.log(mag)
        
        def mag_obj(log_mag_vec):
            mag_vec = np.exp(log_mag_vec)
            f = phases_to_f(best_phases, mag_vec)
            return _compute_score(f)
        
        # Coordinate descent on log-magnitude
        max_mag_passes = 10
        for mag_pass in range(max_mag_passes):
            if time_remaining() < 15.0:
                break
            
            improved = False
            # Random subset
            idx = rng.choice(n, size=100, replace=False)
            idx = np.sort(idx)
            
            for i in idx:
                if time_remaining() < 15.0:
                    break
                
                orig_val = log_mag[i]
                scale = 0.5
                
                def obj_mag(val):
                    test_log_mag = log_mag.copy()
                    test_log_mag[i] = val
                    return mag_obj(test_log_mag)
                
                res = minimize_scalar(
                    obj_mag,
                    bounds=(orig_val - scale, orig_val + scale),
                    method='bounded',
                    options={'xatol': 1e-4, 'maxiter': 10}
                )
                
                test_f = phases_to_f(best_phases, np.exp(res.x))
                test_score = _compute_score(test_f)
                
                if test_score < current_score - 1e-10:
                    log_mag[i] = res.x
                    current_score = test_score
                    improved = True
            
            if not improved:
                break
        
        # Update best if improved
        final_mag = np.exp(log_mag)
        final_f = phases_to_f(best_phases, final_mag)
        final_score = _compute_score(final_f)
        
        if final_score < best_score:
            best_score = final_score
            mag = final_mag
    
    # Final: compute exact score and return
    f = phases_to_f(best_phases, mag)
    f = np.maximum(f, 0.0)
    return list(f)
