"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _objective(h):
    """Compute the value to minimize."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    return abs(2.0 * N * max_conv / (s * s))

def _grad(h):
    """Gradient of the objective (numerical or analytical)."""
    # Use finite differences for simplicity, but this is slow
    # Better: analytical gradient
    s = np.sum(h)
    conv = np.convolve(h, h)
    max_val = np.max(conv)
    # The max is at some index k*. For a smooth surrogate, we'd use soft-max
    # For now, use the subgradient at the argmax
    k_max = np.argmax(conv)
    # d(conv[k])/dh[i] = h[i+k] + h[i-k] (with bounds)
    # This is getting complex; use numerical gradient
    eps = 1e-7
    grad = np.zeros_like(h)
    for i in range(len(h)):
        h[i] += eps
        f1 = _objective(h)
        h[i] -= 2*eps
        f2 = _objective(h)
        h[i] += eps
        grad[i] = (f1 - f2) / (2*eps)
    return grad

def _generate_initial(rng):
    """Generate initial point with spectral properties to reduce autoconv peak."""
    h = rng.standard_normal(N)
    # Remove DC component to make sum small, then add back a controlled amount
    # Actually, we want sum(h) to be non-zero but arrange h to minimize max autoconv
    # Strategy: create h with approximately constant-magnitude DFT (flat spectrum)
    # This makes autocorrelation close to delta, minimizing off-peak values
    # But we need sum(h) != 0, so we need a DC component
    
    # Generate a random phase sequence (chirp-like)
    k = np.arange(N)
    # Use a quadratic phase (chirp) which has flat magnitude spectrum
    phase = 2 * np.pi * (k * k) / N
    h = np.cos(phase) + 1j * np.sin(phase)
    # Take real part, add small random perturbation
    h = np.real(h) + 0.1 * rng.standard_normal(N)
    # Ensure sum is not zero
    s = np.sum(h)
    if abs(s) < 1e-6:
        h += 0.01
    return h

def _soft_max(conv, beta=50.0):
    """Soft maximum using log-sum-exp for smooth optimization."""
    # Soft max: (1/beta) * log(sum(exp(beta*x)))
    max_x = np.max(conv)
    return max_x + (1.0/beta) * np.log(np.sum(np.exp(beta * (conv - max_x))))

def _smooth_objective(h, beta=50.0):
    """Smooth surrogate for the objective using soft-max."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    soft_max_val = _soft_max(conv, beta)
    return abs(2.0 * N * soft_max_val / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_val = _objective(best_h)  # Should be 2.0
    
    # Try multiple initial points and optimize from each
    n_starts = 8
    
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
        
        # Generate diverse initial points
        if start_idx == 0:
            h = _generate_initial(rng)
        elif start_idx == 1:
            # Chirp with different parameter
            k = np.arange(N)
            phase = 2 * np.pi * (k * k * 3) / N
            h = np.cos(phase) + 0.05 * rng.standard_normal(N)
            if abs(np.sum(h)) < 1e-6:
                h += 0.01
        elif start_idx == 2:
            # m-sequence like: use Legendre sequence
            h = np.ones(N)
            for i in range(1, N):
                # Simple pseudo-random based on bit manipulation
                h[i] = 1.0 if ((i * 7) % 13) < 6 else -1.0
            h = h + 0.02 * rng.standard_normal(N)
        elif start_idx == 3:
            # Fourier-domain: random magnitudes with flat-ish spectrum
            H = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            # Make magnitude more uniform
            mag = np.abs(H)
            mag = np.sqrt(mag + 1.0)  # compress dynamic range
            H = mag * np.exp(1j * np.angle(H))
            h = np.real(np.fft.ifft(H))
            if abs(np.sum(h)) < 1e-6:
                h += 0.01
        elif start_idx == 4:
            # BPSK-like with structured phases
            k = np.arange(N)
            h = np.sign(np.sin(2 * np.pi * k * 7 / N) + 0.1 * rng.standard_normal(N))
            h = h + 0.05 * rng.standard_normal(N)
            if abs(np.sum(h)) < 1e-6:
                h += 0.01
        elif start_idx == 5:
            # Two-tone interference
            k = np.arange(N)
            h = np.cos(2*np.pi*k*3/N) + 0.5*np.cos(2*np.pi*k*17/N)
            h = h + 0.05 * rng.standard_normal(N)
            if abs(np.sum(h)) < 1e-6:
                h += 0.01
        elif start_idx == 6:
            # Random with DC removal then re-add
            h = rng.standard_normal(N)
            h -= np.mean(h)  # zero mean
            h += 0.5  # add DC
            if abs(np.sum(h)) < 1e-6:
                h += 0.01
        else:
            # Pure random
            h = rng.standard_normal(N)
            if abs(np.sum(h)) < 1e-6:
                h += 0.01
        
        # Evaluate initial
        val = _objective(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Optimize from this starting point
        # Use L-BFGS-B with numerical gradient
        # First, try a few gradient descent steps with the smooth objective
        lr = 0.01
        for step in range(200):
            if time.time() > deadline:
                break
            
            h_curr = h.copy()
            s = np.sum(h_curr)
            if abs(s) < 1e-10:
                break
            
            conv = np.convolve(h_curr, h_curr)
            beta = 50.0
            max_c = np.max(conv)
            # Soft max gradient
            exps = np.exp(beta * (conv - max_c))
            probs = exps / np.sum(exps)
            
            # Gradient of soft_max w.r.t. conv
            # d(soft_max)/d(conv[j]) = probs[j]
            
            # conv[k] = sum_i h[i]*h[i+k] (for valid range)
            # d(conv[k])/dh[i] = h[i+k] + h[i-k] (where in bounds)
            # This is complex; use finite difference for gradient of smooth objective
            
            eps = 1e-6
            grad = np.zeros_like(h_curr)
            for i in range(0, N, 2):  # Sample every other for speed
                h_curr[i] += eps
                f1 = _smooth_objective(h_curr, beta)
                h_curr[i] -= 2*eps
                f2 = _smooth_objective(h_curr, beta)
                h_curr[i] += eps
                grad[i] = (f1 - f2) / (2*eps)
                # Mirror for odd indices (approximation)
                if i+1 < N:
                    grad[i+1] = grad[i]
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                h = h_curr - lr * grad / grad_norm * np.linalg.norm(h_curr)
            
            val = _objective(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
        
        # Try scipy optimization from best found so far
        if time.time() < deadline - 20.0:
            try:
                result = minimize(
                    _smooth_objective,
                    best_h,
                    args=(100.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                h_opt = result.x
                val_opt = _objective(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
        
        # Polish: local refinement around best
        if time.time() < deadline - 5.0:
            # Try small perturbations
            for polish_iter in range(50):
                if time.time() > deadline:
                    break
                h_try = best_h + 0.01 * rng.standard_normal(N)
                val_try = _objective(h_try)
                if val_try < best_val:
                    best_val = val_try
                    best_h = h_try.copy()
    
    # Final validation
    final_val = _objective(best_h)
    if final_val >= best_val or not np.isfinite(final_val):
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
