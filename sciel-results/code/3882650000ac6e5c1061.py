"""Optimized program for the third autocorrelation inequality task."""
import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    def compute_value(h):
        """Compute the objective value (lower is better)."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))
    
    def smooth_max(x, tau=0.01):
        """Soft approximation of max using log-sum-exp."""
        x_max = np.max(x)
        return x_max + tau * np.log(np.sum(np.exp((x - x_max) / tau)))
    
    def objective(params):
        """Objective for L-BFGS-B: minimize smooth max(conv) / sum(h)^2."""
        h = np.exp(params)  # Ensure positivity? No, we need signed values.
        # Better: use params directly as h, but constrain sum(h) away from 0
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10
        conv = np.convolve(h, h)
        # Use smooth max to make it differentiable
        sm = smooth_max(conv, tau=0.05)
        return 2.0 * N * sm / (s * s)
    
    def objective_signed(params):
        """Objective allowing signed h, with penalty if sum is small."""
        h = params
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=0.05)
        return 2.0 * N * sm / (s * s)
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Try multiple initializations with low-frequency structure
    rng = np.random.RandomState(42)
    
    for trial in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Low-frequency perturbation: h[i] = 1 + a * cos(2*pi*k*i/N) + b * sin(2*pi*k*i/N)
        k = rng.randint(1, 10)
        a = rng.uniform(-0.5, 0.5)
        b = rng.uniform(-0.5, 0.5)
        phase = rng.uniform(0, 2 * np.pi)
        h0 = 1.0 + a * np.cos(2 * np.pi * k * np.arange(N) / N + phase) + b * np.sin(2 * np.pi * k * np.arange(N) / N + phase)
        
        # Ensure sum is not too small
        s0 = np.sum(h0)
        if abs(s0) < 1.0:
            h0 = h0 + (1.0 - s0) / N
        
        # Run L-BFGS-B
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective_signed,
                h0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Trial {trial}: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Trial {trial}: error {e}", file=sys.stderr)
    
    # Final safety check
    if time.time() > deadline - 2.0:
        return np.ones(N)
    
    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)
    
# EVOLVE-BLOCK-END
