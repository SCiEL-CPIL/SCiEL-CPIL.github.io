"""Optimized program for the third autocorrelation inequality task."""
import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
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
    
    def chirp_init(m, shift=0):
        """Generate chirp-based initialization.
        h[i] = cos(2*pi*m*i*(i+1)/(2*N) + shift)
        This creates a quadratic phase that gives near-constant autoconvolution."""
        i = np.arange(N)
        phase = np.pi * m * i * (i + 1) / N + shift
        return np.cos(phase)
    
    def objective_with_penalty(h, tau=0.03):
        """Objective with smooth max and sum penalty."""
        s = np.sum(h)
        if abs(s) < 1e-3:
            return 1e10
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        val = 2.0 * N * sm / (s * s)
        # Add small penalty for large |h| to keep values reasonable
        val += 1e-6 * np.sum(h**2)
        return val
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Phase 1: Chirp-based initialization with parameter sweep
    # Try different chirp parameters m (integer) and phase shifts
    if time.time() < deadline - 50.0:
        for m in range(1, 20):
            if time.time() > deadline - 45.0:
                break
            for shift_frac in np.linspace(0, 2*np.pi, 8, endpoint=False):
                if time.time() > deadline - 45.0:
                    break
                h_chirp = chirp_init(m, shift_frac)
                # Ensure sum is not too small
                s = np.sum(h_chirp)
                if abs(s) < 1.0:
                    h_chirp = h_chirp + (1.0 - s) / N
                val = compute_value(h_chirp)
                if val < best_val:
                    best_h = h_chirp.copy()
                    best_val = val
                    print(f"Chirp m={m}, shift={shift_frac:.3f}: {best_val:.6f}", file=sys.stderr)
    
    # Phase 2: Simulated annealing in the time domain
    # This is a structural change: SA can explore the landscape more broadly
    if time.time() < deadline - 15.0:
        rng = np.random.RandomState(42)
        h_sa = best_h.copy()
        val_sa = compute_value(h_sa)
        T = 0.1  # Initial temperature
        T_min = 1e-6
        cooling = 0.999
        max_iter = 5000
        
        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            if T < T_min:
                break
            
            # Generate perturbation: random subset of elements
            n_perturb = rng.randint(5, 50)
            indices = rng.choice(N, n_perturb, replace=False)
            delta = rng.normal(0, 0.1, n_perturb)
            
            h_new = h_sa.copy()
            h_new[indices] += delta
            
            # Ensure sum constraint is approximately maintained
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-3:
                h_new = h_new + (1.0 - s_new) / N
            
            val_new = compute_value(h_new)
            
            if np.isfinite(val_new):
                delta_val = val_new - val_sa
                if delta_val < 0 or rng.random() < np.exp(-delta_val / T):
                    h_sa = h_new
                    val_sa = val_new
                    if val_sa < best_val:
                        best_h = h_sa.copy()
                        best_val = val_sa
                        print(f"SA iter {it}: improved to {best_val:.6f}", file=sys.stderr)
            
            T *= cooling
    
    # Phase 3: Gradient-based refinement from best SA result
    if time.time() < deadline - 5.0:
        try:
            from scipy.optimize import minimize
            
            def obj_for_minimize(params):
                h = params
                s = np.sum(h)
                if abs(s) < 1e-3:
                    return 1e10
                conv = np.convolve(h, h)
                # Use a tighter smooth max for final refinement
                sm = smooth_max(conv, tau=0.01)
                return 2.0 * N * sm / (s * s)
            
            result = minimize(
                obj_for_minimize,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 20}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Final L-BFGS-B: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"L-BFGS-B error: {e}", file=sys.stderr)
    
    # Phase 4: Try a few more chirp initializations with refinement
    if time.time() < deadline - 3.0:
        rng2 = np.random.RandomState(123)
        for trial in range(5):
            if time.time() > deadline - 3.0:
                break
            m = rng2.randint(1, 30)
            shift = rng2.uniform(0, 2*np.pi)
            h0 = chirp_init(m, shift)
            s0 = np.sum(h0)
            if abs(s0) < 1.0:
                h0 = h0 + (1.0 - s0) / N
            
            try:
                from scipy.optimize import minimize
                def obj_trial(params):
                    h = params
                    s = np.sum(h)
                    if abs(s) < 1e-3:
                        return 1e10
                    conv = np.convolve(h, h)
                    sm = smooth_max(conv, tau=0.02)
                    return 2.0 * N * sm / (s * s)
                
                result = minimize(
                    obj_trial,
                    h0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                h_opt = result.x
                val_opt = compute_value(h_opt)
                if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                    best_h = h_opt.copy()
                    best_val = val_opt
                    print(f"Chirp trial {trial} m={m}: improved to {best_val:.6f}", file=sys.stderr)
            except Exception as e:
                pass
    
    # Final safety check
    if time.time() > deadline - 2.0:
        return np.ones(N)
    
    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)
    
# EVOLVE-BLOCK-END
