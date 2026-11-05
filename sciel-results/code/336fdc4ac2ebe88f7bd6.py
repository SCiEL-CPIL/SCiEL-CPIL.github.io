"""Hinge-loss autoconvolution minimizer for the third autocorrelation inequality."""
import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    best_h = np.ones(N).astype(float)
    best_val = compute_value(best_h)  # 2.0

    # ---- Phase 1: Hinge-loss optimization via alternating linearization ----
    # Objective: minimize  2*N*max(conv)/s^2
    # Reformulate: minimize  2*N*t / s^2  s.t. conv[k] <= t for all k
    # Alternating: fix s, minimize t s.t. conv <= t (LP-like via hinge)
    # Fix t, maximize s^2 (or minimize -s^2) s.t. conv <= t
    
    # Use hinge loss: L = sum_k max(0, conv[k] - t)^2 + lambda * t
    # This is differentiable almost everywhere; use L-BFGS-B
    
    def hinge_objective(params, t_target):
        h = params
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10
        conv = np.convolve(h, h)
        # Hinge: penalize conv entries above t_target
        excess = np.maximum(0.0, conv - t_target)
        hinge = np.sum(excess ** 2)
        # Also penalize t_target being too large (proxy for max)
        # Normalize: we want to minimize max(conv)/s^2
        # Use ratio directly with hinge on normalized conv
        norm_conv = conv / (s * s)
        excess_norm = np.maximum(0.0, norm_conv - 2.0 * t_target / N)
        return np.sum(excess_norm ** 2) + 0.01 * t_target

    # Better approach: direct hinge on the ratio
    def hinge_ratio_objective(params, threshold):
        h = params
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10
        conv = np.convolve(h, h)
        ratio_conv = conv / (s * s)
        # We want max(ratio_conv) <= threshold
        # Hinge loss on ratio_conv - threshold
        excess = np.maximum(0.0, ratio_conv - threshold)
        return np.sum(excess ** 2)

    # Phase 1a: Reduce threshold iteratively with hinge loss
    try:
        from scipy.optimize import minimize
        
        current_threshold = best_val / (2.0 * N)  # = 1/(2N) * best_val... 
        # Actually: value = 2*N*max(conv)/s^2, so max(conv)/s^2 = value/(2*N)
        current_threshold = best_val / (2.0 * N)
        
        for iteration in range(30):
            if time.time() > deadline - 40.0:
                break
            
            def obj(params):
                h = params
                s = np.sum(h)
                if abs(s) < 1e-4:
                    return 1e10
                conv = np.convolve(h, h)
                ratio_conv = conv / (s * s)
                excess = np.maximum(0.0, ratio_conv - current_threshold)
                return np.sum(excess ** 2)
            
            result = minimize(
                obj,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            
            h_opt = result.x
            val_opt = compute_value(h_opt)
            
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                # Tighten threshold
                current_threshold = best_val / (2.0 * N) * 0.95
                print(f"Hinge iter {iteration}: val={best_val:.6f}, thresh={current_threshold:.8f}", file=sys.stderr)
            else:
                # No improvement; tighten threshold slightly more aggressively
                current_threshold *= 0.90
                # Try from a perturbed start
                rng = np.random.RandomState(iteration * 7 + 3)
                h_perturbed = best_h + 0.01 * rng.randn(N)
                result2 = minimize(
                    obj,
                    h_perturbed,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                h_opt2 = result2.x
                val_opt2 = compute_value(h_opt2)
                if val_opt2 < best_val and np.all(np.isfinite(h_opt2)) and abs(np.sum(h_opt2)) > 1e-6:
                    best_h = h_opt2.copy()
                    best_val = val_opt2
                    current_threshold = best_val / (2.0 * N) * 0.95
                    print(f"Hinge iter {iteration} (perturbed): val={best_val:.6f}", file=sys.stderr)
                else:
                    # Stuck, break
                    break
    except Exception as e:
        print(f"Hinge phase error: {e}", file=sys.stderr)

    # ---- Phase 2: Spectral method - minimize peak via frequency-domain shaping ----
    # The autoconvolution peak is related to the L1 norm of the Fourier transform
    # |conv|_inf <= |F|^2 in some sense. Try to shape h to have flat spectrum.
    # Use: h = IFFT(|H|^2 * phase) where we optimize the phase to minimize peak
    try:
        if time.time() < deadline - 30.0:
            # Start from current best_h
            h_work = best_h.copy()
            s_work = np.sum(h_work)
            
            for spec_iter in range(20):
                if time.time() > deadline - 25.0:
                    break
                
                H = np.fft.fft(h_work)
                # Magnitude spectrum
                mag = np.abs(H)
                phase = np.angle(H)
                
                # Try to flatten the magnitude spectrum (reduce concentration)
                # by applying a soft threshold / compression
                # This reduces the peak in time domain of |H|^2
                mag_compressed = np.sign(mag) * np.sqrt(mag)  # compress dynamic range
                # Or: mag_compressed = mag / (1 + mag) 
                
                # Reconstruct with compressed magnitude
                h_new = np.fft.ifft(mag_compressed * np.exp(1j * phase)).real
                
                # Ensure real and reasonable
                if np.all(np.isfinite(h_new)):
                    val_new = compute_value(h_new)
                    if val_new < best_val:
                        best_h = h_new.copy()
                        best_val = val_new
                        print(f"Spectral iter {spec_iter}: val={best_val:.6f}", file=sys.stderr)
                    else:
                        # Try mild compression
                        for alpha in [0.7, 0.5, 0.3]:
                            mag_c = np.sign(mag) * (mag ** alpha)
                            h_try = np.fft.ifft(mag_c * np.exp(1j * phase)).real
                            if np.all(np.isfinite(h_try)):
                                val_try = compute_value(h_try)
                                if val_try < best_val:
                                    best_h = h_try.copy()
                                    best_val = val_try
                                    print(f"Spectral iter {spec_iter} alpha={alpha}: val={best_val:.6f}", file=sys.stderr)
                                    break
    except Exception as e:
        print(f"Spectral phase error: {e}", file=sys.stderr)

    # ---- Phase 3: Coordinate descent with exact max tracking ----
    # Directly minimize the true objective using coordinate descent
    try:
        if time.time() < deadline - 15.0:
            h_cd = best_h.copy()
            val_cd = best_val
            
            for cd_iter in range(5000):
                if time.time() > deadline - 12.0:
                    break
                
                # Pick a coordinate (cycling or random)
                i = cd_iter % N
                h_old = h_cd[i]
                
                # Try small perturbations
                best_delta = 0.0
                best_val_i = val_cd
                for delta in [-0.5, -0.1, -0.01, 0.01, 0.1, 0.5]:
                    h_cd[i] = h_old + delta
                    s = np.sum(h_cd)
                    if abs(s) < 1e-6:
                        h_cd[i] = h_old
                        continue
                    conv = np.convolve(h_cd, h_cd)
                    v = abs(2.0 * N * np.max(conv) / (s * s))
                    if v < best_val_i:
                        best_val_i = v
                        best_delta = delta
                h_cd[i] = h_old + best_delta
                val_cd = best_val_i
                
                if val_cd < best_val:
                    best_h = h_cd.copy()
                    best_val = val_cd
                    print(f"CD iter {cd_iter}: val={best_val:.6f}", file=sys.stderr)
    except Exception as e:
        print(f"CD phase error: {e}", file=sys.stderr)

    # ---- Phase 4: Final polish with tight L-BFGS on true max via smooth approximation ----
    try:
        if time.time() < deadline - 5.0 and best_val < 2.0:
            from scipy.optimize import minimize
            
            def final_objective(params):
                h = params
                s = np.sum(h)
                if abs(s) < 1e-4:
                    return 1e10
                conv = np.convolve(h, h)
                # Very tight smooth max
                x_max = np.max(conv)
                tau = 0.001
                sm = x_max + tau * np.log(np.sum(np.exp((conv - x_max) / tau)))
                return 2.0 * N * sm / (s * s)
            
            result = minimize(
                final_objective,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 30}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Final polish: val={best_val:.6f}", file=sys.stderr)
    except Exception as e:
        print(f"Final polish error: {e}", file=sys.stderr)

    # Safety check
    if time.time() > deadline - 2.0:
        return np.ones(N)

    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
