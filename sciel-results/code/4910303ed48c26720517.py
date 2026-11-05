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

    def objective_signed(params, tau=0.05):
        """Objective allowing signed h, with penalty if sum is small."""
        h = params
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    # Precompute the base square wave pattern: alternating +1 and -1
    # This gives a strong DC component when windowed
    base_square = np.ones(N)
    base_square[::2] = -1.0

    # Precompute polynomial basis: p(x) = sum_{k=0}^{K} a_k * x^k
    # where x = (2*i/N - 1) in [-1, 1]
    K_POLY = 8  # polynomial order
    x_poly = 2.0 * np.arange(N, dtype=float) / N - 1.0  # in [-1, 1]
    # Polynomial basis matrix: poly_basis[k, i] = x_i^k
    poly_basis = np.vander(x_poly, N=K_POLY + 1, increasing=True)  # shape (N, K_POLY+1)

    def poly_window_objective(coeffs, tau=0.05):
        """Objective for polynomial-windowed square wave.
        coeffs: [a_0, a_1, ..., a_K] where window(x) = sum_k a_k * x^k
        h[i] = base_square[i] * window(x_i)
        """
        window = poly_basis @ coeffs
        h = base_square * window
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    def poly_window_objective_tight(coeffs, tau=0.01):
        """Tighter version for final polish."""
        window = poly_basis @ coeffs
        h = base_square * window
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    # Build DCT-II basis matrix for additional spectral refinement
    i_idx = np.arange(N, dtype=float)
    k_idx = np.arange(N, dtype=float)
    dct_basis = np.cos(np.pi * np.outer(k_idx, (2*i_idx + 1)) / (2*N))
    dct_norm = np.sqrt(np.sum(dct_basis**2, axis=0))
    dct_basis = dct_basis / dct_norm

    best_h = np.ones(N)
    best_val = compute_value(best_h)

    # Phase 1: Optimize polynomial window coefficients
    if time.time() < deadline - 60.0:
        try:
            from scipy.optimize import minimize
            
            # Initial guess: constant window (all ones)
            c_init = np.zeros(K_POLY + 1)
            c_init[0] = 1.0  # constant term = 1
            
            result = minimize(
                poly_window_objective,
                c_init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            c_opt = result.x
            window_opt = poly_basis @ c_opt
            h_poly = base_square * window_opt
            val_poly = compute_value(h_poly)
            
            if val_poly < best_val and np.all(np.isfinite(h_poly)) and abs(np.sum(h_poly)) > 1e-6:
                best_h = h_poly.copy()
                best_val = val_poly
                print(f"Poly window: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Poly window error: {e}", file=sys.stderr)

    # Phase 2: Try different initializations for polynomial window
    if time.time() < deadline - 50.0:
        try:
            from scipy.optimize import minimize
            rng = np.random.RandomState(42)
            
            for restart in range(8):
                if time.time() > deadline - 40.0:
                    break
                
                # Random low-order polynomial initialization
                c_init = np.zeros(K_POLY + 1)
                c_init[0] = rng.uniform(0.5, 1.5)
                for k in range(1, min(4, K_POLY + 1)):
                    c_init[k] = rng.uniform(-0.3, 0.3)
                
                result = minimize(
                    poly_window_objective,
                    c_init,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                c_opt = result.x
                window_opt = poly_basis @ c_opt
                h_poly = base_square * window_opt
                val_poly = compute_value(h_poly)
                
                if val_poly < best_val and np.all(np.isfinite(h_poly)) and abs(np.sum(h_poly)) > 1e-6:
                    best_h = h_poly.copy()
                    best_val = val_poly
                    print(f"Poly restart {restart}: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Poly restart error: {e}", file=sys.stderr)

    # Phase 3: DCT-spectral refinement on the best result so far
    if time.time() < deadline - 40.0:
        try:
            from scipy.optimize import minimize
            
            # Initialize DCT coefficients from current best
            c_dct_init = dct_basis @ best_h
            
            def dct_objective(c, tau=0.05):
                h = dct_basis.T @ c
                s = np.sum(h)
                if abs(s) < 1e-4:
                    return 1e10 + (1e-4 - abs(s)) * 1e6
                conv = np.convolve(h, h)
                sm = smooth_max(conv, tau=tau)
                return 2.0 * N * sm / (s * s)
            
            result = minimize(
                dct_objective,
                c_dct_init,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = dct_basis.T @ result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"DCT refinement: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DCT refinement error: {e}", file=sys.stderr)

    # Phase 4: Local refinement in direct h-space
    if time.time() < deadline - 25.0:
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective_signed,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Direct L-BFGS-B: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Direct L-BFGS-B error: {e}", file=sys.stderr)

    # Phase 5: High-precision polish with tighter smooth max
    if time.time() < deadline - 10.0 and best_val < 2.0:
        try:
            from scipy.optimize import minimize
            
            result = minimize(
                lambda p: objective_signed(p, tau=0.01),
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Polish: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Polish error: {e}", file=sys.stderr)

    # Final safety check
    if time.time() > deadline - 2.0:
        return np.ones(N)

    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
