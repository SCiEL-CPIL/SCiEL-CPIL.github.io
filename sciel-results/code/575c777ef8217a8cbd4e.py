import numpy as np
import scipy.special
import time
from scipy.optimize import minimize_scalar, minimize

def construct():
    # Deadline guard: 630s limit, use 580s as safety margin
    start_time = time.time()
    deadline = start_time + 580.0

    def time_remaining():
        return deadline - time.time()

    try:
        m, n = 12, 1
        
        # Seed roots: [4, 8, ..., 48]
        seed_roots = np.array([4.0 * (i + 1) for i in range(m)])
        
        # Laguerre roots for L_12^(1)(x)
        lag_roots = scipy.special.roots_genlaguerre(m, 1)
        lag_roots = np.sort(lag_roots.real)

        # Fast proxy for the uncertainty bound
        # f(x) = exp(-x^2/4) * Q(x^2) where Q(y) = prod(y - r_i)
        # We want to minimize sigma_x * sigma_k
        
        def compute_bound(roots):
            """
            Fast computation of the uncertainty product sigma_x * sigma_k.
            Uses Gauss-Hermite for spatial moments and FFT for spectral moments.
            """
            roots = np.sort(np.asarray(roots, dtype=float))
            if np.any(roots <= 0):
                return 1e10
            
            # 1. Spatial Part: sigma_x
            # |f(x)|^2 = exp(-x^2/2) * Q(x^2)^2
            # Change variable u = x^2/2 => x = sqrt(2u), dx = du/sqrt(2u)
            # Integral is over u in [0, inf).
            # Weight is exp(-u) * (2u)^(-1/2) * (2u)^? 
            # Let's stick to direct Gauss-Hermite for stability and speed.
            # f(x) = exp(-x^2/4) Q(x^2)
            # |f(x)|^2 = exp(-x^2/2) Q(x^2)^2
            # Gauss-Hermite nodes x_i, weights w_i for weight exp(-x^2).
            # We need weight exp(-x^2/2). 
            # Let y = x/sqrt(2). Then exp(-x^2/2) dx = exp(-y^2) sqrt(2) dy.
            # So we can use standard GH nodes for y, scale back to x.
            
            # Use 64 nodes for good accuracy with degree 12 polynomial squared (deg 24)
            # GH with 64 nodes is exact for polynomials up to degree 127.
            y_nodes, y_weights = scipy.special.roots_hermite(64)
            x_nodes = y_nodes * np.sqrt(2.0)
            x_weights = y_weights * np.sqrt(2.0)
            
            # Compute Q(x^2)
            y_vals = x_nodes**2
            # Evaluate Q(y) = prod(y - r_i)
            # Use np.poly for stability? For 12 roots, it's fine.
            # Or direct product to avoid coeff instability?
            # Direct product is safer for evaluation.
            Q_vals = 1.0
            for r in roots:
                Q_vals *= (y_vals - r)
            
            f_sq = np.exp(-x_nodes**2 / 2.0) * Q_vals**2 * x_weights
            
            norm_sq = np.sum(f_sq)
            if norm_sq < 1e-15:
                return 1e10
            
            # sigma_x^2 = <x^2> = sum(x^2 * f_sq) / norm_sq
            mean_x2 = np.sum(x_nodes**2 * f_sq) / norm_sq
            sigma_x = np.sqrt(mean_x2)
            
            # 2. Spectral Part: sigma_k
            # Need Fourier transform of f(x).
            # f(x) is even.
            # Use FFT on a grid.
            # Range: roots up to ~50. x^2 up to 2500. exp(-x^2/4) decays fast.
            # x range [-20, 20] is sufficient.
            N = 2048
            x_grid = np.linspace(-20, 20, N)
            dx = x_grid[1] - x_grid[0]
            
            y_grid = x_grid**2
            Q_grid = 1.0
            for r in roots:
                Q_grid *= (y_grid - r)
            
            f_grid = np.exp(-x_grid**2 / 4.0) * Q_grid
            
            # FFT
            F = np.fft.fft(f_grid)
            F = np.fft.fftshift(F)
            k = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2.0 * np.pi
            
            F_sq = np.abs(F)**2
            # Normalize F_sq by dx^2? 
            # Parseval: sum(|f|^2) * dx = (1/N) sum(|F|^2) * (2*pi/dx)^-1?
            # Actually, we just need the ratio for sigma_k.
            # sigma_k^2 = <k^2> = sum(k^2 * |F|^2) / sum(|F|^2)
            
            norm_k = np.sum(F_sq)
            if norm_k < 1e-15:
                return 1e10
            
            mean_k2 = np.sum(k**2 * F_sq) / norm_k
            sigma_k = np.sqrt(mean_k2)
            
            return sigma_x * sigma_k

        # Strategy 1: Optimize scaling of Laguerre roots
        # roots = s * lag_roots
        def bound_scaled_lag(s):
            if s <= 0:
                return 1e10
            return compute_bound(s * lag_roots)

        # Search for optimal s
        # Seed roots are roughly 4..48. Laguerre roots are roughly 1..25.
        # So s should be around 4/1 to 48/25 ~ 2.
        # Let's search s in [0.5, 5.0]
        res_s = minimize_scalar(bound_scaled_lag, bounds=(0.5, 5.0), method='bounded', options={'xatol': 1e-4})
        
        best_roots = seed_roots
        best_score = compute_bound(seed_roots)
        
        if res_s.success:
            s_opt = res_s.x
            candidate_roots = s_opt * lag_roots
            candidate_score = res_s.fun
            if candidate_score < best_score:
                best_roots = candidate_roots
                best_score = candidate_score

        # Strategy 2: Local refinement of the best roots found so far
        # Use Nelder-Mead with a limited number of iterations to avoid timeout
        # and to escape local minima if the scaling wasn't enough.
        
        # Check time
        if time_remaining() > 10.0:
            # Try refining the best roots found
            # We need to ensure roots stay positive.
            # Nelder-Mead doesn't handle bounds well, so we'll penalize negative roots.
            
            def bound_with_penalty(roots):
                if np.any(roots <= 0):
                    return 1e10 + np.sum(roots[roots <= 0]**2)
                return compute_bound(roots)
            
            # Start from best_roots
            # Use a smaller maxiter to stay within time
            res_nm = minimize(
                bound_with_penalty,
                best_roots,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if res_nm.fun < best_score:
                best_roots = res_nm.x
                best_score = res_nm.fun

        # Final verification and formatting
        best_roots = np.sort(np.maximum(best_roots, 1e-6))
        
        if len(best_roots) != m:
            return [float(r) for r in seed_roots]
            
        return [float(r) for r in best_roots]

    except Exception as e:
        # Fallback to seed
        return [4.0 * (i + 1) for i in range(12)]
