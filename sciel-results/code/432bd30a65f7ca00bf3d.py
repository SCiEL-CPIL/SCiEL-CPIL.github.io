# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Discrete 3-D Kakeya set: union of 64 sheared tubes (N = 8).
    
    Strategy:
    1. Exploit symmetry: set x_i = y_i. Then V = integral_0^1 Ux(c)^2 dc.
    2. The 1D union length U(c) is a piecewise-linear function of c with 
       breakpoints at values where the ordering of interval endpoints changes.
    3. Compute the exact integral of U(c)^2 by finding all breakpoints and 
       integrating piecewise.
    4. Use gradient-free optimization (Nelder-Mead) with a fast, exact objective.
    5. Refine with coordinate descent and high-precision Nelder-Mead polish.
    """
    N = 8
    deadline = time.time() + 580.0

    i_vals = np.arange(N, dtype=np.float64)
    shear = i_vals / N  # shape (N,)
    width = 1.0 / N

    def union_len_vec(x, c):
        """Vectorized 1D union length at a single c value."""
        L = x + c * shear
        R = L + width
        order = np.argsort(L)
        L_s = L[order]
        R_s = R[order]
        max_r = L_s[0]
        u_len = 0.0
        for ii in range(N):
            if L_s[ii] > max_r:
                u_len += R_s[ii] - L_s[ii]
                max_r = R_s[ii]
            else:
                if R_s[ii] > max_r:
                    max_r = R_s[ii]
        return u_len

    def exact_objective(x):
        """
        Compute integral_0^1 U(c)^2 dc exactly, where U(c) is the 1D union length 
        of intervals [x_i + c * shear_i, x_i + c * shear_i + width].
        """
        a_coeff = np.concatenate([shear, shear])
        b_coeff = np.concatenate([x, x + width])
        
        breakpoints = set()
        breakpoints.add(0.0)
        breakpoints.add(1.0)
        
        for p in range(2 * N):
            for q in range(p + 1, 2 * N):
                da = a_coeff[p] - a_coeff[q]
                db = b_coeff[p] - b_coeff[q]
                if abs(da) < 1e-15:
                    continue
                c_val = -db / da
                if 0.0 <= c_val <= 1.0:
                    breakpoints.add(c_val)
        
        breakpoints = np.sort(np.array(list(breakpoints), dtype=np.float64))
        
        if len(breakpoints) < 2:
            return 0.0
        
        total_integral = 0.0
        for idx in range(len(breakpoints) - 1):
            c0 = breakpoints[idx]
            c1 = breakpoints[idx + 1]
            
            if c1 - c0 < 1e-15:
                continue
            
            u0 = union_len_vec(x, c0)
            u1 = union_len_vec(x, c1)
            
            m = (u1 - u0) / (c1 - c0)
            b = u0 - m * c0
            
            integral = (m ** 2) * (c1 ** 3 - c0 ** 3) / 3.0 + \
                       (2 * m * b) * (c1 ** 2 - c0 ** 2) / 2.0 + \
                       (b ** 2) * (c1 - c0)
            
            total_integral += integral
        
        return total_integral

    def coordinate_descent(x, max_iter=100, tol=1e-12):
        """Simple coordinate descent refinement with adaptive step sizes."""
        best_x = x.copy()
        best_val = exact_objective(best_x)
        
        step_sizes = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
        
        for it in range(max_iter):
            if time.time() > deadline - 30:
                break
            improved = False
            for d in range(N):
                for delta in step_sizes:
                    for sign in [1.0, -1.0]:
                        trial = best_x.copy()
                        trial[d] += sign * delta
                        val = exact_objective(trial)
                        if val < best_val - tol:
                            best_x = trial
                            best_val = val
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break
        
        return best_x, best_val

    # Initial guess: centered symmetric configuration
    k = np.arange(N, dtype=np.float64)
    x0 = -k / (2.0 * N)

    # Optimize using Nelder-Mead
    best_x = x0.copy()
    best_val = exact_objective(x0)

    # Try multiple initializations
    inits = [
        x0.copy(),
        -k / (2.0 * N) * 0.5,
        -k / (2.0 * N) * 1.5,
        np.linspace(0, 1, N) - 0.5,
        np.linspace(-0.5, 0.5, N),
        -k / (2.0 * N) * 0.8,
        -k / (2.0 * N) * 1.2,
        # Additional diverse initializations
        np.sort(np.random.default_rng(42).uniform(-0.5, 0.5, N)),
        np.sort(np.random.default_rng(43).uniform(-0.3, 0.3, N)),
        -k / N * 0.7,
        -k / N * 1.3,
        -k / N * 1.0,
    ]

    for x_start in inits:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(exact_objective, x_start, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if np.all(np.isfinite(res.x)) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Coordinate descent refinement
    if time.time() < deadline - 20:
        try:
            best_x, best_val = coordinate_descent(best_x, max_iter=200)
        except Exception:
            pass

    # Final high-precision polish pass
    if time.time() < deadline - 15:
        try:
            res = minimize(exact_objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16,
                                    'adaptive': True})
            if np.all(np.isfinite(res.x)) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
        
        # Even tighter final polish if time permits
        if time.time() < deadline - 5:
            try:
                res = minimize(exact_objective, best_x, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-15, 'fatol': 1e-17,
                                        'adaptive': True})
                if np.all(np.isfinite(res.x)) and res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                pass

    # Construct the 64x2 array
    positions = np.zeros((64, 2), dtype=np.float64)
    for k_idx in range(64):
        i = k_idx // 8
        j = k_idx % 8
        positions[k_idx, 0] = best_x[i]
        positions[k_idx, 1] = best_x[j]

    return positions

# EVOLVE-BLOCK-END
