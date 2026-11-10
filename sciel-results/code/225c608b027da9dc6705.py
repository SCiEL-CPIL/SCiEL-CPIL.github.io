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
    """
    N = 8
    deadline = time.time() + 580.0

    i_vals = np.arange(N, dtype=np.float64)
    shear = i_vals / N  # shape (N,)
    width = 1.0 / N

    def union_len_at(c, x):
        """Compute 1D union length at height c for base positions x."""
        L = x + c * shear
        R = L + width
        order = np.argsort(L)
        L_s = L[order]
        R_s = R[order]
        # Merge intervals
        max_r = -np.inf
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
        # Build arrays of coefficients: E_p(c) = a_p * c + b_p
        a_coeff = np.concatenate([shear, shear])  # shape (2N,)
        b_coeff = np.concatenate([x, x + width])   # shape (2N,)
        
        # Find all c where any two endpoints are equal
        breakpoints = {0.0, 1.0}
        
        # Vectorize: for each pair (p,q), c = (b_q - b_p) / (a_p - a_q)
        # Use broadcasting to compute all pairs at once
        a_diff = a_coeff[:, None] - a_coeff[None, :]  # shape (2N, 2N)
        b_diff = b_coeff[None, :] - b_coeff[:, None]  # shape (2N, 2N)
        
        # Only consider upper triangle (p < q)
        mask = np.triu(np.ones((2*N, 2*N), dtype=bool), k=1)
        valid = mask & (np.abs(a_diff) > 1e-15)
        
        c_vals = np.zeros((2*N, 2*N))
        c_vals[valid] = b_diff[valid] / a_diff[valid]
        
        # Keep only those in [0, 1]
        in_range = valid & (c_vals >= 0.0) & (c_vals <= 1.0)
        for c_val in c_vals[in_range]:
            breakpoints.add(float(c_val))
        
        breakpoints = np.sort(np.array(list(breakpoints), dtype=np.float64))
        
        if len(breakpoints) < 2:
            return 0.0
        
        # Integrate U(c)^2 over each subinterval
        total_integral = 0.0
        for idx in range(len(breakpoints) - 1):
            c0 = breakpoints[idx]
            c1 = breakpoints[idx + 1]
            
            if c1 - c0 < 1e-15:
                continue
            
            u0 = union_len_at(c0, x)
            u1 = union_len_at(c1, x)
            
            # U(c) is linear: U(c) = m*c + b
            m = (u1 - u0) / (c1 - c0)
            b = u0 - m * c0
            
            # Integral of (m*c + b)^2 from c0 to c1
            integral = (m ** 2) * (c1 ** 3 - c0 ** 3) / 3.0 + \
                       (2 * m * b) * (c1 ** 2 - c0 ** 2) / 2.0 + \
                       (b ** 2) * (c1 - c0)
            
            total_integral += integral
        
        return total_integral

    # Initial guesses
    k = np.arange(N, dtype=np.float64)
    inits = [
        -k / (2.0 * N),
        -k / (2.0 * N) * 0.5,
        -k / (2.0 * N) * 1.5,
        np.linspace(0, 1, N) - 0.5,
        np.linspace(-0.5, 0.5, N),
        np.sin(np.linspace(0, np.pi, N)) * 0.3 - 0.15,
        -np.cos(np.linspace(0, np.pi, N)) * 0.25,
        np.random.default_rng(42).uniform(-0.5, 0.5, N),
        np.random.default_rng(43).uniform(-0.3, 0.3, N),
    ]

    best_x = inits[0].copy()
    best_val = exact_objective(best_x)

    # Try multiple initializations with Nelder-Mead
    for x_start in inits[1:]:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(exact_objective, x_start, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if np.all(np.isfinite(res.x)) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Refine with tighter tolerance
    if time.time() < deadline - 30:
        try:
            res = minimize(exact_objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14,
                                    'adaptive': True})
            if np.all(np.isfinite(res.x)) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Final polish with even tighter tolerance
    if time.time() < deadline - 15:
        try:
            res = minimize(exact_objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-13, 'fatol': 1e-15,
                                    'adaptive': True})
            if np.all(np.isfinite(res.x)) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
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
