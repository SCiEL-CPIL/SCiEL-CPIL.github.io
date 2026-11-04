# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = 195  # seconds

    # Precompute weights W_k = sqrt(pi) * 2^k * k!
    # k in {0, 4, 8, 12}
    ks = [0, 4, 8, 12]
    W = {}
    for k in ks:
        # 2^k * k!
        val = (2 ** k) * math.factorial(k)
        W[k] = val * math.sqrt(math.pi)
    
    # Indices for d_m calculation
    # m in {1, 3, 5, 7, 9, 11, 13}
    ms = [1, 3, 5, 7, 9, 11, 13]
    
    # Precompute c_k access helper
    # c is dict or array
    # We will pass c0, c4, c8. c12 is derived.
    
    def compute_bound_analytic(c0, c4, c8):
        c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 40320.0
        
        # Get coefficients
        coeffs = {
            0: c0,
            4: c4,
            8: c8,
            12: c12
        }
        
        N = 0.0
        X = 0.0
        for k in ks:
            ck = coeffs[k]
            wk = W[k]
            N += ck * ck * wk
            X += ck * ck * (k + 0.5) * wk
        
        if N <= 0 or not np.isfinite(N):
            return np.inf
            
        K = 0.0
        for m in ms:
            # d_m = (m+1) c_{m+1} - 0.5 c_{m-1}
            cp1 = coeffs.get(m + 1, 0.0)
            cm1 = coeffs.get(m - 1, 0.0)
            dm = (m + 1) * cp1 - 0.5 * cm1
            wm = W.get(m, 0.0) # W is defined for 0,4,8,12. m are 1,3,5,7,9,11,13.
            # Wait, W_m is needed.
            # W_m = sqrt(pi) * 2^m * m!
            # We didn't precompute W for odd m.
            # Let's compute on the fly or precompute for all m in ms.
            
        # Need to precompute W for ms as well
        # Let's do it in the outer scope or re-calc.
        # For speed, precompute W for all needed m.
        
        return np.inf # Placeholder

    # Re-do precomputation to include W for ms
    W_all = {}
    for k in ks:
        W_all[k] = (2 ** k) * math.factorial(k) * math.sqrt(math.pi)
    for m in ms:
        W_all[m] = (2 ** m) * math.factorial(m) * math.sqrt(math.pi)

    def compute_bound_analytic(c0, c4, c8):
        c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 40320.0
        
        coeffs = {
            0: c0,
            4: c4,
            8: c8,
            12: c12
        }
        
        N = 0.0
        X = 0.0
        for k in ks:
            ck = coeffs[k]
            wk = W_all[k]
            N += ck * ck * wk
            X += ck * ck * (k + 0.5) * wk
        
        if N <= 0 or not np.isfinite(N):
            return np.inf
            
        K = 0.0
        for m in ms:
            cp1 = coeffs.get(m + 1, 0.0)
            cm1 = coeffs.get(m - 1, 0.0)
            dm = (m + 1) * cp1 - 0.5 * cm1
            wm = W_all[m]
            K += dm * dm * wm
        
        if K < 0 or not np.isfinite(K):
            return np.inf
            
        return math.sqrt(X * K) / N

    def compute_bound_from_spherical(theta, phi):
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        return compute_bound_analytic(c0, c4, c8)

    # Initial best guess from seed
    c_seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    c_seed_norm = c_seed / np.linalg.norm(c_seed)
    c0, c4, c8 = c_seed_norm
    theta_init = np.arccos(np.clip(c8, -0.999999, 0.999999))
    phi_init = np.arctan2(c4, c0)
    if phi_init < 0:
        phi_init += 2 * np.pi

    best_bound = compute_bound_from_spherical(theta_init, phi_init)
    best_theta = theta_init
    best_phi = phi_init

    # Global search
    rng = np.random.RandomState(42)
    starts = [(theta_init, phi_init)]
    
    for _ in range(50):
        t = rng.uniform(0.01, np.pi - 0.01)
        p = rng.uniform(0.01, 2 * np.pi - 0.01)
        starts.append((t, p))
        
    for t in np.linspace(0.2, np.pi - 0.2, 20):
        for p in np.linspace(0.1, 2 * np.pi - 0.1, 20):
            starts.append((t, p))

    start_scores = []
    for t, p in starts:
        b = compute_bound_from_spherical(t, p)
        if np.isfinite(b) and b < best_bound:
            best_bound = b
            best_theta, best_phi = t, p
        start_scores.append((b, t, p))
    
    start_scores.sort(key=lambda x: x[0] if np.isfinite(x[0]) else np.inf)
    top_starts = [(t, p) for _, t, p in start_scores[:30] if np.isfinite(compute_bound_from_spherical(t, p))]
    
    if (best_theta, best_phi) not in top_starts:
        top_starts.append((best_theta, best_phi))

    # Local refinement
    for t, p in top_starts:
        if time.time() - start_time >= deadline - 30:
            break
        try:
            res = minimize(
                compute_bound_from_spherical, 
                x0=[t, p], 
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 50000, 'adaptive': True, 'maxfev': 100000}
            )
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_theta, best_phi = res.x[0], res.x[1]
        except Exception:
            pass

    # SLSQP
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(
                compute_bound_from_spherical, 
                x0=[best_theta, best_phi],
                method='SLSQP',
                bounds=[(0.001, np.pi - 0.001), (0.001, 2 * np.pi - 0.001)],
                options={'maxiter': 10000, 'ftol': 1e-18, 'disp': False, 'maxfev': 50000}
            )
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_theta, best_phi = res.x[0], res.x[1]
        except Exception:
            pass

    # Final polish
    if time.time() - start_time < deadline - 3:
        try:
            res = minimize(
                compute_bound_from_spherical, 
                x0=[best_theta, best_phi], 
                method='Nelder-Mead',
                options={'xatol': 1e-15, 'fatol': 1e-18, 'maxiter': 100000, 'adaptive': True, 'maxfev': 200000}
            )
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_theta, best_phi = res.x[0], res.x[1]
        except Exception:
            pass

    c0 = np.sin(best_theta) * np.cos(best_phi)
    c4 = np.sin(best_theta) * np.sin(best_phi)
    c8 = np.cos(best_theta)
    
    return np.array([c0, c4, c8])

# EVOLVE-BLOCK-END
