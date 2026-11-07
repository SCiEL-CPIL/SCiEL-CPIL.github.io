"""Rewritten program using a parametric root-construction approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """Return three coefficients [c0, c1, c2] that minimize the value."""
    t0 = time.time()
    deadline = t0 + 200  # Safety margin before 210s kill

    # Constants
    H4_0 = 12.0
    H8_0 = 1344.0
    H12_0 = 1032192.0

    # Coefficients of Q(t) = g(x)/x^2 = a5*t^5 + a4*t^4 + a3*t^3 + a2*t^2 + a1*t + a0
    # a0 = -48*c1 - 13440*c2 - 5505024*c3
    # a1 = 16*c1 + 11520*c2 + 9953280*c3
    # a2 = -3072*c2 - 5806080*c3
    # a3 = 256*c2 + 1290240*c3
    # a4 = -122880*c3
    # a5 = 4096*c3
    # with c3 = -(c0 + 12*c1 + 1344*c2)/1032192

    # Parametric approach:
    # Let Q(t) = (t - r1)(t - r2)(t - r3)(t - r4)(t - r5) * k + lower_terms
    # But we have 5 free parameters (c0,c1,c2 determine c3, giving 3 free params for 5 coefficients of Q)
    # Actually Q is degree 5 with 6 coefficients, but c3 is determined, so 3 free params.
    # 
    # Alternative: Fix the roots of Q(t) for t>0 and solve for coefficients.
    # Q(t) = a5 * (t - s1)(t - s2)(t - s3)(t - s4)(t - s5) where s_i > 0
    # But Q has specific structure: a4/a5 = -122880/4096 = -30, a3/a5 = 1290240/4096 = 315, etc.
    # So Q(t)/a5 = t^5 - 30t^4 + 315t^3 + (a2/a5)t^2 + (a1/a5)t + (a0/a5)
    # And a2/a5 = (-3072*c2 - 5806080*c3)/4096*c3 = -3072*c2/(4096*c3) - 5806080/4096 = -0.75*c2/c3 - 1417.5
    # This is getting complex. Let me try a different approach.
    
    # Direct optimization with better method: use scipy.optimize with bounds and multiple starts
    from scipy.optimize import minimize, differential_evolution
    
    def compute_value(c0, c1, c2):
        """Compute r^2/(2*pi) for given coefficients."""
        c3 = -(c0 + H4_0 * c1 + H8_0 * c2) / H12_0
        
        # Check sign at infinity: g(x)/x^2 ~ 4096*c3*x^10, so if c3 < 0, g -> -inf
        # In that case, g is replaced by -g, so we use -c3, -c1, -c2, -c0
        # But since value depends on r^2, and the roots are the same (just signs flipped in Q),
        # let's just handle it: if c3 < 0, flip all coefficients
        if c3 < 0:
            c0, c1, c2, c3 = -c0, -c1, -c2, -c3
        
        a0 = -48*c1 - 13440*c2 - 5505024*c3
        a1 = 16*c1 + 11520*c2 + 9953280*c3
        a2 = -3072*c2 - 5806080*c3
        a3 = 256*c2 + 1290240*c3
        a4 = -122880*c3
        a5 = 4096*c3
        
        coeffs = [a5, a4, a3, a2, a1, a0]
        
        try:
            roots_t = np.roots(coeffs)
        except:
            return 1e10
        
        real_pos_roots = []
        for rt in roots_t:
            if abs(rt.imag) < 1e-10:
                tr = rt.real
                if tr > 1e-12:
                    real_pos_roots.append(tr)
        
        if not real_pos_roots:
            return 1e10
        
        # Find roots where Q changes sign
        best_r2 = 1e10
        for tr in real_pos_roots:
            eps = 1e-8 * max(1.0, abs(tr))
            Q_minus = np.polyval(coeffs, tr - eps)
            Q_plus = np.polyval(coeffs, tr + eps)
            if Q_minus * Q_plus < 0:
                if tr < best_r2:
                    best_r2 = tr
        
        if best_r2 >= 1e10:
            return 1e10
        
        return best_r2 / (2.0 * np.pi)
    
    # Seed
    seed = np.array([-113/100, 1/25, 1/3240])
    best_coeffs = seed.copy()
    best_val = compute_value(seed[0], seed[1], seed[2])
    
    # Strategy 1: Differential evolution with wider bounds
    if time.time() < deadline - 80:
        try:
            bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
            de_result = differential_evolution(
                lambda x: compute_value(x[0], x[1], x[2]),
                bounds, maxiter=100, popsize=25, seed=123,
                tol=1e-12, polish=False, init='latinhypercube'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_coeffs = de_result.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Multiple random restarts with Nelder-Mead
    if time.time() < deadline - 20:
        rng = np.random.default_rng(456)
        for i in range(50):
            if time.time() > deadline - 15:
                break
            # Random start near the best found so far
            if i == 0:
                x0 = best_coeffs.copy()
            else:
                scale = 0.5 * (1 - i/50) + 0.05
                x0 = best_coeffs + scale * rng.standard_normal(3)
            
            try:
                nm_result = minimize(
                    lambda x: compute_value(x[0], x[1], x[2]),
                    x0, method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
                )
                if nm_result.fun < best_val:
                    best_val = nm_result.fun
                    best_coeffs = nm_result.x.copy()
            except Exception:
                pass
    
    # Final refinement
    if time.time() < deadline - 5:
        try:
            nm_final = minimize(
                lambda x: compute_value(x[0], x[1], x[2]),
                best_coeffs, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True}
            )
            if nm_final.fun < best_val:
                best_val = nm_final.fun
                best_coeffs = nm_final.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
