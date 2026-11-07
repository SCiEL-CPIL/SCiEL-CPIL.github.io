"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_value(c0, c1, c2):
    """Compute r^2/(2*pi) for given coefficients."""
    try:
        # H0(0)=1, H4(0)=12, H8(0)=168, H12(0)=9240
        H0_0 = 1.0
        H4_0 = 12.0
        H8_0 = 168.0
        H12_0 = 9240.0
        
        c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
        
        # q(y) where y = x^2, q(y) = g(x)/x^2
        # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
        # H0 = 1
        # H4 = 16x^4 - 48x^2 + 12 -> in y: 16y^2 - 48y + 12
        # H8 = 256x^8 - 1920x^6 + 4320x^4 - 3360x^2 + 168 -> in y: 256y^4 - 1920y^3 + 4320y^2 - 3360y + 168
        # H12 = 4096x^12 - 57600x^10 + 403200x^8 - 1120000x^6 + 1209600x^4 - 554400x^2 + 9240
        #   in y: 4096y^6 - 57600y^5 + 403200y^4 - 1120000y^3 + 1209600y^2 - 554400y + 9240
        
        # g(x)/x^2 = c0*(1/x^2) + c1*(16y^2-48y+12)/y + c2*(256y^4-1920y^3+4320y^2-3360y+168)/y + c3*(4096y^6-...+9240)/y
        
        # Wait, g(0) = 0, so g(x) = x^2 * q(x). Let me verify.
        # g(0) = c0 + 12*c1 + 168*c2 + 9240*c3 = 0 by construction.
        # So g(x) has no constant term, meaning g(x)/x^2 is a polynomial.
        
        # g(x) = c0*1 + c1*(16x^4-48x^2+12) + c2*(256x^8-1920x^6+4320x^4-3360x^2+168) + c3*(4096x^12-57600x^10+403200x^8-1120000x^6+1209600x^4-554400x^2+9240)
        
        # Coefficient of x^0: c0 + 12*c1 + 168*c2 + 9240*c3 = 0 (by construction)
        # Coefficient of x^2: -48*c1 - 3360*c2 - 554400*c3
        # Coefficient of x^4: 16*c1 + 4320*c2 + 1209600*c3
        # Coefficient of x^6: -1920*c2 - 1120000*c3
        # Coefficient of x^8: 256*c2 + 403200*c3
        # Coefficient of x^10: -57600*c3
        # Coefficient of x^12: 4096*c3
        
        # q(x) = g(x)/x^2, so q as polynomial in x has:
        # x^0: -48*c1 - 3360*c2 - 554400*c3
        # x^2: 16*c1 + 4320*c2 + 1209600*c3
        # x^4: -1920*c2 - 1120000*c3
        # x^6: 256*c2 + 403200*c3
        # x^8: -57600*c3
        # x^10: 4096*c3
        
        # In terms of y = x^2:
        # q(y) = a0 + a1*y + a2*y^2 + a3*y^3 + a4*y^4 + a5*y^5
        a0 = -48.0*c1 - 3360.0*c2 - 554400.0*c3
        a1 = 16.0*c1 + 4320.0*c2 + 1209600.0*c3
        a2 = -1920.0*c2 - 1120000.0*c3
        a3 = 256.0*c2 + 403200.0*c3
        a4 = -57600.0*c3
        a5 = 4096.0*c3
        
        # Check leading coefficient sign: if a5 < 0, flip all (g -> -g)
        if a5 < 0:
            a0, a1, a2, a3, a4, a5 = -a0, -a1, -a2, -a3, -a4, -a5
        
        # Find roots of q(y) = a5*y^5 + a4*y^4 + a3*y^3 + a2*y^2 + a1*y + a0
        coeffs = [a5, a4, a3, a2, a1, a0]
        roots_y = np.roots(coeffs)
        
        # Find positive real roots where sign changes
        best_y = -1
        for ry in roots_y:
            if abs(ry.imag) < 1e-10 and ry.real > 1e-12:
                y0 = ry.real
                # Check sign change: evaluate q at y0-eps and y0+eps
                eps = 1e-8 * max(1.0, y0)
                q_neg = np.polyval(coeffs, y0 - eps)
                q_pos = np.polyval(coeffs, y0 + eps)
                if q_neg * q_pos < 0:
                    if y0 > best_y:
                        best_y = y0
        
        if best_y <= 0:
            return float('inf')
        
        return best_y / (2.0 * np.pi)
    
    except Exception:
        return float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 170  # safety margin before 210s kill
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_value = _compute_value(seed[0], seed[1], seed[2])
    best_c = seed.copy()
    best_val = seed_value
    
    def objective(params):
        return _compute_value(params[0], params[1], params[2])
    
    # Phase 1: Differential evolution with wide bounds
    if time.time() < deadline - 60:
        try:
            bounds = [(-10, 10), (-10, 10), (-10, 10)]
            de_result = differential_evolution(
                objective, bounds, seed=42, maxiter=100, 
                popsize=20, tol=1e-10, polish=False,
                init='sobol', updating='immediate'
            )
            val = _compute_value(de_result.x[0], de_result.x[1], de_result.x[2])
            if val < best_val:
                best_val = val
                best_c = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multiple Nelder-Mead refinements from diverse starting points
    if time.time() < deadline - 20:
        # Start from best found
        try:
            res = minimize(objective, best_c, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            val = _compute_value(res.x[0], res.x[1], res.x[2])
            if val < best_val:
                best_val = val
                best_c = res.x.copy()
        except Exception:
            pass
        
        # Try random perturbations
        for i in range(30):
            if time.time() > deadline - 10:
                break
            np.random.seed(1000 + i)
            scale = 0.5
            init = best_c + np.random.randn(3) * scale
            try:
                res = minimize(objective, init, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
                val = _compute_value(res.x[0], res.x[1], res.x[2])
                if val < best_val:
                    best_val = val
                    best_c = res.x.copy()
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_c, method='Nelder-Mead',
                           options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True})
            val = _compute_value(res.x[0], res.x[1], res.x[2])
            if val < best_val:
                best_val = val
                best_c = res.x.copy()
        except Exception:
            pass
    
    return best_c

# EVOLVE-BLOCK-END
