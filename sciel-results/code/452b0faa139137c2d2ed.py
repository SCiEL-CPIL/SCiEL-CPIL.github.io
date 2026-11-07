import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
import sys

# Precomputed constants
H0_0 = 1.0
H4_0 = 3.0
H8_0 = 105.0
H12_0 = 10395.0

# Coefficients for H_n(x) expressed in powers of y=x^2
# H0(y) = 1
# H4(y) = y^2 - 6y + 3
# H8(y) = y^4 - 28y^3 + 280y^2 - 840y + 105
# H12(y) = y^6 - 60y^5 + 1260y^4 - 12600y^3 + 45360y^2 - 30240y + 10395

# We want P(y) = g(sqrt(y))/y
# g(y) = c0*1 + c1*(y^2 - 6y + 3) + c2*(y^4 - 28y^3 + 280y^2 - 840y + 105) + c3*(y^6 - 60y^5 + 1260y^4 - 12600y^3 + 45360y^2 - 30240y + 10395)
# g(0) = c0 + 3c1 + 105c2 + 10395c3 = 0 => c3 = -(c0 + 3c1 + 105c2)/10395

# P(y) = g(y)/y = c3*y^5 + (-60c3)*y^4 + (1260c3+c2)*y^3 + (-12600c3-28c2)*y^2 + (45360c3+280c2-6c1)*y + (-30240c3-840c2-6c1)
# Note: The constant term of g(y) is 0, so division by y is exact.
# Let's verify the constant term of P(y) is the coeff of y^1 in g(y).
# g(y) = ... + A1*y + A0. A0=0.
# P(y) = ... + A2*y + A1.
# So P(y) is a degree 5 polynomial.

def _compute_value(c0, c1, c2):
    c3 = -(c0 + 3.0 * c1 + 105.0 * c2) / 10395.0
    
    # Coefficients of P(y) from highest power to lowest
    # P(y) = a5*y^5 + a4*y^4 + a3*y^3 + a2*y^2 + a1*y + a0
    a5 = c3
    a4 = -60.0 * c3
    a3 = 1260.0 * c3 + c2
    a2 = -12600.0 * c3 - 28.0 * c2
    a1 = 45360.0 * c3 + 280.0 * c2 - 6.0 * c1
    a0 = -30240.0 * c3 - 840.0 * c2 - 6.0 * c1
    
    # If a5 is very small, the degree might drop. np.roots handles this by ignoring leading zeros? 
    # No, np.roots requires the array. If a5=0, it's degree 4.
    # We should strip leading zeros to avoid numerical issues with roots at infinity.
    coeffs = [a5, a4, a3, a2, a1, a0]
    
    # Find roots
    # To be safe against leading zeros, we can find the first non-zero
    # But for generic c's, a5=c3 is likely non-zero.
    try:
        roots = np.roots(coeffs)
    except Exception:
        return 1e100
        
    # Filter real positive roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-9:
            r_real = r.real
            if r_real > 1e-9: # Must be positive
                real_roots.append(r_real)
                
    if not real_roots:
        # If no positive real roots, what is the value?
        # The problem says "r be the largest real root... at which q changes sign".
        # If there are no such roots, the bound might be 0 or undefined.
        # However, g(x) is a polynomial. If it doesn't change sign for x>0, 
        # it might be always positive or always negative.
        # The task says "If g tends to a negative limit at infinity, g is replaced by -g".
        # This implies we care about the behavior at infinity too?
        # Actually, the value is r^2 / 2pi. If r doesn't exist, maybe r=0?
        # Let's assume if no sign-changing root, value is 0 (or very small).
        # But wait, if q(x) is always positive for x>0, then there are no sign changes.
        # The "upper bound on C4" logic might imply something else.
        # However, usually in these tasks, the optimal solution HAS a sign-changing root.
        # Let's return a large value if no root found, to penalize this.
        return 1e100

    # Check for sign change (odd multiplicity)
    # Evaluate P(y) at r - eps and r + eps
    # Use polyval for speed? Or manual?
    # For degree 5, manual or polyval is fine.
    
    max_y = 0.0
    eps_factor = 1e-6
    for r in real_roots:
        eps = eps_factor * max(1.0, r)
        y_left = r - eps
        y_right = r + eps
        
        # P(y) = a5*y^5 + ... + a0
        val_left = ((a5 * y_left + a4) * y_left + a3) * y_left * y_left + (a2 * y_left + a1) * y_left + a0
        val_right = ((a5 * y_right + a4) * y_right + a3) * y_right * y_right + (a2 * y_right + a1) * y_right + a0
        
        # Check sign change
        if val_left * val_right < 0:
            if r > max_y:
                max_y = r
                
    if max_y == 0.0:
        # No sign changing root found
        return 1e100
        
    return max_y / (2.0 * np.pi)

def construct():
    start_time = time.time()
    deadline = 200.0
    
    # Seed
    seed_coeffs = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    seed_value = _compute_value(*seed_coeffs)
    
    best_coeffs = seed_coeffs.copy()
    best_value = seed_value
    
    # Global Search with Differential Evolution
    if time.time() - start_time < 120.0:
        try:
            bounds = [(-5.0, 5.0), (-5.0, 5.0), (-0.1, 0.1)]
            # c2 is small in seed, keep it tighter?
            # Let's try wider first.
            bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
            
            def objective(x):
                return _compute_value(x[0], x[1], x[2])
            
            result = differential_evolution(
                objective, bounds,
                maxiter=200, popsize=20,
                tol=1e-10, seed=42,
                polish=True,
                updating='deferred'
            )
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Local Refinement
    if time.time() - start_time < 180.0:
        # Try multiple local starts
        rng = np.random.default_rng(42)
        
        starts = [best_coeffs.copy()]
        # Add random starts
        for _ in range(50):
            s = seed_coeffs + rng.normal(0, 1.0, 3)
            starts.append(s)
            
        for s in starts:
            if time.time() - start_time > 190.0:
                break
            try:
                # Nelder-Mead is robust for non-smooth
                res = minimize(
                    _compute_value, s,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True}
                )
                if res.fun < best_value:
                    best_value = res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass
                
    return best_coeffs
