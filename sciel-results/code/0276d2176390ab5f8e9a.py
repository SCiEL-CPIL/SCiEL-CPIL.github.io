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
    deadline = 195.0  # Safety margin before 210s kill
    
    # Seed
    seed_coeffs = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    seed_value = _compute_value(*seed_coeffs)
    
    best_coeffs = seed_coeffs.copy()
    best_value = seed_value
    
    # Local refinement around seed with tighter bounds
    # The seed is likely close to optimal, so focus search there
    local_bounds = [
        (-1.5, -0.5),  # c0 around -1.13
        (0.0, 0.5),    # c1 around 0.04
        (-0.01, 0.01)  # c2 around 0.0003
    ]
    
    # Use COBYLA for constrained optimization with smooth surrogate
    # Create a smooth approximation: penalize when P(y) < 0 for y > 0
    def smooth_objective(x):
        c0, c1, c2 = x
        c3 = -(c0 + 3.0 * c1 + 105.0 * c2) / 10395.0
        
        a5 = c3
        a4 = -60.0 * c3
        a3 = 1260.0 * c3 + c2
        a2 = -12600.0 * c3 - 28.0 * c2
        a1 = 45360.0 * c3 + 280.0 * c2 - 6.0 * c1
        a0 = -30240.0 * c3 - 840.0 * c2 - 6.0 * c1
        
        # Sample P(y) on a grid to check for negative values
        # If P(y) < 0 anywhere, it's invalid (or we need to find where it crosses)
        # For the smooth objective, we want to minimize the largest root where sign changes
        
        # Find roots
        coeffs = [a5, a4, a3, a2, a1, a0]
        try:
            roots = np.roots(coeffs)
        except Exception:
            return 1e100
        
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-10:
                r_real = r.real
                if r_real > 1e-8:
                    real_roots.append(r_real)
        
        if not real_roots:
            return 1e100
        
        max_y = 0.0
        for r in real_roots:
            # Check if this is a sign-changing root
            eps = 1e-8 * max(1.0, r)
            y_left = r - eps
            y_right = r + eps
            
            val_left = ((a5 * y_left + a4) * y_left + a3) * y_left * y_left + (a2 * y_left + a1) * y_left + a0
            val_right = ((a5 * y_right + a4) * y_right + a3) * y_right * y_right + (a2 * y_right + a1) * y_right + a0
            
            if val_left * val_right < 0:
                if r > max_y:
                    max_y = r
        
        if max_y == 0.0:
            return 1e100
            
        return max_y / (2.0 * np.pi)
    
    # Phase 1: Focused local search around seed using Nelder-Mead
    if time.time() - start_time < 80.0:
        try:
            res = minimize(
                smooth_objective, seed_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Grid search in tight neighborhood for fine-tuning
    if time.time() - start_time < 140.0:
        # Fine grid around best found
        c0_center, c1_center, c2_center = best_coeffs
        c0_range = 0.05
        c1_range = 0.02
        c2_range = 0.002
        n_steps = 25
        
        c0_vals = np.linspace(c0_center - c0_range, c0_center + c0_range, n_steps)
        c1_vals = np.linspace(c1_center - c1_range, c1_center + c1_range, n_steps)
        c2_vals = np.linspace(c2_center - c2_range, c2_center + c2_range, n_steps)
        
        for c0 in c0_vals:
            if time.time() - start_time > 150.0:
                break
            for c1 in c1_vals:
                if time.time() - start_time > 155.0:
                    break
                for c2 in c2_vals:
                    if time.time() - start_time > 160.0:
                        break
                    val = _compute_value(c0, c1, c2)
                    if val < best_value:
                        best_value = val
                        best_coeffs = np.array([c0, c1, c2])
    
    # Phase 3: Final high-precision polish
    if time.time() - start_time < 185.0:
        try:
            res = minimize(
                _compute_value, best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return best_coeffs
