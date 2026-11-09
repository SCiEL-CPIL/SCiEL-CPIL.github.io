"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def construct():
    """Return six coefficients for the growth-index approximation."""
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Define cosmology models to fit against
    # (Om0, w0, wa) tuples covering a range of CPL parameters
    cosmologies = [
        (0.3, -1.0, 0.0),    # LambdaCDM
        (0.3, -1.0, 0.3),    # CPL with positive wa
        (0.3, -1.0, -0.3),   # CPL with negative wa
        (0.25, -0.9, 0.0),   # Different Om0
        (0.35, -1.1, 0.2),   # Different w0
        (0.3, -0.8, -0.2),   # Another CPL
    ]
    
    # a values to check
    a_values = np.linspace(0.05, 1.0, 20)
    
    def compute_Om(a, Om0, w0, wa):
        """Compute Om(a) for CPL dark energy."""
        Edm = Om0 * a**(-3)
        Ede = (1 - Om0) * a**(-3*(1 + w0 + wa))
        return Edm / (Edm + Ede)
    
    def gamma_from_coeffs(coeffs, a, Om0, w0, wa):
        """Compute gamma from the six-coefficient formula."""
        c0, c1, c2, c3, c4, c5 = coeffs
        q = 1 - compute_Om(a, Om0, w0, wa)
        u = 1 + w0 + wa * (1 - a)
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
        return gamma
    
    def f_approx(coeffs, a, Om0, w0, wa):
        """Compute f = Om(a)^gamma from the approximation."""
        gamma = gamma_from_coeffs(coeffs, a, Om0, w0, wa)
        Om_a = compute_Om(a, Om0, w0, wa)
        return Om_a ** gamma
    
    def D_normalized_approx(coeffs, a, Om0, w0, wa):
        """Compute D(a)/D(1) using the approximation."""
        # Integrate -ln(f) from ln(a) to 0
        def integrand(x):
            a_x = np.exp(x)
            f_x = f_approx(coeffs, a_x, Om0, w0, wa)
            return -np.log(f_x)
        
        integral = 0.0
        if a > 0:
            integral = solve_ivp(integrand, [np.log(a), 0.0], [0.0], 
                                method='RK45', rtol=1e-10, atol=1e-12)[1][-1]
        return np.exp(-integral)
    
    def compute_reference(Om0, w0, wa):
        """Solve the ODE to get reference f and D."""
        # Second-order ODE for growth factor in CPL cosmology
        # D'' + (3 + H'/H) D' - (3/2) Om D = 0
        # In terms of ln(a) = x: d2D/dx2 + (3 + dlnH/dx) dD/dx - (3/2) Om D = 0
        
        def ode_system(x, y):
            a_x = np.exp(x)
            Om_x = compute_Om(a_x, Om0, w0, wa)
            # dlnH/dx = -3/2 * (1 + w(a))
            w_a = w0 + wa * (1 - a_x)
            dlnH_dx = -1.5 * (1 + w_a)
            D, dD = y
            d2D = -(3 + dlnH_dx) * dD + 1.5 * Om_x * D
            return [dD, d2D]
        
        # Integrate from a=1 (x=0) backwards to a=a_min
        # Use boundary conditions: D(1)=1, D'(1) = f(1) = Om(1)^gamma_ref
        # We need to find f(1) by shooting method
        
        def shoot_f1(f1_guess):
            y0 = [1.0, f1_guess]
            sol = solve_ivp(ode_system, [0.0, np.log(0.05)], y0, 
                          method='RK45', rtol=1e-10, atol=1e-12, dense_output=True)
            if not sol.success:
                return 1e10
            # Check behavior at small a: D should behave as a
            a_min = 0.05
            y_min = sol.sol(np.log(a_min))
            D_min, dD_min = y_min
            # For small a, D ~ a, so D/a should approach 1
            ratio = D_min / a_min
            return abs(ratio - 1.0)
        
        # Find f1 using brentq
        from scipy.optimize import brentq
        try:
            f1 = brentq(shoot_f1, 0.1, 1.5, xtol=1e-10)
        except:
            f1 = compute_Om(1.0, Om0, w0, wa) ** 0.55
        
        # Solve with correct f1
        y0 = [1.0, f1]
        sol = solve_ivp(ode_system, [0.0, np.log(0.05)], y0, 
                       method='RK45', rtol=1e-10, atol=1e-12, dense_output=True)
        
        if not sol.success:
            return None
        
        return sol, f1
    
    def objective(coeffs):
        """Compute maximum relative error across all cosmologies and a values."""
        max_err = 0.0
        for Om0, w0, wa in cosmologies:
            ref_data = compute_reference(Om0, w0, wa)
            if ref_data is None:
                continue
            sol, f1_ref = ref_data
            
            for a in a_values:
                x = np.log(a)
                if x >= 0:
                    continue
                    
                # Reference values
                y_ref = sol.sol(x)
                D_ref, dD_ref = y_ref
                f_ref = dD_ref / D_ref
                
                # Approximation values
                f_app = f_approx(coeffs, a, Om0, w0, wa)
                D_app = D_normalized_approx(coeffs, a, Om0, w0, wa)
                
                # Relative errors
                err_f = abs(f_app - f_ref) / abs(f_ref)
                err_D = abs(D_app - D_ref) / abs(D_ref)
                
                max_err = max(max_err, err_f, err_D)
        
        return 100 * max_err
    
    # Initial guess
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Try to optimize
    best_coeffs = x0
    best_score = 100 * 0.5  # Initial worst-case estimate
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-4,
                                  'adaptive': True})
        
        if time.time() < deadline:
            if result.fun < best_score:
                best_coeffs = result.x
                best_score = result.fun
    except Exception:
        pass
    
    # Check gamma bounds
    def check_gamma_bounds(coeffs):
        for Om0, w0, wa in cosmologies:
            for a in a_values:
                gamma = gamma_from_coeffs(coeffs, a, Om0, w0, wa)
                if gamma < 0.1 or gamma > 1.2:
                    return False
        return True
    
    if not check_gamma_bounds(best_coeffs):
        # Fall back to constant gamma
        best_coeffs = x0
    
    # Ensure coefficients are in [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return list(best_coeffs)
# EVOLVE-BLOCK-END
