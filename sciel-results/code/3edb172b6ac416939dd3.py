"""Improved six-coefficient growth-index approximation using numerical optimization."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import warnings

# EVOLVE-BLOCK-START
def construct():
    # Set up the optimization problem
    # We will fit the 6 coefficients to minimize the max relative error
    # on a set of representative CPL cosmologies.
    
    # Representative cosmologies (w0, wa, Om)
    # Om is usually ~0.3. We vary w0 and wa.
    cosmologies = [
        (-1.0, 0.0, 0.3),   # LCDM
        (-1.0, 0.5, 0.3),   # CPL
        (-0.9, 0.0, 0.3),
        (-1.1, 0.0, 0.3),
        (-1.0, 1.0, 0.3),
        (-1.0, -1.0, 0.3),
        (-0.8, 0.5, 0.3),
        (-1.2, 0.5, 0.3),
        (-1.0, 0.0, 0.2),
        (-1.0, 0.0, 0.4),
    ]
    
    # Scale factors to check: a from 0.05 to 1.0
    # x = ln(a), so x from ln(0.05) to 0
    x_vals = np.linspace(np.log(0.05), 0, 50)
    
    def compute_true_growth(w0, wa, Om, x_grid):
        """Solve the ODE for D(x) and return f(x) and D(x)."""
        def E_sq(x):
            a = np.exp(x)
            rho_m = Om * a**(-3)
            rho_k = (1 - Om) * a**(-2) # Assuming flat for simplicity? 
            # Task doesn't specify curvature. Usually assume flat.
            # If not flat, we need Ok. Let's assume flat.
            rho_de = (1 - Om) * a**(-3*(1+w0)) * np.exp(3*wa*(1-a))
            return rho_m + rho_de # H^2/H0^2
        
        def dE_sq_dx(x):
            a = np.exp(x)
            # d/da (rho_m + rho_de) * a
            # rho_m = Om a^-3 -> d/da = -3 Om a^-4
            # rho_de = (1-Om) a^-3(1+w0) exp(3wa(1-a))
            # d/da rho_de = (1-Om) [ -3(1+w0) a^-3(1+w0)-1 exp(3wa(1-a)) + a^-3(1+w0) exp(3wa(1-a)) * (-3wa) ]
            # = (1-Om) a^-3(1+w0) exp(3wa(1-a)) [ -3(1+w0) - 3wa ]
            # Note: d/da (1-a) = -1.
            
            term_m = -3 * Om * a**(-4)
            
            base_de = (1 - Om) * a**(-3*(1+w0)) * np.exp(3*wa*(1-a))
            deriv_de = base_de * (-3*(1+w0) - 3*wa)
            
            dE_da = term_m + deriv_de
            return a * dE_da
        
        def ode(x, y):
            D, D_x = y
            E2 = E_sq(x)
            if E2 <= 0:
                return [0, 0]
            E = np.sqrt(E2)
            dE_dx = dE_sq_dx(x) / (2*E)
            
            # D_xx + (2 + (1/a) * dE/dx / E) * D_x - Om(a) * E^2 * D = 0
            # Wait, the term is - Omega_m(a) * E(x)^2 * D ?
            # From derivation: D_xx + (2 + 1/a E'/E) D_x - Om(a) E^2 D = 0?
            # Let's check units.
            # 4 pi G rho_m = H^2 Om(a).
            # Term was - H^2 Om(a) D.
            # Divided by H^2: - Om(a) D.
            # So the coefficient is just - Om(a).
            
            coeff_x = 2 + (1/np.exp(x)) * (dE_dx / E)
            coeff_D = -Om * np.exp(x)**(-3) # Om(a)
            
            D_xx = -coeff_x * D_x - coeff_D * D
            return [D_x, D_xx]
        
        # Initial conditions at x=0 (a=1)
        # D(0) = 1
        # D_x(0) = f(0) * D(0) = f(0)
        # Approx f(0) = Om**0.55
        f0 = Om**0.55
        y0 = [1.0, f0]
        
        sol = solve_ivp(ode, [0, x_grid[0]], y0, t_eval=x_grid, rtol=1e-8, atol=1e-10)
        
        if not sol.success:
            # Fallback to simple approximation if solver fails
            a = np.exp(x_grid)
            Om_a = Om * a**(-3) / E_sq(x_grid)
            f_approx = Om_a**0.55
            D_approx = np.exp(-np.trapz(f_approx, x_grid, initial=0)[:-1]) # Rough
            return f_approx, D_approx

        D = sol.y[0]
        D_x = sol.y[1]
        f = D_x / D
        return f, D

    def loss_function(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        
        # Check gamma bounds
        # We will check this on the grid
        max_err = 0.0
        
        for w0, wa, Om in cosmologies:
            # Compute true values
            f_true, D_true = compute_true_growth(w0, wa, Om, x_vals)
            
            # Compute ansatz values
            a = np.exp(x_vals)
            Om_a = Om * a**(-3) / (Om * a**(-3) + (1-Om) * a**(-3*(1+w0)) * np.exp(3*wa*(1-a)))
            q = 1 - Om_a
            u = 1 + w0 + wa * (1 - a)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
            
            # Check gamma bounds
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1000.0 # Heavy penalty
            
            f_ans = Om_a**gamma
            
            # Compute D_ans from f_ans
            # D(a)/D(1) = exp(-integral_0^ln(a) f(x) dx)
            # x_vals are negative (past). integral from 0 to x is negative of integral from x to 0.
            # integral_0^x f dx = - integral_x^0 f dx
            # So D/D(1) = exp(integral_x^0 f dx)
            # We have f_ans on x_vals.
            # x_vals is sorted from min to 0.
            # integral_x^0 f dx is the cumulative integral from x to 0.
            # Let's compute cumulative integral from 0 to x.
            # np.trapz(f, x) gives integral from x[0] to x[i].
            # We want integral from 0 to x[i].
            # Since x[i] < 0, integral_0^{x[i]} = - integral_{x[i]}^0.
            
            # Let's compute I[i] = integral_{x[0]}^{x[i]} f_ans dx
            I = np.trapz(f_ans, x_vals, initial=0)
            # I[0] = 0 (at x_min)
            # I[-1] = integral_{x_min}^0 f_ans dx
            # We want integral_0^{x[i]} f_ans dx = I[i] - I[-1]
            # So D_ans[i] = exp(-(I[i] - I[-1])) = exp(I[-1] - I[i])
            
            D_ans = np.exp(I[-1] - I)
            
            # Relative errors
            # Avoid division by zero or near zero
            f_rel_err = np.abs((f_ans - f_true) / f_true)
            D_rel_err = np.abs((D_ans - D_true) / D_true)
            
            max_err = max(max_err, np.max(f_rel_err), np.max(D_rel_err))
        
        return max_err

    # Initial guess
    x0 = [0.55, -0.05, 0.01, 0.1, 0.0, 0.0]
    
    # Optimize
    # Use Nelder-Mead as it doesn't require gradients and handles non-smooth max well
    res = minimize(loss_function, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    best_coeffs = res.x
    # Clip to [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return list(best_coeffs)
# EVOLVE-BLOCK-END
