import numpy as np
import scipy.optimize
import scipy.integrate
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 seconds margin for safety

    # Representative cosmologies for training
    # We want to cover the likely validation space robustly.
    # Omega_m0 in [0.2, 0.4], w0 in [-1.5, -0.5], wa in [-1.0, 1.0]
    om_m0_vals = [0.2, 0.25, 0.3, 0.35, 0.4]
    w0_vals = [-1.5, -1.2, -1.0, -0.8, -0.5]
    wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    cosmologies = []
    for om in om_m0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                cosmologies.append((om, w0, wa))

    # Grid for a values to check errors
    # Log-spaced from 0.05 to 1.0
    # Use a dense grid to capture structure
    a_grid = np.logspace(np.log10(0.05), 0.0, 100)
    x_grid = np.log(a_grid)

    # Pre-compute reference solutions
    # We need f_ref (growth rate) for each cosmology on a_grid.
    # D_ref is derived from f_ref.
    ref_data = []
    
    # Helper to compute Omega_m(a)
    def get_om_m(a, om_m0, w0, wa):
        om_l0 = 1.0 - om_m0
        # rho_DE(a) ~ a^{-3(1+w0)} * exp(-3wa(a-1))
        de_term = om_l0 * (a ** (-3.0 * (1.0 + w0))) * np.exp(-3.0 * wa * (a - 1.0))
        m_term = om_m0 * (a ** -3.0)
        return m_term / (m_term + de_term)

    # Generate reference data
    for (om_m0, w0, wa) in cosmologies:
        om_l0 = 1.0 - om_m0
        
        def w_of_a(a):
            return w0 + wa * (1.0 - a)
        
        def om_m_of_a(a):
            return get_om_m(a, om_m0, w0, wa)

        # Integrate ODE from a=1 (x=0) backwards to a_min
        # D'' + (3/2 + 3/2 w) D' - 3/2 Om_m D = 0
        # Initial conditions at a=1: D(1)=1, D'(1) = f(1)
        # Use a high-accuracy seed for f(1) to ensure stability
        f_seed = om_m0 ** 0.55
        
        def ode_system(x, y):
            a = np.exp(x)
            om_m = om_m_of_a(a)
            w_a = w_of_a(a)
            dD = y[1]
            dDp = -1.5 * (1.0 + w_a) * y[1] + 1.5 * om_m * y[0]
            return [dD, dDp]

        x_min = np.log(0.05)
        
        # Use solve_ivp with high accuracy
        try:
            sol = scipy.integrate.solve_ivp(ode_system, [0, x_min], [1.0, f_seed], 
                                            method='DOP853', rtol=1e-10, atol=1e-12,
                                            dense_output=True, max_step=0.05)
            
            if not sol.success:
                raise RuntimeError("Integration failed")

            sol_grid = sol.sol(x_grid)
            D_vals = sol_grid[0]
            f_vals = sol_grid[1] / D_vals
        except Exception:
            # Fallback to constant gamma if integration fails
            gamma_const = 0.55
            om_m_grid = np.array([om_m_of_a(a) for a in a_grid])
            f_vals = om_m_grid ** gamma_const
            # Integrate f to get D
            # D = exp(-int f dx)
            # Use cumulative trapezoidal rule
            x_rev = -x_grid[::-1]
            f_rev = f_vals[::-1]
            dx = x_rev[1] - x_rev[0]
            integrals_rev = np.zeros_like(x_rev)
            for i in range(1, len(x_rev)):
                integrals_rev[i] = integrals_rev[i-1] + 0.5 * (f_rev[i-1] + f_rev[i]) * dx
            integrals = integrals_rev[::-1]
            D_vals = np.exp(-integrals)

        ref_data.append((a_grid, D_vals, f_vals, om_m0, w0, wa))

    # Objective function: Minimize the maximum relative error
    def max_error_objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        
        for (a_grid, D_ref, f_ref, om_m0, w0, wa) in ref_data:
            om_m_grid = np.array([get_om_m(a, om_m0, w0, wa) for a in a_grid])
            q_grid = 1.0 - om_m_grid
            u_grid = 1.0 + w0 + wa * (1.0 - a_grid)
            
            gamma_grid = c0 + c1*q_grid + c2*q_grid**2 + c3*u_grid + c4*wa*a_grid + c5*u_grid*q_grid
            
            # Check gamma bounds
            # If gamma is out of bounds, return a large error
            if np.any(gamma_grid < 0.1) or np.any(gamma_grid > 1.2):
                return 1e6
            
            f_approx = om_m_grid ** gamma_grid
            rel_err_f = np.abs((f_approx - f_ref) / f_ref)
            max_err = max(max_err, np.max(rel_err_f))
            
            # Compute D_approx
            x_rev = -x_grid[::-1]
            f_rev = f_approx[::-1]
            dx = x_rev[1] - x_rev[0]
            integrals_rev = np.zeros_like(x_rev)
            for i in range(1, len(x_rev)):
                integrals_rev[i] = integrals_rev[i-1] + 0.5 * (f_rev[i-1] + f_rev[i]) * dx
            integrals = integrals_rev[::-1]
            D_approx = np.exp(-integrals)
            
            rel_err_D = np.abs((D_approx - D_ref) / D_ref)
            max_err = max(max_err, np.max(rel_err_D))
            
        return max_err

    # Initial guess: Constant gamma 0.55
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_coeffs = x0
    best_cost = max_error_objective(x0)
    
    # Use Differential Evolution for global optimization
    # This is more robust for non-convex minimax problems
    try:
        # Time check
        if time.time() < deadline - 20.0:
            # DE requires bounds
            bounds = [(-10, 10)] * 6
            
            # Use a smaller population for speed, but multiple restarts
            result = scipy.optimize.differential_evolution(
                max_error_objective, 
                bounds, 
                seed=42, 
                maxiter=100, 
                tol=1e-5,
                popsize=10,
                polish=True  # Use L-BFGS to polish the result
            )
            
            if result.fun < best_cost:
                best_cost = result.fun
                best_coeffs = result.x
                
    except Exception:
        pass

    # Final Validation
    # Check if best_coeffs is valid (gamma in [0.1, 1.2])
    valid = True
    test_a = np.logspace(np.log10(0.01), 0.0, 500)
    for (om_m0, w0, wa) in cosmologies:
        om_m_grid = np.array([get_om_m(a, om_m0, w0, wa) for a in test_a])
        q_grid = 1.0 - om_m_grid
        u_grid = 1.0 + w0 + wa * (1.0 - test_a)
        gamma_grid = best_coeffs[0] + best_coeffs[1]*q_grid + best_coeffs[2]*q_grid**2 + best_coeffs[3]*u_grid + best_coeffs[4]*wa*test_a + best_coeffs[5]*u_grid*q_grid
        if np.any(gamma_grid < 0.1) or np.any(gamma_grid > 1.2):
            valid = False
            break
            
    if not valid:
        # Revert to safe constant
        best_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    return best_coeffs.tolist()
# EVOLVE-BLOCK-END
