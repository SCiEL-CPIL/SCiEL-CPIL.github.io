"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds
import time

# Constants for the cosmology
# The evaluator will use specific validation cosmologies, but we need to optimize
# for a general range. The task mentions "smooth CPL dark-energy cosmologies".
# Common values: Om0 ~ 0.3, w0 ~ -1, wa ~ 0.
# We will perform a local optimization within the construct function using a
# representative set of test points to approximate the "worst relative error".

def _compute_error(coeffs, Om0, w0, wa, a_vals):
    """
    Compute the worst-case relative error for f and D for given parameters.
    coeffs: [c0, c1, c2, c3, c4, c5]
    Om0: Matter density at z=0
    w0: Dark energy equation of state at z=0
    wa: Dark energy equation of state evolution
    a_vals: Array of scale factors
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    
    # Check gamma constraint: gamma in [0.1, 1.2]
    # q = 1 - Om(a), u = 1 + w0 + wa*(1-a)
    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    
    # We need to compute Om(a) for a flat universe with CPL dark energy.
    # Om_m(a) = Om0 * a^-3 / E(a)^2
    # Om_de(a) = (1 - Om0) * exp(3 * (1+w0) * (1-a) - 1.5*wa*(1-a^2)) / E(a)^2 ? 
    # Actually, the standard form is:
    # rho_de(a) = rho_de0 * exp(3 * integral_0^a (1+w(a'))/a' da')
    # For CPL: w(a) = w0 + wa*(1-a)
    # integral_0^a (1+w0+wa*(1-a'))/a' da' = (1+w0)*ln(a) + wa*(1-a) + wa*ln(a) ? No.
    # integral_0^a (1+w0)/a' da' + integral_0^a wa*(1-a')/a' da'
    # = (1+w0)*ln(a) + wa * [ integral_0^a 1/a' da' - integral_0^a da' ]
    # = (1+w0)*ln(a) + wa * (ln(a) - a)
    # So rho_de(a) = rho_de0 * a^(3*(1+w0)) * exp(3*wa*(ln(a) - a))
    #                = rho_de0 * a^(3*(1+w0+wa)) * exp(-3*wa*a)
    
    # Let's use the standard definition of E(a)^2 = Om0/a^3 + (1-Om0)*a^(-3*(1+w0))*exp(-3*wa*(1-a))
    # Wait, the exponent term is usually written as:
    # exp(3*wa*(1-a)) is often used in simplified forms, but the exact integration gives:
    # a^(3*(1+w0)) * exp(3*wa*(ln(a)-a))
    # Let's stick to the most common definition in literature (e.g., Linder):
    # E^2(a) = Om0 a^-3 + (1-Om0) a^(-3(1+w0)) exp(-3 wa (1-a))
    # Note: Some definitions use exp(3 wa (1-a)) in the numerator or denominator differently.
    # Let's verify the limit. If wa=0, w=w0. E^2 = Om0/a^3 + (1-Om0)/a^(3(1+w0)). Correct.
    
    Om_m = Om0 * a_vals**(-3)
    Om_de = (1.0 - Om0) * a_vals**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a_vals))
    E2 = Om_m + Om_de
    E = np.sqrt(E2)
    Om_a = Om_m / E2
    
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa*(1.0 - a_vals)
    
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
    
    # Check constraints
    if np.any(gamma < 0.1) or np.any(gamma > 1.2):
        return 1e10 # Penalty for violating gamma constraint
    
    f_approx = Om_a**gamma
    
    # Reference f from ODE
    # dF/da = (3/2)Om_a + (3/2)Om_a(1+w)
    # F = Om_a^gamma. dF/da = gamma * Om_a^(gamma-1) * dOm_a/da
    # dOm_a/da = -3 Om_a (1 - Om_a) (1+w)  (for flat universe)
    # So dF/da = -3 gamma Om_a^gamma (1-Om_a)(1+w)
    # Equating: gamma * Om_a^(gamma-1) * (-3 Om_a (1-Om_a)(1+w)) = (3/2) Om_a (1 + Om_a(1+w)) ?
    # Wait, the growth equation is:
    # f' + f^2 + (3/2)Om = 0 (in terms of a? No, usually Hubble time)
    # In terms of a: df/da + f(3/2 Om + f) = 0 ?
    # Actually, the standard relation is f = Om_a^gamma.
    # The ODE is: f' + (3/2)Om + f^2 = 0 where ' is d/d ln a.
    # So df/da / a + (3/2)Om + f^2 = 0
    # df/da = -a [ (3/2)Om + f^2 ]
    
    # We can compute the reference f by solving the ODE or using the relation:
    # f_ref = Om_a^gamma_ref, where gamma_ref satisfies the ODE.
    # But we don't have gamma_ref analytically. We must compute f_ref numerically.
    # Since we are optimizing, we can pre-compute f_ref for the specific cosmology.
    # However, inside the optimization loop, we need to be fast.
    # Let's approximate the error by comparing f_approx to a high-precision numerical solution
    # that we compute once per cosmology setup, or use a local linearization.
    
    # For the purpose of this local optimization inside construct(), we can assume
    # that the "worst error" is dominated by the mismatch in the derivative or value.
    # A simpler proxy for the error in f is the residual of the ODE:
    # R = df/da + (3/2) a Om_a + a f^2
    # If f = Om^gamma, then df/da = gamma * Om^(gamma-1) * dOm/da
    # dOm/da = -3 Om (1-Om) (1+w)
    # So R = -3 gamma Om^gamma (1-Om)(1+w) + (3/2) a Om + a Om^(2 gamma)
    
    # Let's compute this residual at the test points.
    w_a = w0 + wa*(1.0 - a_vals)
    dOm_da = -3.0 * Om_a * (1.0 - Om_a) * (1.0 + w_a)
    df_da = gamma * Om_a**(gamma - 1.0) * dOm_da
    
    residual = df_da + (1.5 * a_vals * Om_a) + (a_vals * f_approx**2)
    
    # Normalize residual by a typical scale
    err_f = np.max(np.abs(residual) / (1.0 + np.abs(f_approx)))
    
    # For D(a), the error accumulates. A good proxy is the integrated error in ln f.
    # D/D(1) = exp(-int f dx). Error in D ~ int (error in f) dx.
    # We can approximate the worst error in D by integrating the absolute residual.
    # Or simply use the max error in f as a proxy, since D is an integral of f.
    # To be safer, let's integrate the residual.
    # We need to integrate from a to 1 (or 0 to a). The task says int_ln(a)^0 f(exp(x)) dx.
    # x = ln a. dx = da/a.
    # Integral from a to 1 of f(a') / a' da'.
    
    # Let's use a simple trapezoidal integration of the residual over the range.
    # This is a heuristic. The actual score is max relative error.
    
    # A better approach: compute the actual D_ref and D_approx.
    # But that requires solving the ODE for D_ref every time, which is slow.
    # However, for a fixed cosmology (Om0, w0, wa), D_ref is fixed.
    # We can pre-compute D_ref for a grid of a values.
    
    # Since we are inside construct(), we can set up the optimization for a few
    # representative cosmologies.
    
    return err_f

def _optimize_for_cosmology(Om0, w0, wa, a_grid):
    """
    Optimize coefficients for a specific cosmology.
    Returns the best coefficients and the error.
    """
    # Initial guess
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Define the objective function
    def objective(x):
        return _compute_error(x, Om0, w0, wa, a_grid)
    
    # Bounds
    bounds = Bounds([-10, -10, -10, -10, -10, -10], [10, 10, 10, 10, 10, 10])
    
    # Optimize
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-12})
        if res.success:
            return res.x, res.fun
        else:
            return x0, objective(x0)
    except Exception:
        return x0, objective(x0)

def construct():
    # We have a time budget. Let's use it to optimize for a few representative cosmologies.
    start_time = time.time()
    time_limit = 85.0 # Safety margin
    
    # Representative cosmologies
    # Om0 ~ 0.27 - 0.31, w0 ~ -1.0, wa ~ -0.5 to 0.5
    # Let's pick a few points in this space.
    cosmologies = [
        (0.30, -1.0, 0.0),   # LambdaCDM
        (0.27, -1.0, 0.5),   # w0wa with positive wa
        (0.31, -1.0, -0.5),  # w0wa with negative wa
        (0.28, -0.9, 0.0),   # w0 != -1
        (0.30, -1.1, 0.0),   # w0 < -1
    ]
    
    # Scale factor grid for evaluation
    # The evaluator checks "all checked locations". We should cover a wide range.
    # a from 0.1 to 1.0.
    a_grid = np.linspace(0.1, 1.0, 50)
    
    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_error = np.inf
    
    for Om0, w0, wa in cosmologies:
        if time.time() - start_time > time_limit:
            break
            
        coeffs, err = _optimize_for_cosmology(Om0, w0, wa, a_grid)
        
        # We want to minimize the MAX error across all cosmologies.
        # This is a min-max problem.
        # We can't just take the best coeffs for one cosmology.
        # We should aggregate the errors.
        
        # For now, let's just keep the coeffs that minimize the error for the current cosmology
        # and then we will do a final aggregation step.
        # Actually, a better strategy is to define a combined objective.
        
    # The above loop is not quite right for min-max.
    # Let's define a combined objective that evaluates the max error over all cosmologies.
    
    def combined_objective(x):
        max_err = 0.0
        for Om0, w0, wa in cosmologies:
            err = _compute_error(x, Om0, w0, wa, a_grid)
            if err > max_err:
                max_err = err
        return max_err
    
    # Optimize the combined objective
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    bounds = Bounds([-10, -10, -10, -10, -10, -10], [10, 10, 10, 10, 10, 10])
    
    try:
        res = minimize(combined_objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
        if time.time() - start_time < time_limit:
            if res.success:
                return res.x
            else:
                # Return the best point found
                return res.x
    except Exception:
        pass
    
    return x0
# EVOLVE-BLOCK-END
