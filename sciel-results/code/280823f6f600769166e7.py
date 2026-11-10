"""Improved growth index approximation using direct numerical optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

def _compute_reference_growth(Om_m0, w0, wa, a_vals):
    """
    Numerically solve the growth equation for f and D for given cosmology parameters.
    Returns f_ref and D_ref (normalized) at a_vals.
    """
    # Parameters for the ODE
    # f = d ln D / d ln a
    # f' = (1/2) f^2 + (3/2) Om_m(a) - f
    # We integrate from a=1 (z=0) backwards to a_min.
    # D(1) = 1.
    # f(1) is approximated by -0.55 (or more accurately derived).
    # Let's use a small step integration or solve_ivp.
    # Since we need high accuracy and speed for optimization, we'll use a simple RK4 or solve_ivp with tight tolerances.
    
    a_min = min(a_vals)
    a_max = 1.0
    
    # Define Omega_m(a)
    def Om_m(a):
        if a <= 0: return 1.0
        # CPL density: rho_de ~ a^(-3(1+w0+wa)) is an approximation for small wa.
        # More accurate: rho_de(a) = rho_de0 * a^{-3(1+w0)} * exp(-3*wa*(1-a))
        # Let's use the exponential form for better accuracy across wa range.
        # Om_de0 = 1 - Om_m0
        # rho_de(a) / rho_de(1) = a^{-3(1+w0)} * exp(-3*wa*(1-a))
        # rho_m(a) / rho_m(1) = a^{-3}
        # Om_m(a) = (Om_m0 * a^{-3}) / (Om_m0 * a^{-3} + (1-Om_m0) * a^{-3(1+w0)} * exp(-3*wa*(1-a)))
        
        if a == 1.0:
            return Om_m0
        
        term_m = Om_m0 * (a ** -3)
        term_de = (1 - Om_m0) * (a ** (-3 * (1 + w0))) * np.exp(-3 * wa * (1 - a))
        total = term_m + term_de
        if total == 0:
            return 0.0
        return term_m / total

    # Initial condition for f at a=1
    # f(1) ~ 0.55 + 0.05(1+w0) is a common approx, but let's just use a slightly larger a for start?
    # No, we need D(1)=1. We can integrate forward from a_min to 1? No, D is usually defined such that D(1)=1.
    # It is easier to integrate backwards from a=1 to a_min.
    # Let's use solve_ivp from a=1 to a_min.
    
    # Define ODE for f as function of ln(a) to handle small a better?
    # x = ln(a). a = exp(x).
    # df/dx = (1/2) f^2 + (3/2) Om_m(exp(x)) - f
    # d ln D / dx = f
    # So D(x) = D(1) * exp(∫ f dx)
    
    x_vals = np.log(a_vals)
    x_start = 0.0 # ln(1)
    x_end = min(x_vals)
    
    # Initial f at a=1
    # Using the Linder approx as initial guess is fine for the solver to converge.
    f_init = 0.55 + 0.05 * (1 + w0)
    
    # We need to solve for f(x) and lnD(x) simultaneously?
    # Or just f(x) and then integrate f to get D.
    # Let's solve for f(x) using solve_ivp.
    
    def ode_fun(x, y):
        f_val = y[0]
        a_val = np.exp(x)
        Om_m_val = Om_m(a_val)
        df_dx = 0.5 * f_val**2 + 1.5 * Om_m_val - f_val
        return [df_dx]
    
    # Solve from x_start to x_end
    # We want values at x_vals.
    # x_vals are negative or zero. x_start=0. x_end < 0.
    # solve_ivp requires t_eval to be sorted in the same direction as t_span.
    # So t_span = (0, x_end). t_eval = sorted(x_vals, reverse=True) ? 
    # Actually, x_vals are decreasing from 0 to x_end.
    # Let's sort x_vals in descending order (from 0 to x_end) to match t_span direction.
    # Wait, x_vals from a_vals (0.05 to 1.0) -> ln(0.05) to 0.
    # So x_vals are increasing if a_vals are increasing.
    # If we integrate from 0 to x_end (negative), we are going backwards.
    # t_span = (0.0, x_end). 
    # t_eval must be in [x_end, 0.0] and sorted? No, solve_ivp handles direction.
    # t_eval should be within t_span.
    
    # Let's just evaluate at the specific x_vals.
    # Sort x_vals in descending order (from 0 to x_end) to align with integration direction 0 -> x_end.
    sort_idx = np.argsort(x_vals)[::-1]
    x_eval = x_vals[sort_idx]
    
    # Ensure x_eval starts at 0? No, 0 is the start.
    # If 0 is not in x_eval, we just integrate to x_end and interpolate?
    # Better: Include 0 in t_eval if possible, or just integrate and use dense output.
    
    try:
        sol = minimize._ode_solver or None # Placeholder, we will use solve_ivp directly
    except:
        pass
        
    from scipy.integrate import solve_ivp
    
    # We need to be careful with the range.
    # If x_end is very negative, it might be slow. But a_min=0.05 -> ln(0.05) ~ -3.
    # This is small.
    
    sol = solve_ivp(ode_fun, [0.0, x_end], [f_init], t_eval=x_eval, method='RK45', rtol=1e-8, atol=1e-10)
    
    if not sol.success:
        # Fallback to rough approximation if solver fails
        f_ref = np.full_like(a_vals, f_init)
        D_ref = np.ones_like(a_vals)
        return f_ref, D_ref

    f_ref_sorted = sol.y[0]
    
    # Now compute D_ref.
    # ln D(x) = ln D(1) + ∫_0^x f(s) ds
    # Since we integrated backwards, ∫_0^x f(s) ds = - ∫_x^0 f(s) ds.
    # We have f at x_eval. We can integrate f from x_end to 0 using cumulative trapz.
    # x_eval is sorted descending (0 to x_end).
    # Let's flip to ascending (x_end to 0) for integration.
    x_asc = x_eval[::-1]
    f_asc = f_ref_sorted[::-1]
    
    # Integrate f from x_end to x
    # ln D(x) - ln D(1) = ∫_1^a f da'/a' = ∫_0^x f(s) ds
    # We have values at x_asc.
    # Let's compute cumulative integral from x_end to 0.
    # I(x) = ∫_{x_end}^x f(s) ds.
    # Then ∫_0^x f(s) ds = I(x) - I(0).
    # Since D(1)=1, ln D(1)=0.
    # So ln D(x) = ∫_0^x f(s) ds.
    
    # Using np.cumsum with trapezoidal rule.
    dx = np.diff(x_asc)
    trapz_integrand = 0.5 * (f_asc[:-1] + f_asc[1:]) * dx
    cum_int = np.concatenate(([0.0], np.cumsum(trapz_integrand)))
    # cum_int[i] is integral from x_asc[0] to x_asc[i].
    # We want integral from 0 to x.
    # x_asc[0] is x_end (min). x_asc[-1] is 0 (max).
    # So cum_int[-1] is ∫_{x_end}^0 f(s) ds.
    # We want ∫_0^x f(s) ds = - ∫_x^0 f(s) ds.
    # Let J(x) = ∫_x^0 f(s) ds.
    # J(x) = ∫_{x_end}^0 f(s) ds - ∫_{x_end}^x f(s) ds = cum_int[-1] - cum_int[i] (where x_asc[i]=x).
    # So ln D(x) = - J(x) = cum_int[i] - cum_int[-1].
    
    ln_D_asc = cum_int - cum_int[-1]
    
    # Map back to original a_vals order.
    # x_eval was x_vals[sort_idx].
    # So f_ref_sorted corresponds to x_eval.
    # We need to unsort f_ref_sorted and ln_D_asc to match a_vals.
    
    f_ref = np.empty_like(f_ref_sorted)
    D_ref = np.empty_like(f_ref_sorted)
    
    f_ref[sort_idx] = f_ref_sorted
    D_ref[sort_idx] = np.exp(ln_D_asc)
    
    return f_ref, D_ref

def construct():
    # Define grid for optimization
    # We sample a subset of the validation space to keep optimization fast.
    Om_m0_vals = np.array([0.2, 0.25, 0.3])
    w0_vals = np.array([-1.2, -0.5, 0.0, 0.5])
    wa_vals = np.array([-0.5, 0.0, 0.5])
    a_vals = np.linspace(0.05, 1.0, 20)
    
    # Pre-compute reference data
    ref_data = []
    for Om_m0 in Om_m0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                f_ref, D_ref = _compute_reference_growth(Om_m0, w0, wa, a_vals)
                ref_data.append((Om_m0, w0, wa, a_vals, f_ref, D_ref))
                
    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        
        for Om_m0, w0, wa, a, f_ref, D_ref in ref_data:
            # Calculate approx f and D
            # q = 1 - Om_m(a)
            # u = 1 + w0 + wa*(1-a)
            # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
            # f = Om_m(a)^gamma
            
            # Compute Om_m(a) using the same CPL approximation as in the evaluator?
            # The task says "evaluator computes...". We should match the evaluator's Om_m(a) definition.
            # Standard CPL: rho_de ~ a^{-3(1+w0)} exp(-3wa(1-a)).
            # Let's assume the evaluator uses this standard form.
            
            a_safe = np.where(a > 1e-6, a, 1e-6)
            term_m = Om_m0 * (a_safe ** -3)
            term_de = (1 - Om_m0) * (a_safe ** (-3 * (1 + w0))) * np.exp(-3 * wa * (1 - a_safe))
            total = term_m + term_de
            Om_m_a = term_m / total
            
            q = 1 - Om_m_a
            u = 1 + w0 + wa * (1 - a_safe)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_safe + c5*u*q
            
            # Check gamma constraint [0.1, 1.2]
            # If violated, return a large penalty
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e6
            
            f_approx = Om_m_a ** gamma
            
            # Relative error for f
            err_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-9)
            max_err = max(max_err, np.max(err_f))
            
            # Calculate D_approx
            # D(a)/D(1) = exp(-∫_a^1 f(x) dx) ? 
            # Task: D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
            # Let x = ln(a'). Integral from ln(a) to 0 of f(exp(x)) dx.
            # Let's approximate this integral numerically using the same a_grid.
            # We need f at points between a and 1.
            # We can use the approximated f values on the grid and interpolate/integrate.
            
            # Since a_vals are fixed, we can pre-compute the integral for each a in the grid
            # using the approximated f values.
            
            # However, a_vals in ref_data are the same for all models.
            # We can compute the integral for each a_i using trapezoidal rule on the grid.
            # The grid is a_vals.
            # For a given a_target = a_vals[i], we need ∫_{ln(a_target)}^0 f(exp(x)) dx.
            # This is equivalent to ∫_{a_target}^1 f(a') da'/a'.
            
            # We have f_approx at a_vals.
            # Let's compute the cumulative integral from 1 down to a_min.
            # I(a) = ∫_a^1 f(a') da'/a'
            # I(1) = 0.
            # I(a_i) = I(a_{i+1}) + ∫_{a_i}^{a_{i+1}} f(a') da'/a'
            
            # Let's sort a_vals in ascending order. They are already linspace 0.05 to 1.0.
            # So a[0] is min, a[-1] is 1.0.
            # We compute I from 1.0 backwards.
            
            # f_approx is computed for all a_safe.
            # Let's compute the integrand g(a) = f(a)/a.
            g = f_approx / a_safe
            
            # Trapezoidal rule for ∫_{a_i}^{a_{i+1}} g(a) da
            # a is ascending.
            # dx = a_{i+1} - a_i
            # Integral_i = 0.5 * (g_i + g_{i+1}) * (a_{i+1} - a_i)
            
            da = np.diff(a_safe)
            trapz_vals = 0.5 * (g[:-1] + g[1:]) * da
            
            # Cumulative sum from the end (1.0) backwards.
            # I[-1] = 0 (at a=1)
            # I[-2] = Integral_{a_{-2}}^{a_{-1}}
            # I[i] = sum of trapz_vals from i to end-1
            
            # We want I for each a.
            # Let's reverse the trapz_vals to sum from 1.0 downwards.
            # trapz_vals[i] is integral from a[i] to a[i+1].
            # I[i] = sum_{j=i}^{N-2} trapz_vals[j]
            
            # Use cumsum on reversed array
            rev_trapz = trapz_vals[::-1]
            cum_int_rev = np.concatenate(([0.0], np.cumsum(rev_trapz)))
            # cum_int_rev[0] = 0 (corresponds to a[-1]=1.0)
            # cum_int_rev[1] = integral from a[-2] to a[-1]
            # ...
            # cum_int_rev[k] = integral from a[N-1-k] to a[N-1]
            
            # We need I[i] for each i.
            # I[i] corresponds to cum_int_rev[N-1-i].
            
            N = len(a_safe)
            I = np.zeros(N)
            for i in range(N):
                I[i] = cum_int_rev[N - 1 - i]
                
            D_approx = np.exp(-I)
            
            # Relative error for D
            err_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-9)
            max_err = max(max_err, np.max(err_D))
            
        return max_err

    # Initial guess from Linder
    x0 = [0.55, 0.0, 0.0, 0.05, 0.0, 0.0]
    
    # Optimize
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    coeffs = res.x
    coeffs = np.clip(coeffs, -10, 10)
    
    return coeffs.tolist()

# EVOLVE-BLOCK-END
