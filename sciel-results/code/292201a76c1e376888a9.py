"""Improved growth-index ansatz with direct optimization from task structure."""
import numpy as np
from scipy.optimize import minimize
import time
import sys

# EVOLVE-BLOCK-START

# Deadline guard: evaluator kills after 90s; use 80s safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # 55s limit to be safe, leaving time for import/setup

def _check_deadline():
    return time.time() > _DEADLINE

def construct():
    """Return six coefficients [c0..c5] for the growth-index ansatz.
    
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    f = Om(a)^gamma
    D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
    
    We optimize these coefficients to minimize worst-case relative error
    across validation cosmologies for both f and D.
    """
    
    # Initial best result: the seed
    best_coeff = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = np.inf
    
    # Representative validation cosmologies (Om0, w0, wa)
    # We choose a spread to cover the likely test space
    val_cosmos = [
        (0.30, -1.0, 0.0),   # Lambda CDM
        (0.30, -1.0, 0.1),   # Positive wa
        (0.30, -1.0, -0.1),  # Negative wa
        (0.30, -0.9, 0.0),   # w0 > -1
        (0.30, -1.1, 0.0),   # w0 < -1
        (0.28, -1.0, 0.0),   # Lower Om0
        (0.32, -1.0, 0.0),   # Higher Om0
    ]
    
    # Scale factor grid for evaluation (log-spaced)
    # We use a moderate number of points to balance accuracy and speed
    a_grid = np.logspace(np.log10(0.1), 0.0, 100)
    
    # Pre-compute reference values for each cosmology
    # Structure: list of dicts or arrays
    # For each cosmology, we store:
    # a_ref, Om_ref, f_ref, D_ref
    ref_data = []
    
    def _compute_reference(Om0, w0, wa, a_vals):
        """
        Compute reference f and D for a given cosmology and a-grid.
        Uses numerical integration of the growth ODE.
        """
        a_vals = np.asarray(a_vals)
        n = len(a_vals)
        
        # 1. Compute Omega_m(a) for all a
        Om_a = np.zeros(n)
        for i, a in enumerate(a_vals):
            if a <= 0:
                Om_a[i] = 1.0
                continue
            # DE density ratio
            # u = 1 + w0 + wa*(1-a)
            u = 1.0 + w0 + wa * (1.0 - a)
            if abs(wa) < 1e-10:
                # Constant w
                if abs(1.0 + w0) < 1e-10:
                    # Lambda-like, but w=-1 exactly?
                    rho_de_ratio = 1.0
                else:
                    rho_de_ratio = a ** (-3.0 * (1.0 + w0))
            else:
                # CPL approximation
                # rho_de(a) / rho_de(1) = a^(-3(1+w0)) * (u / (1+w0))^(-3*wa/wa) ?
                # Standard formula: rho_de(a) ∝ a^(-3(1+w0)) (1+w0+wa(1-a))^(-3wa/wa)
                # Wait, the exponent is -3(1+w0+wa(1-a))/wa? No.
                # The exact integral is \int \frac{1+w(a)}{a} da.
                # For CPL, the approximation is:
                # rho_de(a) = rho_de(1) * a^{-3(1+w0)} * (1 + w0 + wa(1-a))^{-3wa/wa}
                # Let's verify the exponent.
                # d ln rho / d ln a = -3(1+w(a)) = -3(1+w0+wa(1-a))
                # ln rho(a) - ln rho(1) = -3 \int_1^a \frac{1+w0+wa(1-a')}{a'} da'
                # = -3 [ (1+w0) ln a + wa \int_1^a \frac{1-a'}{a'} da' ]
                # = -3 [ (1+w0) ln a + wa (ln a - (a-1)) ]
                # = -3 ln a [ 1+w0+wa ] + 3 wa (a-1)
                # So rho_de(a)/rho_de(1) = exp( -3 ln a (1+w0+wa) + 3 wa (a-1) )
                # This is the "exact" CPL integration result often used.
                # Let's use this exponential form as it is more accurate than the power-law approx.
                exponent = -3.0 * np.log(a) * (1.0 + w0 + wa) + 3.0 * wa * (a - 1.0)
                rho_de_ratio = np.exp(exponent)
            
            # Matter density ratio
            rho_m_ratio = a ** -3.0
            
            # Total density ratio
            # Om(a) = Om0 * rho_m_ratio / (Om0 * rho_m_ratio + (1-Om0) * rho_de_ratio)
            denom = Om0 * rho_m_ratio + (1.0 - Om0) * rho_de_ratio
            if denom == 0:
                Om_a[i] = 0.0
            else:
                Om_a[i] = Om0 * rho_m_ratio / denom
        
        # 2. Solve ODE for D(a)
        # ODE: D'' + P(x) D' - Q(x) D = 0 where x = ln a
        # P(x) = 3 + 1.5 * Om(a)
        # Q(x) = 1.5 * Om(a)
        # We integrate from x=0 (a=1) to x=ln(a_min).
        # Initial conditions at a=1: D(1)=1, D'(1)=f(1)=Om0^gamma_ref?
        # No, for the reference, we don't use the approximation.
        # We need the "true" f_ref.
        # The ODE determines D up to a constant. We normalize D(1)=1.
        # D'(1) is determined by the ODE and the requirement that D is the growing mode.
        # Actually, we can just integrate backwards from a=1.
        # At a=1, we can choose D(1)=1 and D'(1) = f_approx(1) or just a reasonable value?
        # No, f_ref is derived from D_ref.
        # The ODE is 2nd order. We need 2 initial conditions.
        # Usually, we set D(1)=1. What is D'(1)?
        # In the standard growth approximation, f(1) = Om0^gamma.
        # But for the reference, we should use the "exact" value.
        # However, the "exact" value depends on the initial conditions of the universe.
        # For a flat universe, the growing mode at a=1 has f(1) ≈ Om0^0.55.
        # But to be independent, we can just integrate the ODE and normalize D(1)=1.
        # The derivative D'(1) is a free parameter in the linear ODE? No, the ODE is homogeneous.
        # Any linear combination is a solution. We want the specific solution that corresponds to the linear growth of density perturbations.
        # This solution is uniquely defined by the requirement that D(a) ~ a at early times (matter dominated).
        # Or, more simply, we can use the fact that for a flat universe, the growing mode satisfies:
        # D'(1) = f(1) where f(1) is the "exact" growth rate.
        # But we don't know f(1) exactly without solving the ODE from early times.
        # Alternative: Integrate from early times (a=0.1) to a=1?
        # At a=0.1, D ~ a. So D(0.1)=0.1, D'(0.1)=1.
        # Then integrate forward to a=1.
        # Then normalize D(1) to 1.
        # This is robust.
        
        x_grid = np.log(a_vals)
        # We need to integrate from x_min to 0.
        # Let's use solve_ivp.
        def ode_func(x, y):
            # y = [D, D']
            a = np.exp(x)
            # Find Om(a) at this x
            # We have Om_a on a_vals. We need interpolation or direct calculation.
            # Direct calculation is better to avoid interpolation errors.
            u = 1.0 + w0 + wa * (1.0 - a)
            if abs(wa) < 1e-10:
                if abs(1.0 + w0) < 1e-10:
                    rho_de_ratio = 1.0
                else:
                    rho_de_ratio = a ** (-3.0 * (1.0 + w0))
            else:
                exponent = -3.0 * np.log(a) * (1.0 + w0 + wa) + 3.0 * wa * (a - 1.0)
                rho_de_ratio = np.exp(exponent)
            
            rho_m_ratio = a ** -3.0
            denom = Om0 * rho_m_ratio + (1.0 - Om0) * rho_de_ratio
            if denom == 0:
                Om = 0.0
            else:
                Om = Om0 * rho_m_ratio / denom
            
            P = 3.0 + 1.5 * Om
            Q = 1.5 * Om
            return [y[1], -P * y[1] + Q * y[0]]
        
        # Initial conditions at x_min (a=0.1)
        x_min = np.log(0.1)
        y0 = [0.1, 1.0]  # D ~ a, D' ~ 1
        
        # Integrate from x_min to 0
        try:
            sol = solve_ivp(ode_func, [x_min, 0.0], y0, t_eval=x_grid, method='RK45', rtol=1e-8, atol=1e-10)
            if not sol.success:
                return None
            D_vals = sol.y[0]
            D_prime_vals = sol.y[1]
            
            # Normalize D(1) = 1
            # Find index closest to x=0
            idx_1 = np.argmin(np.abs(x_grid))
            D_norm = D_vals[idx_1]
            if D_norm == 0:
                return None
            D_vals = D_vals / D_norm
            D_prime_vals = D_prime_vals / D_norm
            
            # f_ref = D'
            f_ref = D_prime_vals
            D_ref = D_vals
            
            return {
                'a': a_vals,
                'Om': Om_a,
                'f': f_ref,
                'D': D_ref
            }
        except Exception:
            return None

    # Import solve_ivp locally to avoid global import overhead if not needed, but it's standard
    from scipy.integrate import solve_ivp
    
    for c in val_cosmos:
        ref = _compute_reference(c[0], c[1], c[2], a_grid)
        if ref:
            ref_data.append(ref)
        
        if _check_deadline():
            break

    if not ref_data:
        return best_coeff.tolist()

    def objective(coeff):
        c0, c1, c2, c3, c4, c5 = coeff
        max_err = 0.0
        
        for ref in ref_data:
            a = ref['a']
            Om = ref['Om']
            f_ref = ref['f']
            D_ref = ref['D']
            
            # Compute our approximation
            u = 1.0 + ref['w0'] + ref['wa'] * (1.0 - a) # Wait, we didn't store w0, wa in ref dict.
            # Let's store w0, wa in ref dict.
            # I'll modify the loop to pass them.
            
            # Re-calc q, u
            q = 1.0 - Om
            # We need w0, wa. Let's assume they are in ref or pass them.
            # I will update the _compute_reference to return w0, wa.
            
            # For now, let's assume we have them.
            # Actually, I'll just re-structure the data storage.
            
            # Let's use the stored w0, wa from the loop variable? No, ref_data is a list.
            # I should store w0, wa in the dict.
            
            # Let's check the dict keys.
            # I will update the code below to include w0, wa in the dict.
            
            # Placeholder: get w0, wa
            # I'll refactor the loop to store w0, wa.
            
            # For this draft, I'll just use the first cosmology's w0, wa? No, that's wrong.
            # I need to store w0, wa in the ref dict.
            
            # Let's assume ref has 'w0' and 'wa'.
            w0 = ref['w0']
            wa = ref['wa']
            
            u = 1.0 + w0 + wa * (1.0 - a)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
            
            # Check bounds
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e6 # Penalize
            
            f_ours = np.power(np.maximum(Om, 1e-15), gamma)
            
            # D_ours
            # D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
            # We have f on a_grid. We can integrate numerically.
            # x = ln a. dx = da/a.
            # Integral from ln(a) to 0 of f(e^x) dx.
            # Let's compute the cumulative integral from 0 to ln(a) and negate.
            # I = \int_0^{ln a} f(e^x) dx
            # D = exp(-I)
            
            # Use cumulative trapezoid
            x = np.log(a)
            # We need integral from 0 to x.
            # scipy.integrate.cumulative_trapezoid
            from scipy.integrate import cumulative_trapezoid
            I = cumulative_trapezoid(f_ours, x, initial=0)
            # I[0] corresponds to x[0] (ln 0.1).
            # We want D(a) = exp(-I(a)).
            # At a=1, x=0, I=0, D=1.
            D_ours = np.exp(-I)
            
            # Relative errors
            # Avoid division by zero
            mask_f = f_ref > 1e-10
            mask_D = D_ref > 1e-10
            
            if np.any(mask_f):
                err_f = np.abs((f_ours[mask_f] - f_ref[mask_f]) / f_ref[mask_f])
                max_err = max(max_err, np.max(err_f))
            
            if np.any(mask_D):
                err_D = np.abs((D_ours[mask_D] - D_ref[mask_D]) / D_ref[mask_D])
                max_err = max(max_err, np.max(err_D))
        
        return max_err

    # Update ref_data to include w0, wa
    # I need to re-run the reference calculation with this fix.
    # Let's just do it in the main flow.
    
    ref_data = []
    for c in val_cosmos:
        Om0, w0, wa = c
        ref = _compute_reference(Om0, w0, wa, a_grid)
        if ref:
            ref['w0'] = w0
            ref['wa'] = wa
            ref_data.append(ref)
        if _check_deadline():
            break

    if not ref_data:
        return best_coeff.tolist()

    # Optimization
    # Start from seed
    x0 = best_coeff
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.success or res.fun < best_score:
            best_coeff = res.x
            best_score = res.fun
    except Exception:
        pass
    
    # Check deadline
    if _check_deadline():
        return best_coeff.tolist()
        
    # Try L-BFGS-B for refinement
    try:
        res2 = minimize(objective, best_coeff, method='L-BFGS-B', bounds=[(-10,10)]*6, options={'maxiter': 100})
        if res2.success or res2.fun < best_score:
            best_coeff = res2.x
            best_score = res2.fun
    except Exception:
        pass

    # Final check of bounds
    # If any coefficient is out of [-10, 10], clamp them
    best_coeff = np.clip(best_coeff, -10.0, 10.0)
    
    return best_coeff.tolist()

# EVOLVE-BLOCK-END
