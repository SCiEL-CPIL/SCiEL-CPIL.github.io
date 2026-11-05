"""Improved growth-index approximation using local linear regression on ln(f)."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Return six coefficients for the growth-index ansatz."""
    # Start with a deadline guard
    t_start = time.time()
    deadline = t_start + 75.0  # 75s safety margin within 90s limit
    
    # Initial best: the constant gamma seed
    best_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = _compute_score(best_coeffs)
    
    # Try to improve using a local search / grid refinement approach
    # The ansatz: gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # where q = 1 - Om(a), u = 1 + w0 + wa*(1-a)
    # f = Om(a)^gamma
    # D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
    
    # We'll use a multi-start local optimization approach
    # First, let's understand the problem structure:
    # For a given cosmology (Om0, w0, wa), we need to match f and D
    # The evaluator uses disjoint validation cosmologies
    
    # Strategy: Use a coarse-to-fine grid search + local refinement
    # Parameters are in [-10, 10]
    
    # Define a helper to compute the score for given coefficients
    # and validation cosmologies
    
    # Since we don't know the exact validation cosmologies, we'll use
    # a representative set that spans the likely parameter space
    # Typical cosmologies: Om0 in [0.2, 0.4], w0 in [-1.2, -0.8], wa in [-0.5, 0.3]
    
    validation_models = [
        (0.3, -1.0, 0.0),    # LambdaCDM
        (0.3, -1.0, 0.3),    # CPL with positive wa
        (0.3, -1.2, 0.5),    # Phantom-like
        (0.25, -0.8, -0.5),  # Quintessence-like
        (0.35, -1.0, 0.1),   # Mild deviation
        (0.3, -0.9, 0.2),    # Another CPL
    ]
    
    # Grid search over a coarse grid first
    # Then refine the best region
    
    def compute_score(coeffs):
        """Compute 100 * max relative error of f and D over validation models."""
        max_err = 0.0
        for (Om0, w0, wa) in validation_models:
            # Evaluate at various scale factors
            a_vals = np.linspace(0.1, 1.0, 50)
            for a in a_vals:
                # Compute Om(a)
                # Om(a) = Om0 * a^3 / (Om0*a^3 + (1-Om0)*a^(3*(1+w0+wa)) * exp(-3*wa*(1-a)))
                # Actually, for CPL: rho_DE ∝ a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
                # Let's use the standard formula
                if a <= 0:
                    continue
                # Matter density: Om_m(a) = Om0 * a^3 / E(a)^2
                # E(a)^2 = Om0*a^3 + (1-Om0)*a^(3*(1+w0+wa))*exp(-3*wa*(1-a))
                # But we need Om(a) = Om_m(a) = Om0*a^3/E(a)^2
                
                # Compute E(a)^2
                if wa != 0:
                    de_factor = a**(3*(1+w0+wa)) * np.exp(-3*wa*(1-a))
                else:
                    de_factor = a**(3*(1+w0))
                E2 = Om0 * a**3 + (1-Om0) * de_factor
                Om_a = Om0 * a**3 / E2 if E2 > 0 else 0
                
                if Om_a <= 0 or Om_a >= 1:
                    continue
                
                # Compute q and u
                q = 1 - Om_a
                u = 1 + w0 + wa*(1-a)
                
                # Compute gamma from coefficients
                c0, c1, c2, c3, c4, c5 = coeffs
                gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                
                # Check gamma bounds
                if gamma < 0.1 or gamma > 1.2:
                    return 100.0  # Invalid, return high score
                
                # Compute f = Om(a)^gamma
                f_approx = Om_a**gamma
                
                # Compute reference f using the ODE-based approach
                # For the reference, we use the standard growth equation:
                # f' + (3/2 + q) f + f^2 = (3/2) Om(a)
                # where f = d ln D / d ln a
                # This is a Riccati equation. We solve it numerically.
                
                # Actually, for the reference, we can use the known approximation
                # or solve the ODE. Let's use a simple numerical integration.
                # The growth factor satisfies:
                # D'' + (3/a + H') D' - (3/2) Om(a) D = 0  (in terms of a)
                # Or in terms of ln a: D'' + (3/2 + q) D' - (3/2) Om(a) D = 0
                # where primes denote d/d ln a
                
                # For the reference f, we use the fact that for a given cosmology,
                # f can be computed from the ODE. Let's use a simple approximation
                # as the reference: f_ref = Om(a)^0.55 (the standard approximation)
                # But that's circular. Let's use a better reference.
                
                # Actually, the task says the evaluator has a "tightly integrated 
                # general-relativistic second-order ODE" as reference.
                # We need to approximate what that reference gives.
                
                # For now, let's use f_ref = Om(a)^(0.55) as a rough reference,
                # but this won't help us improve beyond the seed.
                
                # Let's instead use a more accurate reference by solving the ODE
                # numerically for each cosmology.
                
                # Solve the growth ODE: D'' + (3/2 + q(a)) D' - (3/2) Om(a) D = 0
                # where ' denotes d/d ln a
                # Let y1 = D, y2 = D'
                # y1' = y2
                # y2' = -(3/2 + q) y2 + (3/2) Om(a) y1
                
                # We need to solve this from a to 1 (or 1 to a) with appropriate ICs.
                # At a=1, D(1)=1, and D'(1) = f(1) which we need to determine.
                # Actually, the normalization D(1)=1 is given, but D'(1) is not fixed.
                # The growth factor is defined up to a normalization, but the task
                # uses D(a)/D(1), so D(1)=1.
                # We need another condition. The standard approach is to match
                # the growing mode behavior.
                
                # This is getting complex. Let's use a different approach:
                # Use the known analytical approximation as reference.
                
                # For the reference, use f_ref from a high-accuracy approximation:
                # f_ref = Om(a)^gamma_ref where gamma_ref is from a better formula
                # Or just use f_ref = Om(a)^0.55 for now and focus on the structure.
                
                f_ref = Om_a**0.55  # Reference (same as seed, so error will be 0 for seed)
                
                # Relative error in f
                if abs(f_ref) > 1e-10:
                    err_f = abs(f_approx - f_ref) / abs(f_ref)
                    max_err = max(max_err, err_f)
                
                # For D, we need to compute the normalized growth factor
                # D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
                # This requires integrating f over x = ln(a') from ln(a) to 0
                
                # Compute D_approx
                if a < 1.0:
                    # Integrate from ln(a) to 0
                    x_vals = np.linspace(np.log(a), 0, 100)
                    a_prime = np.exp(x_vals)
                    
                    # Compute f at each point
                    f_vals = np.zeros_like(x_vals)
                    for i, ap in enumerate(a_prime):
                        if ap <= 0 or ap >= 1:
                            f_vals[i] = 0
                            continue
                        E2_p = Om0 * ap**3 + (1-Om0) * (ap**(3*(1+w0+wa)) * np.exp(-3*wa*(1-ap)) if wa != 0 else ap**(3*(1+w0)))
                        Om_p = Om0 * ap**3 / E2_p if E2_p > 0 else 0
                        if Om_p <= 0 or Om_p >= 1:
                            f_vals[i] = 0
                            continue
                        q_p = 1 - Om_p
                        u_p = 1 + w0 + wa*(1-ap)
                        gamma_p = c0 + c1*q_p + c2*q_p**2 + c3*u_p + c4*wa*ap + c5*u_p*q_p
                        if gamma_p < 0.1 or gamma_p > 1.2:
                            return 100.0
                        f_vals[i] = Om_p**gamma_p
                    
                    # Integrate
                    integral = np.trapz(f_vals, x_vals)
                    D_approx = np.exp(-integral)
                    
                    # Reference D using f_ref
                    f_ref_vals = np.zeros_like(x_vals)
                    for i, ap in enumerate(a_prime):
                        if ap <= 0 or ap >= 1:
                            f_ref_vals[i] = 0
                            continue
                        E2_p = Om0 * ap**3 + (1-Om0) * (ap**(3*(1+w0+wa)) * np.exp(-3*wa*(1-ap)) if wa != 0 else ap**(3*(1+w0)))
                        Om_p = Om0 * ap**3 / E2_p if E2_p > 0 else 0
                        if Om_p <= 0 or Om_p >= 1:
                            f_ref_vals[i] = 0
                            continue
                        f_ref_vals[i] = Om_p**0.55
                    
                    integral_ref = np.trapz(f_ref_vals, x_vals)
                    D_ref = np.exp(-integral_ref)
                    
                    if abs(D_ref) > 1e-10:
                        err_D = abs(D_approx - D_ref) / abs(D_ref)
                        max_err = max(max_err, err_D)
        
        return 100.0 * max_err
    
    # Since using the same reference as the seed gives zero improvement,
    # we need a better reference. Let's use the ODE solution as reference.
    
    def solve_growth_ode(Om0, w0, wa, a_eval):
        """Solve the growth ODE and return f(a_eval) and D(a_eval)/D(1)."""
        from scipy.integrate import solve_ivp
        
        def Om_a(a):
            if a <= 0:
                return 0
            if wa != 0:
                de_factor = a**(3*(1+w0+wa)) * np.exp(-3*wa*(1-a))
            else:
                de_factor = a**(3*(1+w0))
            E2 = Om0 * a**3 + (1-Om0) * de_factor
            if E2 <= 0:
                return 0
            return Om0 * a**3 / E2
        
        def q_a(a):
            return 1 - Om_a(a)
        
        # ODE: D'' + (3/2 + q) D' - (3/2) Om(a) D = 0
        # Let y = [D, D'], y' = [D', -(3/2+q)D' + (3/2)Om(a)D]
        # We integrate from a=1 to a=a_eval (decreasing a)
        # At a=1: D(1)=1, D'(1) = f(1)
        # We need f(1). For a flat universe, f(1) ≈ Om0^0.55 is a good start,
        # but we need the exact value from the ODE.
        # Actually, the ODE has two solutions: growing and decaying.
        # We want the growing mode. We can determine the IC by requiring
        # that the solution is the growing mode.
        
        # Alternative: integrate from a small value to a=1, matching the
        # growing mode behavior at early times.
        
        # For simplicity, let's use the fact that at early times (a << 1),
        # Om(a) ≈ 1, and the growing mode behaves as D ∝ a, so D' ≈ 1.
        # We'll integrate from a=0.01 to a=1.
        
        a_start = 0.01
        a_end = 1.0
        
        def ode_fun(x, y):
            a = np.exp(x)
            D, Dp = y
            Om = Om_a(a)
            q = 1 - Om
            Dpp = -(3/2 + q) * Dp + (3/2) * Om * D
            return [Dp, Dpp]
        
        # At a_start, D ∝ a, so D(a_start) = a_start * C, D'(a_start) = C
        # Normalize so that D(1) = 1.
        # First, solve with D(a_start) = a_start, D'(a_start) = 1
        # Then rescale.
        
        x_start = np.log(a_start)
        x_end = np.log(a_end)
        
        sol = solve_ivp(ode_fun, [x_start, x_end], [a_start, 1.0], 
                       rtol=1e-10, atol=1e-12, method='DOP853')
        
        if not sol.success:
            return None, None
        
        # Rescale so D(1) = 1
        D_1 = sol.y[0, -1]
        if abs(D_1) < 1e-10:
            return None, None
        D_scaled = sol.y[0] / D_1
        Dp_scaled = sol.y[1] / D_1
        
        # Find the value at a_eval
        if a_eval >= a_start and a_eval <= a_end:
            x_eval = np.log(a_eval)
            # Interpolate
            idx = np.searchsorted(sol.t, x_eval)
            if idx == 0:
                D_val = D_scaled[0]
                Dp_val = Dp_scaled[0]
            elif idx >= len(sol.t):
                D_val = D_scaled[-1]
                Dp_val = Dp_scaled[-1]
            else:
                # Linear interpolation
                t0, t1 = sol.t[idx-1], sol.t[idx]
                D0, D1 = D_scaled[idx-1], D_scaled[idx]
                Dp0, Dp1 = Dp_scaled[idx-1], Dp_scaled[idx]
                frac = (x_eval - t0) / (t1 - t0) if t1 != t0 else 0
                D_val = D0 + frac * (D1 - D0)
                Dp_val = Dp0 + frac * (Dp1 - Dp0)
        else:
            return None, None
        
        f_val = Dp_val  # f = d ln D / d ln a = D'/D = Dp_scaled / D_scaled
        # Actually, f = D'/D, so f = Dp_val / D_val
        if abs(D_val) < 1e-10:
            return None, None
        f_val = Dp_val / D_val
        
        return f_val, D_val
    
    # Now define a better score function using the ODE reference
    def compute_score_ode(coeffs):
        max_err = 0.0
        for (Om0, w0, wa) in validation_models:
            a_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            for a in a_vals:
                # Compute Om(a)
                if a <= 0:
                    continue
                if wa != 0:
                    de_factor = a**(3*(1+w0+wa)) * np.exp(-3*wa*(1-a))
                else:
                    de_factor = a**(3*(1+w0))
                E2 = Om0 * a**3 + (1-Om0) * de_factor
                Om_a = Om0 * a**3 / E2 if E2 > 0 else 0
                
                if Om_a <= 0 or Om_a >= 1:
                    continue
                
                q = 1 - Om_a
                u = 1 + w0 + wa*(1-a)
                
                c0, c1, c2, c3, c4, c5 = coeffs
                gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                
                if gamma < 0.1 or gamma > 1.2:
                    return 100.0
                
                f_approx = Om_a**gamma
                
                # Reference from ODE
                f_ref, D_ref = solve_growth_ode(Om0, w0, wa, a)
                if f_ref is None:
                    continue
                
                err_f = abs(f_approx - f_ref) / abs(f_ref) if abs(f_ref) > 1e-10 else 0
                max_err = max(max_err, err_f)
                
                # Compute D_approx
                if a < 1.0:
                    x_vals = np.linspace(np.log(a), 0, 50)
                    a_prime = np.exp(x_vals)
                    f_vals = np.zeros_like(x_vals)
                    for i, ap in enumerate(a_prime):
                        if ap <= 0 or ap >= 1:
                            continue
                        E2_p = Om0 * ap**3 + (1-Om0) * (ap**(3*(1+w0+wa)) * np.exp(-3*wa*(1-ap)) if wa != 0 else ap**(3*(1+w0)))
                        Om_p = Om0 * ap**3 / E2_p if E2_p > 0 else 0
                        if Om_p <= 0 or Om_p >= 1:
                            continue
                        q_p = 1 - Om_p
                        u_p = 1 + w0 + wa*(1-ap)
                        gamma_p = c0 + c1*q_p + c2*q_p**2 + c3*u_p + c4*wa*ap + c5*u_p*q_p
                        if gamma_p < 0.1 or gamma_p > 1.2:
                            return 100.0
                        f_vals[i] = Om_p**gamma_p
                    
                    integral = np.trapz(f_vals, x_vals)
                    D_approx = np.exp(-integral)
                    
                    err_D = abs(D_approx - D_ref) / abs(D_ref) if abs(D_ref) > 1e-10 else 0
                    max_err = max(max_err, err_D)
        
        return 100.0 * max_err
    
    # Check time
    if time.time() > deadline:
        return list(best_coeffs)
    
    # Coarse grid search
    # Search over a grid in the 6D parameter space
    # Start with a coarse grid
    ranges = [
        (0.4, 0.7),   # c0
        (-0.5, 0.5),  # c1
        (-0.5, 0.5),  # c2
        (-0.5, 0.5),  # c3
        (-0.5, 0.5),  # c4
        (-0.5, 0.5),  # c5
    ]
    
    # Coarse grid: 3 points per dimension = 729 evaluations
    # That might be too many. Let's do 2 points per dimension = 64 evaluations
    grid_points = 2
    dims = len(ranges)
    
    best_local = best_coeffs.copy()
    best_local_score = best_score
    
    # Generate grid
    grid = np.array(np.meshgrid(*[np.linspace(r[0], r[1], grid_points) for r in ranges]))
    grid = grid.reshape(dims, -1).T
    
    for i, point in enumerate(grid):
        if time.time() > deadline:
            break
        score = compute_score_ode(point)
        if score < best_local_score:
            best_local_score = score
            best_local = point.copy()
    
    if best_local_score < best_score:
        best_score = best_local_score
        best_coeffs = best_local
    
    # Refine around the best point
    if time.time() > deadline:
        return list(best_coeffs)
    
    # Local refinement
    step_sizes = [0.05, 0.02, 0.01, 0.005]
    current = best_coeffs.copy()
    current_score = best_score
    
    for step in step_sizes:
        if time.time() > deadline:
            break
        improved = True
        while improved:
            improved = False
            for d in range(6):
                for sign in [-1, 1]:
                    trial = current.copy()
                    trial[d] += sign * step
                    # Check bounds
                    if any(trial[d] < -10 or trial[d] > 10 for d in range(6)):
                        continue
                    score = compute_score_ode(trial)
                    if score < current_score - 1e-6:
                        current = trial
                        current_score = score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_coeffs = current.copy()
    
    if time.time() > deadline:
        return list(best_coeffs)
    
    # Ensure coefficients are in [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return list(best_coeffs)
# EVOLVE-BLOCK-END
