"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    """Fit six coefficients to minimize worst-case relative error in f and D."""
    t_start = time.time()
    deadline = t_start + 75.0  # 15s safety margin before 90s kill

    # Grid of CPL parameters (Om0, w0, wa)
    Om0_vals = np.array([0.15, 0.2, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40])
    w0_vals = np.array([-1.5, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7])
    wa_vals = np.array([-0.8, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])

    # Scale factor grid for fitting
    a_vals = np.linspace(0.05, 0.995, 50)

    # Build design matrix and target
    # Use Linder's improved approximation: gamma = (5/7) * (Om(a)/u)^(4/7)
    # Our ansatz: gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # where q = 1 - Om(a)

    rows = []
    targets = []

    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_vals:
                    a_cubed = a**3
                    de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                    E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                    
                    if E2 <= 0:
                        continue
                    Om_a = Om0 * a_cubed / E2
                    q = 1.0 - Om_a
                    u = 1.0 + w0 + wa * (1.0 - a)
                    
                    if u <= 0:
                        continue
                    
                    # Linder-Van de Stark: gamma = (5/7) * (Om(a)/u)^(4/7)
                    gamma_target = (5.0/7.0) * (Om_a / u)**(4.0/7.0)
                    
                    row = [1.0, q, q*q, u, wa*a, u*q]
                    rows.append(row)
                    targets.append(gamma_target)
                    
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if len(rows) < 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    A = np.array(rows)
    b = np.array(targets)

    # Solve least squares as initial guess
    try:
        coeffs0, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    coeffs0 = np.clip(coeffs0, -10.0, 10.0)

    # Define helper functions for evaluation
    def _compute_Om_a(Om0, w0, wa, a):
        a_cubed = a**3
        de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
        E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
        if E2 <= 0:
            return None
        return Om0 * a_cubed / E2

    def _gamma_from_coeffs(c, q, u, wa, a):
        return c[0] + c[1]*q + c[2]*q*q + c[3]*u + c[4]*wa*a + c[5]*u*q

    def _f_value(c, Om_a, q, u, wa, a):
        g = _gamma_from_coeffs(c, q, u, wa, a)
        if g < 0.1:
            g = 0.1
        elif g > 1.2:
            g = 1.2
        return Om_a**g

    def _D_normalized(c, Om0, w0, wa, a_eval):
        """Compute D(a)/D(1) using the ansatz gamma."""
        # Integrate from a_eval to 1: -∫ f(exp(x)) dx where x = ln(a)
        # f = Om(a)^gamma
        # Use numerical integration
        if a_eval >= 1.0:
            return 1.0
        if a_eval <= 0:
            return 0.0
        
        # Integrate from ln(a_eval) to 0
        x_eval = np.log(a_eval)
        # Use trapezoidal rule with enough points
        n_pts = 50
        x_grid = np.linspace(x_eval, 0.0, n_pts)
        a_grid = np.exp(x_grid)
        
        f_vals = []
        for a_i in a_grid:
            Om_a_i = _compute_Om_a(Om0, w0, wa, a_i)
            if Om_a_i is None:
                return None
            q_i = 1.0 - Om_a_i
            u_i = 1.0 + w0 + wa * (1.0 - a_i)
            f_i = _f_value(c, Om_a_i, q_i, u_i, wa, a_i)
            f_vals.append(f_i)
        
        f_arr = np.array(f_vals)
        # D(a)/D(1) = exp(-∫_{ln(a)}^0 f(exp(x)) dx)
        integral = np.trapz(f_arr, x_grid)
        return np.exp(-integral)

    def _compute_max_error(c):
        """Compute 100 * max relative error over validation grid."""
        max_err = 0.0
        for Om0 in Om0_vals:
            for w0 in w0_vals:
                for wa in wa_vals:
                    for a in a_vals:
                        Om_a = _compute_Om_a(Om0, w0, wa, a)
                        if Om_a is None:
                            continue
                        q = 1.0 - Om_a
                        u = 1.0 + w0 + wa * (1.0 - a)
                        if u <= 0:
                            continue
                        
                        # True f (using Linder gamma)
                        g_true = (5.0/7.0) * (Om_a / u)**(4.0/7.0)
                        f_true = Om_a**g_true
                        
                        # Our f
                        f_ours = _f_value(c, Om_a, q, u, wa, a)
                        
                        if f_true > 0:
                            err_f = abs(f_ours - f_true) / f_true
                        else:
                            err_f = 0.0
                        
                        max_err = max(max_err, err_f)
                        
                        # True D using ODE (approximate with same method but true gamma)
                        # For speed, only check D on a subset
                        if a in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
                            # Compute true D
                            x_eval = np.log(a)
                            n_pts = 50
                            x_grid = np.linspace(x_eval, 0.0, n_pts)
                            a_grid = np.exp(x_grid)
                            
                            f_true_vals = []
                            valid = True
                            for a_i in a_grid:
                                Om_a_i = _compute_Om_a(Om0, w0, wa, a_i)
                                if Om_a_i is None:
                                    valid = False
                                    break
                                q_i = 1.0 - Om_a_i
                                u_i = 1.0 + w0 + wa * (1.0 - a_i)
                                if u_i <= 0:
                                    valid = False
                                    break
                                g_i = (5.0/7.0) * (Om_a_i / u_i)**(4.0/7.0)
                                f_true_vals.append(Om_a_i**g_i)
                            
                            if not valid:
                                continue
                            
                            f_true_arr = np.array(f_true_vals)
                            integral_true = np.trapz(f_true_arr, x_grid)
                            D_true = np.exp(-integral_true)
                            
                            D_ours = _D_normalized(c, Om0, w0, wa, a)
                            if D_ours is not None and D_true > 0:
                                err_D = abs(D_ours - D_true) / D_true
                                max_err = max(max_err, err_D)
                        
                        if time.time() > deadline:
                            return max_err
                    if time.time() > deadline:
                        return max_err
                if time.time() > deadline:
                    return max_err
            if time.time() > deadline:
                return max_err
        return max_err

    # Objective: minimize 100 * max_error
    def objective(c):
        c_clipped = np.clip(c, -10.0, 10.0)
        err = _compute_max_error(c_clipped)
        return 100.0 * err

    # Local polish using Nelder-Mead
    best_coeffs = coeffs0.copy()
    best_score = objective(best_coeffs)
    
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 200, 'fatol': 1e-8, 'xatol': 1e-8}
            )
            if result.success or result.fun < best_score:
                candidate = np.clip(result.x, -10.0, 10.0)
                score = objective(candidate)
                if score < best_score:
                    best_coeffs = candidate
                    best_score = score
        except Exception:
            pass

    # Verify gamma stays in [0.1, 1.2]
    valid = True
    for Om0 in Om0_vals[:4]:
        for w0 in w0_vals[:4]:
            for wa in wa_vals[:4]:
                for a in np.linspace(0.05, 0.995, 15):
                    Om_a = _compute_Om_a(Om0, w0, wa, a)
                    if Om_a is None:
                        continue
                    q = 1.0 - Om_a
                    u = 1.0 + w0 + wa * (1.0 - a)
                    gamma = _gamma_from_coeffs(best_coeffs, q, u, wa, a)
                    if gamma < 0.1 or gamma > 1.2:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    return [float(c) for c in best_coeffs]
# EVOLVE-BLOCK-END
