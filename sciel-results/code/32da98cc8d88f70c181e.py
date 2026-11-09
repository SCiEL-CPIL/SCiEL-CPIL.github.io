"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_Om_a(a, w0, wa, Om0=0.3):
    """Compute Omega_m(a) for flat CPL universe."""
    u = 1.0 + w0 + wa * (1.0 - a)
    if u <= 0.01:
        return None
    Om_de0 = 1.0 - Om0
    denom = Om0 * a**3 + Om_de0 / u
    if denom <= 0:
        return None
    Om_a = Om0 * a**3 / denom
    if Om_a <= 0 or Om_a >= 1:
        return None
    return Om_a

def _target_gamma_LW(Om_a, u):
    """Linder-Waugh target gamma."""
    if u <= 0.01 or Om_a <= 0:
        return None
    return (5.0/7.0) * (Om_a / u) ** 0.38

def _gamma_from_coeffs(coeffs, q, u, wa, a):
    """Compute gamma from coefficients."""
    c = coeffs
    return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q

def _fit_coefficients():
    """Fit the six growth-index coefficients via least squares to the Linder-Waugh formula,
    then refine with a bounded optimization to minimize worst-case error."""
    t_start = time.time()
    deadline = 75.0  # Safety margin before 90s kill
    
    # Grid of CPL parameters and scale factors - denser grid
    w0_vals = np.linspace(-1.2, -0.8, 30)
    wa_vals = np.linspace(-0.5, 0.5, 30)
    a_vals = np.linspace(0.3, 1.0, 60)
    
    Omega_m0 = 0.3
    Omega_de0 = 1.0 - Omega_m0
    
    # Build the design matrix and target vector
    rows = []
    targets = []
    
    for w0 in w0_vals:
        for wa in wa_vals:
            for a in a_vals:
                if time.time() - t_start > deadline:
                    break
                
                # Compute u = 1 + w0 + wa*(1-a)
                u = 1.0 + w0 + wa * (1.0 - a)
                if u <= 0.05:  # Avoid singularities
                    continue
                
                # Compute Omega_m(a) for flat CPL universe
                Om_a = _compute_Om_a(a, w0, wa, Omega_m0)
                if Om_a is None:
                    continue
                
                q = 1.0 - Om_a
                target_gamma = _target_gamma_LW(Om_a, u)
                if target_gamma is None:
                    continue
                
                # Design matrix row: [1, q, q^2, u, wa*a, u*q]
                row = np.array([1.0, q, q**2, u, wa * a, u * q])
                rows.append(row)
                targets.append(target_gamma)
            
            if time.time() - t_start > deadline:
                break
        if time.time() - t_start > deadline:
            break
    
    if len(rows) < 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    A = np.array(rows)
    b = np.array(targets)
    
    # Solve least squares
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    
    # Clip to [-10, 10]
    coeffs = np.clip(coeffs, -10.0, 10.0)
    
    # Verify gamma stays in [0.1, 1.2] on the training grid
    def _check_gamma_range(coeffs_to_check):
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_vals:
                    u = 1.0 + w0 + wa * (1.0 - a)
                    if u <= 0.05:
                        continue
                    Om_a = _compute_Om_a(a, w0, wa, Omega_m0)
                    if Om_a is None:
                        continue
                    q = 1.0 - Om_a
                    gamma = _gamma_from_coeffs(coeffs_to_check, q, u, wa, a)
                    if gamma < 0.1 or gamma > 1.2:
                        return False
        return True
    
    if not _check_gamma_range(coeffs):
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Local refinement: minimize worst-case relative error using scipy
    if time.time() - t_start < deadline - 5:
        # Prepare evaluation points for refinement
        eval_rows = []
        eval_targets = []
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_vals:
                    u = 1.0 + w0 + wa * (1.0 - a)
                    if u <= 0.05:
                        continue
                    Om_a = _compute_Om_a(a, w0, wa, Omega_m0)
                    if Om_a is None:
                        continue
                    q = 1.0 - Om_a
                    target_gamma = _target_gamma_LW(Om_a, u)
                    if target_gamma is None:
                        continue
                    row = np.array([1.0, q, q**2, u, wa * a, u * q])
                    eval_rows.append(row)
                    eval_targets.append(target_gamma)
        
        if len(eval_rows) > 0:
            A_eval = np.array(eval_rows)
            b_eval = np.array(eval_targets)
            
            # Objective: minimize max absolute error (surrogate for worst-case)
            def objective(coeffs_vec):
                if time.time() - t_start > deadline:
                    return 1e10
                gamma_pred = A_eval @ coeffs_vec
                errors = np.abs(gamma_pred - b_eval)
                # Check constraint
                if np.any(gamma_pred < 0.1) or np.any(gamma_pred > 1.2):
                    return 1e10
                return np.max(errors)
            
            # Bounds for each coefficient
            bounds = [(-10, 10)] * 6
            
            # Try local refinement starting from LS solution
            try:
                result = minimize(objective, coeffs, method='L-BFGS-B', bounds=bounds,
                                 options={'maxiter': 100, 'ftol': 1e-10})
                if result.success and result.fun < 1e9:
                    refined_coeffs = np.clip(result.x, -10.0, 10.0)
                    if _check_gamma_range(refined_coeffs):
                        coeffs = refined_coeffs
            except Exception:
                pass
    
    # Final verification
    if not _check_gamma_range(coeffs):
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return coeffs.tolist()

def construct():
    """Construct the six growth-index coefficients."""
    t_start = time.time()
    
    try:
        coeffs = _fit_coefficients()
    except Exception:
        coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Ensure finite and in range
    for i in range(6):
        if not np.isfinite(coeffs[i]) or coeffs[i] < -10 or coeffs[i] > 10:
            coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
            break
    
    return coeffs
# EVOLVE-BLOCK-END
