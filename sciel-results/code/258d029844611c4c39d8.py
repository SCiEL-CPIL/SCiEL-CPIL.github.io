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
    then refine with a bounded optimization using a smooth soft-max surrogate for worst-case error."""
    t_start = time.time()
    deadline = 75.0  # Safety margin before 90s kill
    
    # Grid of CPL parameters and scale factors - use non-uniform spacing for better coverage
    w0_vals = np.linspace(-1.2, -0.8, 25)
    wa_vals = np.linspace(-0.5, 0.5, 25)
    # Non-uniform a spacing: denser near a=1 and a=0.3 where errors are typically largest
    a_base = np.linspace(0.3, 1.0, 50)
    a_vals = np.unique(np.concatenate([a_base, np.linspace(0.3, 0.5, 15), np.linspace(0.9, 1.0, 15)]))
    a_vals = np.sort(a_vals)
    
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
    
    # Local refinement: minimize worst-case error using a smooth soft-max surrogate
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
            n_eval = len(eval_rows)
            
            # Soft-max surrogate: log-sum-exp of squared errors
            # This is a smooth upper bound on max error that becomes tight as temperature -> 0
            def soft_max_surrogate(coeffs_vec, temperature=0.01):
                if time.time() - t_start > deadline:
                    return 1e10, np.zeros(6)
                
                gamma_pred = A_eval @ coeffs_vec
                errors = np.abs(gamma_pred - b_eval)
                
                # Check constraint with penalty
                gamma_min = np.min(gamma_pred)
                gamma_max = np.max(gamma_pred)
                penalty = 0.0
                if gamma_min < 0.1:
                    penalty += (0.1 - gamma_min) ** 2 * 1000
                if gamma_max > 1.2:
                    penalty += (gamma_max - 1.2) ** 2 * 1000
                
                # Log-sum-exp for smooth max
                scaled_errors = errors / temperature
                max_scaled = np.max(scaled_errors)
                log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled_errors - max_scaled)))
                soft_max_val = log_sum_exp * temperature
                
                obj = soft_max_val + penalty
                return obj, None
            
            # Gradient of the soft-max surrogate
            def soft_max_gradient(coeffs_vec, temperature=0.01):
                if time.time() - t_start > deadline:
                    return None
                
                gamma_pred = A_eval @ coeffs_vec
                errors = np.abs(gamma_pred - b_eval)
                
                # Check constraint
                gamma_min = np.min(gamma_pred)
                gamma_max = np.max(gamma_pred)
                
                # Soft-max weights
                scaled_errors = errors / temperature
                max_scaled = np.max(scaled_errors)
                exp_vals = np.exp(scaled_errors - max_scaled)
                weights = exp_vals / np.sum(exp_vals)
                
                # Gradient of soft-max w.r.t. coefficients
                # d(soft_max)/d(coeffs) = A_eval.T @ (weights * sign(errors))
                sign_errors = np.sign(gamma_pred - b_eval)
                grad = A_eval.T @ (weights * sign_errors)
                
                # Add constraint gradient
                if gamma_min < 0.1:
                    # Find which points violate lower bound
                    lower_violation = (gamma_pred < 0.1).astype(float)
                    grad += 2.0 * 1000 * (0.1 - gamma_min) * A_eval.T @ lower_violation / n_eval
                if gamma_max > 1.2:
                    upper_violation = (gamma_pred > 1.2).astype(float)
                    grad += 2.0 * 1000 * (gamma_max - 1.2) * A_eval.T @ upper_violation / n_eval
                
                return grad
            
            # Bounds for each coefficient
            bounds = [(-10, 10)] * 6
            
            # Try local refinement starting from LS solution
            try:
                # First pass: L-BFGS-B with the smooth surrogate
                result = minimize(soft_max_surrogate, coeffs, method='L-BFGS-B', bounds=bounds,
                                 jac=lambda x: soft_max_gradient(x),
                                 options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                if result.success and result.fun < 1e9:
                    refined_coeffs = np.clip(result.x, -10.0, 10.0)
                    if _check_gamma_range(refined_coeffs):
                        coeffs = refined_coeffs
            except Exception:
                pass
            
            # Second pass: high-precision polish with tighter temperature
            if time.time() - t_start < deadline - 2:
                try:
                    def soft_max_polish(coeffs_vec, temperature=0.001):
                        if time.time() - t_start > deadline:
                            return 1e10, None
                        gamma_pred = A_eval @ coeffs_vec
                        errors = np.abs(gamma_pred - b_eval)
                        gamma_min = np.min(gamma_pred)
                        gamma_max = np.max(gamma_pred)
                        penalty = 0.0
                        if gamma_min < 0.1:
                            penalty += (0.1 - gamma_min) ** 2 * 10000
                        if gamma_max > 1.2:
                            penalty += (gamma_max - 1.2) ** 2 * 10000
                        scaled_errors = errors / temperature
                        max_scaled = np.max(scaled_errors)
                        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled_errors - max_scaled)))
                        soft_max_val = log_sum_exp * temperature
                        return soft_max_val + penalty, None
                    
                    def soft_max_polish_grad(coeffs_vec, temperature=0.001):
                        if time.time() - t_start > deadline:
                            return None
                        gamma_pred = A_eval @ coeffs_vec
                        errors = np.abs(gamma_pred - b_eval)
                        gamma_min = np.min(gamma_pred)
                        gamma_max = np.max(gamma_pred)
                        scaled_errors = errors / temperature
                        max_scaled = np.max(scaled_errors)
                        exp_vals = np.exp(scaled_errors - max_scaled)
                        weights = exp_vals / np.sum(exp_vals)
                        sign_errors = np.sign(gamma_pred - b_eval)
                        grad = A_eval.T @ (weights * sign_errors)
                        if gamma_min < 0.1:
                            lower_violation = (gamma_pred < 0.1).astype(float)
                            grad += 2.0 * 10000 * (0.1 - gamma_min) * A_eval.T @ lower_violation / n_eval
                        if gamma_max > 1.2:
                            upper_violation = (gamma_pred > 1.2).astype(float)
                            grad += 2.0 * 10000 * (gamma_max - 1.2) * A_eval.T @ upper_violation / n_eval
                        return grad
                    
                    result2 = minimize(soft_max_polish, coeffs, method='L-BFGS-B', bounds=bounds,
                                      jac=lambda x: soft_max_polish_grad(x),
                                      options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
                    if result2.success and result2.fun < 1e9:
                        refined_coeffs2 = np.clip(result2.x, -10.0, 10.0)
                        if _check_gamma_range(refined_coeffs2):
                            coeffs = refined_coeffs2
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
