# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Precompute quadrature nodes and weights
_N_GL = 70
_N_PHI = 130
_x_gl, _w_gl = np.polynomial.legendre.leggauss(_N_GL)
_theta_gl = np.arccos(_x_gl)
_phi_grid = 2 * np.pi * np.arange(_N_PHI) / _N_PHI
_dphi = 2 * np.pi / _N_PHI

# Precompute spherical harmonic values at all quadrature points
# Shape: (21, 70, 130) for m = -10..10
_k = 10
_N_COEFFS = 2 * _k + 1
_Y_precomputed = np.zeros((_N_COEFFS, _N_GL, _N_PHI))

for idx, m in enumerate(range(-_k, _k + 1)):
    abs_m = abs(m)
    P = lpmv(abs_m, _k, _x_gl)  # shape (70,)
    
    n_fact = factorial(_k - abs_m, exact=True)
    d_fact = factorial(_k + abs_m, exact=True)
    N_m = np.sqrt((2 * _k + 1) / (4 * np.pi) * (n_fact / d_fact))
    
    if m == 0:
        _Y_precomputed[idx] = N_m * P[:, None]
    elif m > 0:
        _Y_precomputed[idx] = np.sqrt(2) * N_m * P[:, None] * np.cos(m * _phi_grid)[None, :]
    else:
        _Y_precomputed[idx] = np.sqrt(2) * N_m * P[:, None] * np.sin(abs_m * _phi_grid)[None, :]

# Precompute weights for norm calculations
_W_matrix = _w_gl[:, None] * _dphi

# Precompute Y^T W Y (Gram matrix) and Y^T W Y^3 type matrices for Hessian
# G_ij = sum_p Y_i(p) Y_j(p) W(p) = delta_ij (orthonormal)
# But we need: Hessian of objective involves f^2 * Y_i * Y_j weighted
# Let's precompute the matrix M where M[i,j] = sum_p Y_i(p)*Y_j(p)*W(p) = I (orthonormal)
# For Hessian: d2R/dc_i dc_j involves terms like f^2*Y_i*Y_j and Y_i*Y_j*f^0
# Actually let's just compute on the fly since 21x21 is small

def _compute_ratio(coeffs):
    """
    Compute R = ||f||_4 / ||f||_2 using precomputed quadrature.
    coeffs: length 21 vector c_m for m=-10..10
    Returns: ratio R (lower is better)
    """
    if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)):
        return np.inf
    
    f_val = np.tensordot(coeffs, _Y_precomputed, axes=(0, 0))
    
    f2 = f_val ** 2
    f4 = f_val ** 4
    
    norm2_sq = np.sum(f2 * _W_matrix)
    norm4_4 = np.sum(f4 * _W_matrix)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    
    return norm4 / norm2


def _compute_ratio_and_grad(coeffs):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient with respect to coeffs.
    """
    if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)):
        return np.inf, np.zeros_like(coeffs)
    
    f_val = np.tensordot(coeffs, _Y_precomputed, axes=(0, 0))
    
    f2 = f_val ** 2
    f4 = f_val ** 4
    
    norm2_sq = np.sum(f2 * _W_matrix)
    norm4_4 = np.sum(f4 * _W_matrix)
    
    if norm2_sq <= 0:
        return np.inf, np.zeros_like(coeffs)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    
    R = norm4 / norm2
    
    weighted_f3 = (f_val ** 3) * _W_matrix
    dN4_dc = np.tensordot(_Y_precomputed, weighted_f3, axes=([1, 2], [0, 1]))
    
    weighted_f = f_val * _W_matrix
    dN2_dc = np.tensordot(_Y_precomputed, weighted_f, axes=([1, 2], [0, 1]))
    
    grad = dN4_dc / (norm2 * norm4**3) - norm4 * dN2_dc / (norm2**3)
    
    return R, grad


def _compute_hessian(coeffs):
    """
    Compute the Hessian of R = ||f||_4/||f||_2 with respect to coeffs.
    Uses the formula for d2(N4/N2)/dc_i dc_j.
    
    Let A = N4 = (sum f^4 W)^(1/4), B = N2 = (sum f^2 W)^(1/2)
    R = A/B
    
    dR/dc = (dA/dc)/B - A*dB/dc/B^2
    
    d2R/dc_i dc_j = d2A/(B^2) - 2*dA*dB/(B^3) + A*d2B/(B^3) + 2*A*(dB)^2/(B^4)
    
    where dA/dc_i = (1/(4 A^3)) * sum 4 f^3 Y_i W = (1/A^3) sum f^3 Y_i W
    dB/dc_i = (1/(2 B)) * sum 2 f Y_i W = (1/B) sum f Y_i W
    
    d2A/dc_i dc_j = (1/A^3) sum 3 f^2 Y_i Y_j W - 3/A^5 (sum f^3 Y_i W)(sum f^3 Y_j W)
    d2B/dc_i dc_j = (1/B) sum Y_i Y_j W - 1/B^3 (sum f Y_i W)(sum f Y_j W)
    """
    f_val = np.tensordot(coeffs, _Y_precomputed, axes=(0, 0))
    
    f2 = f_val ** 2
    f3 = f_val ** 3
    f4 = f_val ** 4
    
    norm2_sq = np.sum(f2 * _W_matrix)
    norm4_4 = np.sum(f4 * _W_matrix)
    
    if norm2_sq <= 0 or norm4_4 <= 0:
        return np.zeros((_N_COEFFS, _N_COEFFS))
    
    B = np.sqrt(norm2_sq)  # N2
    A = norm4_4 ** 0.25   # N4
    
    # Precompute weighted sums
    # S1_i = sum f Y_i W (shape 21)
    S1 = np.tensordot(_Y_precomputed, (f_val * _W_matrix), axes=([1, 2], [0, 1]))
    # S2_i = sum f^3 Y_i W (shape 21)
    S2 = np.tensordot(_Y_precomputed, (f3 * _W_matrix), axes=([1, 2], [0, 1]))
    
    # S0_ij = sum Y_i Y_j W = I (orthonormal), but let's compute for safety
    # Actually due to quadrature it should be identity. Let's use identity.
    
    # dA/dc_i = S2_i / A^3
    # dB/dc_i = S1_i / B
    
    # d2A/dc_i dc_j = 3/A^3 * sum(f^2 Y_i Y_j W) - 3/A^5 * S2_i * S2_j
    # d2B/dc_i dc_j = 1/B * sum(Y_i Y_j W) - 1/B^3 * S1_i * S1_j
    
    # Compute sum(f^2 Y_i Y_j W) for all i,j
    # This is: for each quadrature point p, f^2(p)*W(p) * Y_i(p)*Y_j(p), summed over p
    # = (Y^T * diag(f^2*W) * Y)
    fw = f2 * _W_matrix  # shape (70, 130)
    # M2[i,j] = sum_{p} Y_i(p) * Y_j(p) * fw(p)
    # Efficient: M2 = Y_flat^T @ diag(fw_flat) @ Y_flat
    # But that's 21x9100 @ 9100x9100 @ 9100x21 which is expensive
    # Better: M2[i,j] = sum_p Y_i(p) Y_j(p) fw(p)
    # = sum over all (theta_idx, phi_idx)
    # Let's reshape: Y_flat[i, k] where k indexes (theta, phi)
    Y_flat = _Y_precomputed.reshape(_N_COEFFS, -1)  # (21, 9100)
    fw_flat = fw.reshape(-1)  # (9100,)
    M2 = Y_flat * fw_flat[None, :]  # (21, 9100)
    M2 = M2 @ Y_flat.T  # (21, 21)
    
    # M0[i,j] = sum Y_i Y_j W = I (by orthonormality + exact quadrature)
    M0 = np.eye(_N_COEFFS)
    
    # Now compute Hessian
    # d2A = 3/A^3 * M2 - 3/A^5 * outer(S2, S2)
    d2A = (3.0 / A**3) * M2 - (3.0 / A**5) * np.outer(S2, S2)
    
    # d2B = 1/B * M0 - 1/B^3 * outer(S1, S1)
    d2B = (1.0 / B) * M0 - (1.0 / B**3) * np.outer(S1, S1)
    
    # dR/dc vector
    dR = S2 / (B * A**3) - A * S1 / (B**3)
    
    # d2R = d2A/B^2 - 2*outer(dR, dB)/B^3 + A*d2B/B^3 + 2*A*outer(dR, dR)/B^4
    # Wait, let me redo this more carefully.
    # R = A/B
    # dR/dc = (dA/dc)/B - A*(dB/dc)/B^2
    # d2R/dc_i dc_j = (d2A_dcij)/B - (dA_dc_i)*(dB_dc_j)/B^2 - (dA_dc_j)*(dB_dc_i)/B^2
    #                 - A*(d2B_dcij)/B^2 + 2A*(dB_dc_i)*(dB_dc_j)/B^3
    #                 + A*(dB_dc_i)*(dB_dc_j)/B^3  -- no wait
    
    # Let me use product rule properly:
    # R = A * B^(-1)
    # dR/dc_i = dA_i * B^(-1) + A * (-1) * B^(-2) * dB_i
    # d2R/dc_i dc_j = d2A_ij * B^(-1) + dA_i * (-1)*B^(-2)*dB_j + dA_j * (-1)*B^(-2)*dB_i
    #                + A * [(-1)*(-2)*B^(-3)*dB_i*dB_j + (-1)*B^(-2)*d2B_ij]
    # = d2A_ij/B - dA_i*dB_j/B^2 - dA_j*dB_i/B^2
    #   + A * [2*dB_i*dB_j/B^3 - d2B_ij/B^2]
    
    dA = S2 / A**3  # dA/dc_i
    dB = S1 / B     # dB/dc_i
    
    H = d2A / B - (np.outer(dA, dB) + np.outer(dB, dA)) / B**2
    H += A * (2.0 * np.outer(dB, dB) / B**3 - d2B / B**2)
    
    return H


def _newton_on_sphere(c0, max_iter=100, tol=1e-14, deadline=None):
    """
    Newton-type method on the unit sphere for minimizing R = ||f||_4/||f||_2.
    Uses the Hessian of the Lagrangian projected to the tangent space.
    Quadratic convergence near the optimum.
    """
    c = c0 / np.linalg.norm(c0)
    R, grad = _compute_ratio_and_grad(c)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Project gradient to tangent space
        grad_tan = grad - np.dot(grad, c) * c
        grad_tan_norm = np.linalg.norm(grad_tan)
        
        if grad_tan_norm < tol:
            break
        
        # Compute Hessian
        H = _compute_hessian(c)
        
        # Project Hessian to tangent space: H_tan = (I - cc^T) H (I - cc^T)
        P = np.eye(_N_COEFFS) - np.outer(c, c)
        H_tan = P @ H @ P
        
        # Solve H_tan * s = -grad_tan
        # Add damping for numerical stability
        reg = 1e-10
        try:
            s = np.linalg.solve(H_tan + reg * np.eye(_N_COEFFS), -grad_tan)
        except np.linalg.LinAlgError:
            s = -grad_tan / grad_tan_norm
        
        # Ensure s is in tangent space
        s = s - np.dot(s, c) * c
        
        # Check if it's a descent direction
        if np.dot(s, grad_tan) > 0:
            s = -grad_tan / grad_tan_norm
        
        s_norm = np.linalg.norm(s)
        if s_norm < 1e-15:
            break
        
        s = s / s_norm
        
        # Line search: c_new = cos(alpha)*c + sin(alpha)*s
        # Find alpha that minimizes R
        best_alpha = 0.0
        best_R = R
        
        # Try a range of alphas
        alphas = np.concatenate([
            np.linspace(0.001, 0.5, 20),
            np.linspace(0.5, 1.5, 10)
        ])
        
        for alpha in alphas:
            c_trial = np.cos(alpha) * c + np.sin(alpha) * s
            c_trial_norm = np.linalg.norm(c_trial)
            if c_trial_norm < 1e-15:
                continue
            c_trial = c_trial / c_trial_norm
            R_trial = _compute_ratio(c_trial)
            if R_trial < best_R:
                best_R = R_trial
                best_alpha = alpha
        
        if best_alpha == 0.0:
            # No improvement, try smaller steps
            for alpha in np.linspace(0.0001, 0.01, 10):
                c_trial = np.cos(alpha) * c + np.sin(alpha) * s
                c_trial_norm = np.linalg.norm(c_trial)
                if c_trial_norm < 1e-15:
                    continue
                c_trial = c_trial / c_trial_norm
                R_trial = _compute_ratio(c_trial)
                if R_trial < best_R:
                    best_R = R_trial
                    best_alpha = alpha
            
            if best_alpha == 0.0:
                break
        
        c = np.cos(best_alpha) * c + np.sin(best_alpha) * s
        c = c / np.linalg.norm(c)
        R = best_R
        
        # Check convergence
        R, grad = _compute_ratio_and_grad(c)
    
    return c, R


def _trust_region_optimize(c0, max_iter=200, tol=1e-12, deadline=None):
    """
    Trust-region method on the unit sphere for minimizing R = ||f||_4/||f||_2.
    """
    c = c0 / np.linalg.norm(c0)
    R, grad = _compute_ratio_and_grad(c)
    
    # Project gradient to tangent space
    grad_tan = grad - np.dot(grad, c) * c
    
    # Initial trust region radius
    delta = 0.1
    min_delta = 1e-15
    max_delta = 0.5
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad_tan_norm = np.linalg.norm(grad_tan)
        if grad_tan_norm < tol:
            break
        
        # Try to take a full step in the negative gradient direction
        step_dir = -grad_tan / grad_tan_norm
        
        # Try different step sizes
        best_alpha = None
        best_R = R
        best_c = c.copy()
        
        # Try a range of step sizes
        for alpha in np.concatenate([
            np.linspace(0.01, delta, 10),
            np.linspace(delta, min(delta * 2, max_delta), 5)
        ]):
            if np.sin(alpha) > 1:
                continue
            c_trial = np.cos(alpha) * c + np.sin(alpha) * step_dir
            c_trial = c_trial / np.linalg.norm(c_trial)
            
            R_trial, _ = _compute_ratio_and_grad(c_trial)
            if R_trial < best_R:
                best_R = R_trial
                best_alpha = alpha
                best_c = c_trial.copy()
        
        if best_alpha is None:
            # No improvement, shrink trust region
            delta *= 0.5
            if delta < min_delta:
                break
            continue
        
        # Accept step and adjust trust region
        c = best_c
        R = best_R
        
        # Update gradient
        R, grad = _compute_ratio_and_grad(c)
        grad_tan = grad - np.dot(grad, c) * c
        
        # Adjust trust region based on improvement
        if best_R < R - 1e-15:
            delta = min(delta * 1.5, max_delta)
        else:
            delta = max(delta * 0.7, min_delta)
    
    return c, R


def rudin_problem10() -> np.ndarray:
    """
    Find the real coefficients (c_m) for m = -10..10 of a real degree-10 spherical harmonic
    that MINIMIZE the norm ratio ||f||_4 / ||f||_2.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin
    
    best_coeffs = None
    best_ratio = np.inf
    
    def _update_best(c, r):
        nonlocal best_coeffs, best_ratio
        if r < best_ratio and np.all(np.isfinite(c)):
            best_ratio = r
            best_coeffs = c.copy()
    
    n_coeffs = _N_COEFFS
    rng = np.random.default_rng(12345)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. All ones
    c0 = np.ones(n_coeffs)
    candidates.append(c0)
    
    # 2. Single modes
    for m_idx in range(n_coeffs):
        c_single = np.zeros(n_coeffs)
        c_single[m_idx] = 1.0
        candidates.append(c_single)
    
    # 3. Pairs of modes
    for i in range(n_coeffs):
        for j in range(i + 1, n_coeffs):
            c_pair = np.zeros(n_coeffs)
            c_pair[i] = 1.0
            c_pair[j] = 1.0
            candidates.append(c_pair)
    
    # 4. Random normalized vectors
    for _ in range(50):
        c_rand = rng.normal(size=n_coeffs)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)
    
    # 5. Structured patterns
    # Even m only
    c_even = np.zeros(n_coeffs)
    c_even[::2] = 1.0
    candidates.append(c_even)
    
    # Odd m only
    c_odd = np.zeros(n_coeffs)
    c_odd[1::2] = 1.0
    candidates.append(c_odd)
    
    # Positive m only
    c_pos = np.zeros(n_coeffs)
    c_pos[10:] = 1.0
    candidates.append(c_pos)
    
    # Negative m only
    c_neg = np.zeros(n_coeffs)
    c_neg[:11] = 1.0
    candidates.append(c_neg)
    
    # 6. Sparse patterns with specific structures
    for _ in range(30):
        n_nonzero = rng.integers(3, 9)
        indices = rng.choice(n_coeffs, size=n_nonzero, replace=False)
        c_sparse = np.zeros(n_coeffs)
        c_sparse[indices] = rng.normal(size=n_nonzero)
        c_sparse /= np.linalg.norm(c_sparse)
        candidates.append(c_sparse)
    
    # 7. Symmetric patterns exploiting m -> -m symmetry
    # For real harmonics, f is real so c_{-m} and c_m have specific relations
    # Try patterns where c_m = c_{-m} (even in m) or c_m = -c_{-m} (odd in m)
    for _ in range(20):
        c_sym = np.zeros(n_coeffs)
        # Generate for m = 0..10 and mirror
        for m in range(11):
            val = rng.normal()
            c_sym[m + 10] = val  # m
            c_sym[10 - m] = val  # -m (same value for even symmetry)
        c_sym /= np.linalg.norm(c_sym)
        candidates.append(c_sym)
    
    for _ in range(20):
        c_asym = np.zeros(n_coeffs)
        for m in range(11):
            val = rng.normal()
            c_asym[m + 10] = val
            c_asym[10 - m] = -val  # odd symmetry
        c_asym /= np.linalg.norm(c_asym)
        candidates.append(c_asym)
    
    # Sort candidates by their ratio and take top ones for optimization
    scored_candidates = []
    for c in candidates:
        r = _compute_ratio(c)
        if np.isfinite(r):
            scored_candidates.append((r, c))
    scored_candidates.sort(key=lambda x: x[0])
    
    # Update best from initial candidates
    if scored_candidates:
        _update_best(scored_candidates[0][1], scored_candidates[0][0])
    
    # Define objective with gradient for SLSQP
    def _obj_with_grad(c):
        R, grad = _compute_ratio_and_grad(c)
        if np.isinf(R):
            return 1e10, np.zeros_like(c)
        return R, grad
    
    # Equality constraint: ||c||^2 = 1
    def _norm_constraint(c):
        return np.dot(c, c) - 1.0
    
    def _norm_constraint_grad(c):
        return 2.0 * c
    
    constraints = [{'type': 'eq', 'fun': _norm_constraint, 'jac': _norm_constraint_grad}]
    
    # Phase 1: Optimize from top candidates using SLSQP with analytic gradient
    n_slsqp = 15
    for i, (r_init, c_init) in enumerate(scored_candidates[:n_slsqp]):
        if time.time() > deadline - 120:
            break
        
        try:
            res = minimize(
                _obj_with_grad,
                c_init,
                method='SLSQP',
                jac=True,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            c_opt = res.x
            c_opt = c_opt / np.linalg.norm(c_opt)
            r_opt = _compute_ratio(c_opt)
            if np.isfinite(r_opt):
                _update_best(c_opt, r_opt)
        except Exception:
            pass
    
    # Phase 2: Newton refinement on the sphere for the best solutions
    # Take top few solutions and apply Newton
    top_n = min(5, len(scored_candidates))
    
    # Re-evaluate best after SLSQP
    if best_coeffs is not None:
        c_best = best_coeffs / np.linalg.norm(best_coeffs)
        c_newton, r_newton = _newton_on_sphere(c_best, max_iter=50, tol=1e-14, deadline=deadline - 60)
        if np.isfinite(r_newton):
            _update_best(c_newton, r_newton)
    
    # Apply Newton to other good starting points
    for i in range(top_n):
        if time.time() > deadline - 60:
            break
        c_init = scored_candidates[i][1] / np.linalg.norm(scored_candidates[i][1])
        try:
            c_newton, r_newton = _newton_on_sphere(c_init, max_iter=30, tol=1e-14, deadline=deadline - 60)
            if np.isfinite(r_newton):
                _update_best(c_newton, r_newton)
        except Exception:
            pass
    
    # Phase 3: Additional random restarts with SLSQP + Newton
    n_extra = 20
    for i in range(n_extra):
        if time.time() > deadline - 40:
            break
        
        c_start = rng.normal(size=n_coeffs)
        norm_c = np.linalg.norm(c_start)
        if norm_c < 1e-15:
            continue
        c_start_norm = c_start / norm_c
        
        try:
            res = minimize(
                _obj_with_grad,
                c_start_norm,
                method='SLSQP',
                jac=True,
                constraints=constraints,
                options={'maxiter': 150, 'ftol': 1e-12}
            )
            c_opt = res.x / np.linalg.norm(res.x)
            r_opt = _compute_ratio(c_opt)
            if np.isfinite(r_opt):
                _update_best(c_opt, r_opt)
                
                # Newton polish if improved
                if r_opt < best_ratio + 1e-6 and time.time() < deadline - 30:
                    c_newton, r_newton = _newton_on_sphere(c_opt, max_iter=30, tol=1e-14, deadline=deadline - 30)
                    if np.isfinite(r_newton):
                        _update_best(c_newton, r_newton)
        except Exception:
            pass
    
    # Phase 4: Final high-precision Newton polish
    if best_coeffs is not None and time.time() < deadline - 10:
        c_best = best_coeffs / np.linalg.norm(best_coeffs)
        try:
            c_final, r_final = _newton_on_sphere(c_best, max_iter=50, tol=1e-15, deadline=deadline - 5)
            if np.isfinite(r_final):
                _update_best(c_final, r_final)
        except Exception:
            pass
        
        # One more SLSQP with very tight tolerance
        if time.time() < deadline - 5:
            try:
                res = minimize(
                    _obj_with_grad,
                    best_coeffs / np.linalg.norm(best_coeffs),
                    method='SLSQP',
                    jac=True,
                    constraints=constraints,
                    options={'maxiter': 300, 'ftol': 1e-15}
                )
                c_opt = res.x / np.linalg.norm(res.x)
                r_opt = _compute_ratio(c_opt)
                if np.isfinite(r_opt):
                    _update_best(c_opt, r_opt)
            except Exception:
                pass
    
    # Final validation
    if best_coeffs is None or np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        best_coeffs = np.ones(n_coeffs)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs == 0:
            best_coeffs = np.ones(n_coeffs)
        else:
            best_coeffs = best_coeffs / max_abs * 1.0
            if np.max(np.abs(best_coeffs)) < 1e-3:
                best_coeffs = np.ones(n_coeffs)
    
    return best_coeffs.astype(np.float64)


# EVOLVE-BLOCK-END
