import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

START_TIME = time.time()
TIME_LIMIT = 205.0

# Hermite polynomial coefficients in power basis [x^0, x^2, x^4, x^6, x^8, x^10, x^12]
_H0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_H4 = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
_H8 = np.array([105.0, -420.0, 210.0, 0.0, 1.0, 0.0, 0.0])
_H12 = np.array([945.0, -226800.0, 151200.0, -30240.0, 2520.0, -84.0, 1.0])

# Precompute integral table: I[n] = integral x^{2n} exp(-x^2) dx from -inf to inf
_I_table = np.zeros(30)
_I_table[0] = np.sqrt(np.pi)
for _n in range(1, 30):
    _I_table[_n] = _I_table[_n-1] * (2*_n - 1) / 2.0

# Precompute dP/dc vectors (constant)
_dP_dc0 = _H0 - (1.0/945.0)*_H12
_dP_dc4 = _H4 - (3.0/945.0)*_H12
_dP_dc8 = _H8 - (105.0/945.0)*_H12
_dP_dc = np.stack([_dP_dc0, _dP_dc4, _dP_dc8])  # shape (3, 7)

# Precompute R derivative structure: dR[i]/dP[j]
_dR_dP = np.zeros((7, 7))
for i in range(6):
    _dR_dP[i, i] = -1.0
    _dR_dP[i, i+1] = 2.0*(i+1)
_dR_dP[6, 6] = -1.0

# Precompute convolution matrices for efficient computation
_M_norm = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        _M_norm[i, j] = _I_table[i+j]

_M_xnorm = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        _M_xnorm[i, j] = _I_table[i+j+1]

_M_deriv = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        _M_deriv[i, j] = np.dot(_dR_dP[:, j], _I_table[i:i+7])

def _get_P(c0, c4, c8):
    """Get P coefficients given c0, c4, c8 with c12 derived from P(0)=0."""
    c12 = -(c0 + 3.0*c4 + 105.0*c8) / 945.0
    P = c0*_H0 + c4*_H4 + c8*_H8 + c12*_H12
    return P

def _compute_bound(c0, c4, c8):
    """Compute the uncertainty bound sqrt(||xg||^2 * ||g'||^2 / ||g||^4)."""
    P = _get_P(c0, c4, c8)
    
    norm_sq = np.dot(P, _M_norm @ P)
    x_norm_sq = np.dot(P, _M_xnorm @ P)
    
    R = np.zeros(7)
    for k in range(6):
        R[k] = 2*(k+1)*P[k+1] - P[k]
    R[6] = -P[6]
    
    deriv_norm_sq = np.dot(R, _M_norm @ R)
    
    if norm_sq <= 0:
        return 0.0
    
    bound = np.sqrt(x_norm_sq * deriv_norm_sq) / norm_sq
    return bound

def _compute_bound_grad(c0, c4, c8):
    """Compute bound and its gradient using vectorized operations."""
    P = _get_P(c0, c4, c8)
    
    norm_sq = np.dot(P, _M_norm @ P)
    x_norm_sq = np.dot(P, _M_xnorm @ P)
    
    R = np.zeros(7)
    for k in range(6):
        R[k] = 2*(k+1)*P[k+1] - P[k]
    R[6] = -P[6]
    
    deriv_norm_sq = np.dot(R, _M_norm @ R)
    
    if norm_sq <= 0:
        return 0.0, np.zeros(3)
    
    bound = np.sqrt(x_norm_sq * deriv_norm_sq) / norm_sq
    if bound <= 0:
        return bound, np.zeros(3)
    
    d_norm_dP = 2.0 * (_M_norm @ P)
    d_xnorm_dP = 2.0 * (_M_xnorm @ P)
    d_deriv_dP = 2.0 * (_M_deriv.T @ R)
    
    A = x_norm_sq
    B = deriv_norm_sq
    C = norm_sq
    
    dA_dc = _dP_dc @ d_xnorm_dP
    dB_dc = _dP_dc @ d_deriv_dP
    dC_dc = _dP_dc @ d_norm_dP
    
    bound_sq = A * B / (C * C)
    if bound_sq <= 0:
        return bound, np.zeros(3)
    
    factor = 0.5 / bound
    AB_over_C2 = A * B / (C * C)
    d_bound_sq_dc = (B*dA_dc + A*dB_dc - 2.0*AB_over_C2*dC_dc) / C
    
    grad = factor * d_bound_sq_dc
    
    return bound, grad

def _objective(coeffs):
    return _compute_bound(coeffs[0], coeffs[1], coeffs[2])

def _objective_grad(coeffs):
    return _compute_bound_grad(coeffs[0], coeffs[1], coeffs[2])

def _spherical_search(c_init, max_iters=500, initial_step=0.1):
    """Stochastic local search on the unit sphere for the coefficient vector."""
    # Normalize the initial point
    c = c_init.copy()
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-15:
        c = np.array([1.0, 0.0, 0.0])
    else:
        c = c / norm_c
    
    best_c = c.copy()
    best_score = _objective(best_c)
    step_size = initial_step
    
    for i in range(max_iters):
        if time.time() > TIME_LIMIT - 10:
            break
        
        # Generate a random direction on the sphere
        if np.random.random() < 0.7:
            # Small perturbation
            perturbation = np.random.normal(0, step_size, size=3)
        else:
            # Larger perturbation occasionally
            perturbation = np.random.normal(0, step_size * 3, size=3)
        
        new_c = c + perturbation
        new_norm = np.linalg.norm(new_c)
        if new_norm < 1e-15:
            continue
        new_c = new_c / new_norm
        
        new_score = _objective(new_c)
        
        if new_score < best_score:
            best_score = new_score
            best_c = new_c.copy()
            c = new_c
            # Adapt step size: increase slightly on success
            step_size = min(step_size * 1.05, 0.5)
        else:
            # Adapt step size: decrease on failure
            step_size = max(step_size * 0.95, 0.001)
    
    return best_c, best_score

def construct():
    seed = np.array([-1.13, 0.04, 1.0/3240.0])
    best_score = _objective(seed)
    best_coeffs = seed.copy()
    
    np.random.seed(42)
    bounds = [(-10.0, 10.0), (-5.0, 5.0), (-2.0, 2.0)]
    
    # Phase 1: Multi-start L-BFGS-B with diverse initializations
    num_starts = 100
    for i in range(num_starts):
        if time.time() > TIME_LIMIT - 30:
            break
        
        if i == 0:
            x0 = seed.copy()
        elif i == 1:
            x0 = np.array([0.0, 0.0, 0.0])
        elif i == 2:
            x0 = np.array([1.0, -0.1, 0.001])
        elif i == 3:
            x0 = np.array([-1.0, 0.1, -0.001])
        elif i == 4:
            x0 = np.array([0.5, 0.0, 0.0])
        elif i == 5:
            x0 = np.array([-0.5, 0.0, 0.0])
        elif i == 6:
            x0 = np.array([0.0, 0.5, 0.0])
        elif i == 7:
            x0 = np.array([0.0, -0.5, 0.0])
        elif i == 8:
            x0 = np.array([2.0, -0.2, 0.002])
        elif i == 9:
            x0 = np.array([-2.0, 0.2, -0.002])
        elif i == 10:
            x0 = np.array([3.0, -0.3, 0.003])
        elif i == 11:
            x0 = np.array([-3.0, 0.3, -0.003])
        else:
            if i % 3 == 0:
                x0 = np.array([
                    np.random.uniform(bounds[0][0], bounds[0][1]),
                    np.random.uniform(bounds[1][0], bounds[1][1]),
                    np.random.uniform(bounds[2][0], bounds[2][1])
                ])
            elif i % 3 == 1:
                x0 = seed + np.random.normal(0, 0.5, size=3)
                x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
            else:
                x0 = seed + np.random.normal(0, 1.0, size=3)
                x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        try:
            res = minimize(_objective_grad, x0, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Spherical search around the best solution
    if time.time() < TIME_LIMIT - 25:
        try:
            # Normalize the best solution for spherical search
            c_norm = best_coeffs / (np.linalg.norm(best_coeffs) + 1e-15)
            c_sph, score_sph = _spherical_search(c_norm, max_iters=200, initial_step=0.05)
            
            # Map back to original scale
            scale = np.linalg.norm(best_coeffs)
            candidate = c_sph * scale
            candidate_score = _objective(candidate)
            
            if candidate_score < best_score:
                best_score = candidate_score
                best_coeffs = candidate
        except Exception:
            pass
    
    # Phase 3: Local refinement with L-BFGS-B
    for i in range(30):
        if time.time() > TIME_LIMIT - 15:
            break
        
        sigma = 0.1 * (1.0 - i/30.0)**2 + 0.0005
        x0 = best_coeffs + np.random.normal(0, sigma, size=3)
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        try:
            res = minimize(_objective_grad, x0, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: 1D line searches along each coordinate
    if time.time() < TIME_LIMIT - 10:
        for coord in range(3):
            if time.time() > TIME_LIMIT - 5:
                break
            def line_obj(t, coord=coord):
                x = best_coeffs.copy()
                x[coord] = t
                return _objective(x)
            
            try:
                lo = max(bounds[coord][0], best_coeffs[coord] - 1.0)
                hi = min(bounds[coord][1], best_coeffs[coord] + 1.0)
                res = minimize_scalar(line_obj, bounds=(lo, hi),
                                      method='bounded', options={'xatol': 1e-14})
                if res.fun < best_score:
                    new_x = best_coeffs.copy()
                    new_x[coord] = res.x
                    best_coeffs = new_x
                    best_score = _objective(best_coeffs)
            except Exception:
                pass
    
    # Phase 5: Final high-precision L-BFGS-B
    if time.time() < TIME_LIMIT - 5:
        try:
            res = minimize(_objective_grad, best_coeffs, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 6: Final line search polish
    if time.time() < TIME_LIMIT - 2:
        for coord in range(3):
            if time.time() > TIME_LIMIT - 1:
                break
            def line_obj2(t, coord=coord):
                x = best_coeffs.copy()
                x[coord] = t
                return _objective(x)
            
            try:
                lo = max(bounds[coord][0], best_coeffs[coord] - 0.1)
                hi = min(bounds[coord][1], best_coeffs[coord] + 0.1)
                res = minimize_scalar(line_obj2, bounds=(lo, hi),
                                      method='bounded', options={'xatol': 1e-15})
                if res.fun < best_score:
                    new_x = best_coeffs.copy()
                    new_x[coord] = res.x
                    best_coeffs = new_x
                    best_score = _objective(best_coeffs)
            except Exception:
                pass
    
    return best_coeffs
