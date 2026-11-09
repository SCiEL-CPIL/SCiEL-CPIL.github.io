# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 10
_VALS = np.arange(_N)
# Flatten indices for x, y
_XS = np.tile(_VALS, _N)      # x for each cell (row-major: y varies slowest? Actually P[y,x])
# Let's be careful: P[y, x]. ravel() in C order: P[0,0], P[0,1], ..., P[0,9], P[1,0]...
# So index i corresponds to y = i // N, x = i % N.
_YS = np.repeat(_VALS, _N)    # y for each cell
_XS = np.tile(_VALS, _N)      # x for each cell

# Precompute projection indices
# D: x - y. Range: -9 to 9. Offset by 9 -> 0 to 18.
_IDX_D = (_XS - _YS) + 9
_N_D = 19

# H_0: x. Range 0..9.
_IDX_H0 = _XS
_N_H0 = 10

# H_1: x + y. Range 0..18.
_IDX_H1 = _XS + _YS
_N_H1 = 19

# H_2: x + 2y. Range 0..27.
_IDX_H2 = _XS + 2 * _YS
_N_H2 = 28


def _entropy_from_counts(counts):
    """Compute Shannon entropy in bits from a count vector, assuming it sums to 1."""
    # counts is already normalized to sum 1? Or raw probabilities?
    # Let's assume input is probability vector p.
    p = counts
    s = p.sum()
    if s <= 1e-30:
        return 0.0
    q = p / s
    q = q[q > 0.0]
    if len(q) == 0:
        return 0.0
    return float(-(q * np.log2(q)).sum())


def _compute_value(P):
    """Compute D/M ratio for a given distribution P (10x10)."""
    p = P.ravel()
    
    # D: x - y
    d_counts = np.zeros(_N_D)
    np.add.at(d_counts, _IDX_D, p)
    H_D = _entropy_from_counts(d_counts)
    
    # M: max(H_0, H_1, H_2)
    h0_counts = np.zeros(_N_H0)
    np.add.at(h0_counts, _IDX_H0, p)
    H_0 = _entropy_from_counts(h0_counts)
    
    h1_counts = np.zeros(_N_H1)
    np.add.at(h1_counts, _IDX_H1, p)
    H_1 = _entropy_from_counts(h1_counts)
    
    h2_counts = np.zeros(_N_H2)
    np.add.at(h2_counts, _IDX_H2, p)
    H_2 = _entropy_from_counts(h2_counts)
    
    M = max(H_0, H_1, H_2)
    if M <= 1e-9:
        return 0.0
        
    return H_D / M


def _build_from_theta(theta):
    """
    Build P from theta parameters.
    theta[0] = center_y, theta[1] = center_x, theta[2] = sigma_y, theta[3] = sigma_x, theta[4] = rho
    This is a 2D Gaussian on the grid.
    """
    cy, cx, sy, sx, rho = theta
    # Ensure sigmas are positive
    sy = max(sy, 0.1)
    sx = max(sx, 0.1)
    rho = np.clip(rho, -0.999, 0.999)
    
    y, x = np.meshgrid(_VALS, _VALS, indexing='ij')
    dy = (y - cy) / sy
    dx = (x - cx) / sx
    z = (dy - rho * dx) / np.sqrt(1 - rho**2)
    cov = dx**2 + z**2
    P = np.exp(-0.5 * cov)
    
    # Normalize
    s = P.sum()
    if s <= 0:
        return np.ones((_N, _N)) / (_N * _N)
    P /= s
    return P


def _gradient_ascent(P, step_size, n_steps, time_limit):
    """
    Perform gradient ascent on the value D/M by perturbing P directly?
    No, better to optimize the parameters of a parametric family or use scipy.
    However, scipy is allowed. Let's use scipy.optimize.minimize on the log-probabilities
    or the parameters of a Gaussian mixture?
    
    The knowledge suggests: "Adding diagonal-line starts... plus more Gaussian centers/scales and extra polish rounds tighten the optimum".
    
    Let's use scipy.optimize.minimize on the 5 parameters of a single Gaussian,
    starting from multiple initial points (diagonals and grid centers).
    """
    from scipy.optimize import minimize
    
    def neg_value(theta):
        P = _build_from_theta(theta)
        return -_compute_value(P)
    
    best_P = P.copy()
    best_val = _compute_value(P)
    
    # Initial guesses:
    # 1. Center of grid
    # 2. Diagonals x=y (slope 1) -> maximize H1? No, we want to maximize D (x-y).
    #    If we concentrate on x-y=0 (main diagonal), D is low (entropy 0).
    #    To maximize D, we want mass spread over many x-y values.
    #    Uniform is good for D.
    #    To minimize M, we want mass concentrated in few x, x+y, x+2y values.
    #    This is a trade-off.
    
    # Strategy: Start with various Gaussians.
    inits = []
    # Center
    inits.append([4.5, 4.5, 2.0, 2.0, 0.0])
    # Diagonals
    for c in range(-4, 5):
        # x - y = c -> x = y + c
        # Center of this line segment within grid
        y_min = max(0, -c)
        y_max = min(9, 9 - c)
        if y_min <= y_max:
            cy = (y_min + y_max) / 2.0
            cx = (y_min + c + y_max + c) / 2.0
            inits.append([cy, cx, 1.0, 1.0, 0.0])
            
    # Also try wide Gaussians (closer to uniform)
    inits.append([4.5, 4.5, 5.0, 5.0, 0.0])
    inits.append([4.5, 4.5, 3.0, 3.0, 0.5])
    inits.append([4.5, 4.5, 3.0, 3.0, -0.5])
    
    t_start = time.time()
    
    for init in inits:
        if time.time() - t_start > time_limit - 10:
            break
            
        res = minimize(neg_value, init, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res.success or res.fun < -best_val:
            P_new = _build_from_theta(res.x)
            val_new = _compute_value(P_new)
            if val_new > best_val:
                best_val = val_new
                best_P = P_new
                
    return best_P, best_val


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    # Start with uniform
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_val = _compute_value(P)
    
    # Deadline guard
    t_start = time.time()
    time_limit = 630.0
    
    # Run optimization
    try:
        P_opt, val_opt = _gradient_ascent(P, 0.1, 100, time_limit - 10)
        if val_opt > best_val:
            P = P_opt
            best_val = val_opt
    except Exception:
        pass
        
    # Final check and normalization
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P /= s
        
    return P

# EVOLVE-BLOCK-END
