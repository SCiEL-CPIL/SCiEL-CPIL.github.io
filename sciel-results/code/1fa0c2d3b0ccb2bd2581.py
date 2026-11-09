# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_VALS = np.arange(_N)
_YS = np.repeat(_VALS, _N)    # y for each cell
_XS = np.tile(_VALS, _N)      # x for each cell

# Precompute projection indices
_IDX_D = (_XS - _YS) + 9
_N_D = 19
_IDX_H0 = _XS
_N_H0 = 10
_IDX_H1 = _XS + _YS
_N_H1 = 19
_IDX_H2 = _XS + 2 * _YS
_N_H2 = 28

# Precompute entropy lookup tables for speed
def _precompute_entropy_table(n_bins):
    """Precompute entropy for a bin with count k out of total N cells."""
    table = np.zeros(n_bins + 1)
    for k in range(1, n_bins + 1):
        p = k / (n_bins * n_bins)  # This is not quite right for general use, but we'll compute dynamically
        # Actually, let's just compute entropy directly each time but with vectorization
    return table

def _entropy_from_counts(counts):
    """Compute Shannon entropy in bits from a count vector (summing to 1)."""
    p = counts
    s = p.sum()
    if s <= 1e-30:
        return 0.0
    q = p / s
    q = q[q > 1e-15]
    if len(q) == 0:
        return 0.0
    return float(-(q * np.log2(q)).sum())

def _compute_value(P):
    """Compute D/M ratio for a given distribution P (10x10)."""
    p = P.ravel()
    
    d_counts = np.zeros(_N_D)
    np.add.at(d_counts, _IDX_D, p)
    H_D = _entropy_from_counts(d_counts)
    
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
    """
    cy, cx, sy, sx, rho = theta
    sy = max(sy, 0.05)
    sx = max(sx, 0.05)
    rho = np.clip(rho, -0.999, 0.999)
    
    y, x = np.meshgrid(_VALS, _VALS, indexing='ij')
    dy = (y - cy) / sy
    dx = (x - cx) / sx
    z = (dy - rho * dx) / np.sqrt(1 - rho**2)
    cov = dx**2 + z**2
    P = np.exp(-0.5 * cov)
    
    s = P.sum()
    if s <= 0:
        return np.ones((_N, _N)) / (_N * _N)
    P /= s
    return P

def _gaussian_mixture(theta):
    """
    Build P from a Gaussian mixture model.
    theta = [cy1, cx1, sy1, sx1, rho1, w1, cy2, cx2, sy2, sx2, rho2, w2, ...]
    Each component has 6 parameters: cy, cx, sy, sx, rho, weight
    """
    n_components = len(theta) // 6
    P = np.zeros((_N, _N))
    y, x = np.meshgrid(_VALS, _VALS, indexing='ij')
    
    for i in range(n_components):
        params = theta[i*6:(i+1)*6]
        cy, cx, sy, sx, rho, w = params
        sy = max(sy, 0.05)
        sx = max(sx, 0.05)
        rho = np.clip(rho, -0.999, 0.999)
        
        dy = (y - cy) / sy
        dx = (x - cx) / sx
        z = (dy - rho * dx) / np.sqrt(1 - rho**2)
        cov = dx**2 + z**2
        P += w * np.exp(-0.5 * cov)
    
    s = P.sum()
    if s <= 0:
        return np.ones((_N, _N)) / (_N * _N)
    P /= s
    return P

def _optimize_single_gaussian(init, time_budget):
    """Optimize a single Gaussian component."""
    def neg_value(theta):
        P = _build_from_theta(theta)
        return -_compute_value(P)
    
    t_start = time.time()
    best_val = -neg_value(init)
    best_theta = init.copy()
    
    # Try multiple methods
    methods = ['Nelder-Mead', 'Powell']
    for method in methods:
        if time.time() - t_start > time_budget:
            break
        try:
            res = minimize(neg_value, best_theta, method=method, 
                          options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.fun < -best_val:
                best_val = -res.fun
                best_theta = res.x.copy()
        except:
            pass
    
    P = _build_from_theta(best_theta)
    return P, best_val

def _optimize_mixture(n_components, init, time_budget):
    """Optimize a Gaussian mixture model."""
    def neg_value(theta):
        P = _gaussian_mixture(theta)
        return -_compute_value(P)
    
    t_start = time.time()
    best_val = -neg_value(init)
    best_theta = init.copy()
    
    try:
        res = minimize(neg_value, best_theta, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.fun < -best_val:
            best_val = -res.fun
            best_theta = res.x.copy()
    except:
        pass
    
    P = _gaussian_mixture(best_theta)
    return P, best_val

def _local_search(P, time_budget, step_size=0.05, n_steps=1000):
    """Perform local search by perturbing P directly."""
    t_start = time.time()
    best_P = P.copy()
    best_val = _compute_value(P)
    
    for step in range(n_steps):
        if time.time() - t_start > time_budget:
            break
        
        # Random perturbation
        noise = np.random.randn(_N, _N) * step_size
        P_new = best_P + noise
        P_new = np.clip(P_new, 0, None)
        s = P_new.sum()
        if s > 1e-9:
            P_new /= s
        
        val_new = _compute_value(P_new)
        if val_new > best_val:
            best_val = val_new
            best_P = P_new.copy()
            # Decrease step size
            step_size *= 0.95
        else:
            # Occasionally increase step size to escape local minima
            if np.random.rand() < 0.1:
                step_size = min(step_size * 1.5, 0.5)
    
    return best_P, best_val

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_val = _compute_value(P)
    best_P = P.copy()
    
    t_start = time.time()
    time_limit = 620.0  # Safety margin
    
    # Phase 1: Optimize single Gaussians from multiple initial points
    inits = []
    # Center
    inits.append([4.5, 4.5, 2.0, 2.0, 0.0])
    # Diagonals
    for c in range(-4, 5):
        y_min = max(0, -c)
        y_max = min(9, 9 - c)
        if y_min <= y_max:
            cy = (y_min + y_max) / 2.0
            cx = (y_min + c + y_max + c) / 2.0
            inits.append([cy, cx, 1.5, 1.5, 0.0])
    # Wide Gaussians
    inits.append([4.5, 4.5, 5.0, 5.0, 0.0])
    inits.append([4.5, 4.5, 3.0, 3.0, 0.5])
    inits.append([4.5, 4.5, 3.0, 3.0, -0.5])
    inits.append([4.5, 4.5, 1.0, 3.0, 0.0])
    inits.append([4.5, 4.5, 3.0, 1.0, 0.0])
    
    for init in inits:
        if time.time() - t_start > time_limit - 30:
            break
        time_budget = min(2.0, time_limit - (time.time() - t_start) - 10)
        if time_budget < 0.5:
            break
        try:
            P_opt, val_opt = _optimize_single_gaussian(init, time_budget)
            if val_opt > best_val:
                best_val = val_opt
                best_P = P_opt
        except:
            pass
    
    # Phase 2: Try Gaussian mixtures
    if time.time() - t_start < time_limit - 20:
        # 2-component mixture
        init_mix = [4.0, 4.0, 1.5, 1.5, 0.0, 0.5,
                    6.0, 6.0, 1.5, 1.5, 0.0, 0.5]
        time_budget = min(10.0, time_limit - (time.time() - t_start) - 10)
        if time_budget > 2:
            try:
                P_opt, val_opt = _optimize_mixture(2, init_mix, time_budget)
                if val_opt > best_val:
                    best_val = val_opt
                    best_P = P_opt
            except:
                pass
        
        # 3-component mixture
        if time.time() - t_start < time_limit - 15:
            init_mix3 = [3.0, 3.0, 1.2, 1.2, 0.0, 0.34,
                         6.0, 6.0, 1.2, 1.2, 0.0, 0.33,
                         4.5, 4.5, 1.5, 1.5, 0.5, 0.33]
            time_budget = min(15.0, time_limit - (time.time() - t_start) - 10)
            if time_budget > 2:
                try:
                    P_opt, val_opt = _optimize_mixture(3, init_mix3, time_budget)
                    if val_opt > best_val:
                        best_val = val_opt
                        best_P = P_opt
                except:
                    pass
    
    # Phase 3: Local search on the best P found
    if time.time() - t_start < time_limit - 5:
        time_budget = time_limit - (time.time() - t_start) - 3
        if time_budget > 1:
            try:
                P_opt, val_opt = _local_search(best_P, time_budget)
                if val_opt > best_val:
                    best_val = val_opt
                    best_P = P_opt
            except:
                pass
    
    # Final check and normalization
    P = np.clip(best_P, 0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P /= s
        
    return P

# EVOLVE-BLOCK-END
