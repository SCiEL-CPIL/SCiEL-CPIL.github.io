# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_VALS = np.arange(_N)
# P[y, x]. ravel() in C order: index i -> y = i // N, x = i % N
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


def _shannon_entropy(p):
    """Compute Shannon entropy in bits from a probability vector."""
    p = p[p > 0.0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _compute_D_M(P):
    """Compute D and M for a given distribution P (10x10)."""
    p = P.ravel()
    s = p.sum()
    if s <= 1e-30:
        return 0.0, 0.0
    
    # D: x - y
    d_counts = np.zeros(_N_D)
    np.add.at(d_counts, _IDX_D, p)
    d_counts /= s
    H_D = _shannon_entropy(d_counts)
    
    # M: max(H_0, H_1, H_2)
    h0_counts = np.zeros(_N_H0)
    np.add.at(h0_counts, _IDX_H0, p)
    h0_counts /= s
    H_0 = _shannon_entropy(h0_counts)
    
    h1_counts = np.zeros(_N_H1)
    np.add.at(h1_counts, _IDX_H1, p)
    h1_counts /= s
    H_1 = _shannon_entropy(h1_counts)
    
    h2_counts = np.zeros(_N_H2)
    np.add.at(h2_counts, _IDX_H2, p)
    h2_counts /= s
    H_2 = _shannon_entropy(h2_counts)
    
    M = max(H_0, H_1, H_2)
    
    return H_D, M


def _ratio_from_P(P):
    """Compute D/M ratio for a given distribution P (10x10)."""
    D, M = _compute_D_M(P)
    if M <= 1e-9:
        return 0.0
    return D / M


def _soft_max_entropy(H_vals, temperature=0.05):
    """
    Soft-max surrogate for max(H_vals) to make the objective differentiable.
    As temperature -> 0, this approaches max(H_vals).
    """
    H_vals = np.asarray(H_vals)
    if len(H_vals) == 0:
        return 0.0
    max_h = np.max(H_vals)
    if temperature <= 1e-10:
        return max_h
    exp_vals = np.exp(H_vals / temperature - max_h / temperature)
    return max_h + temperature * np.log(np.sum(exp_vals))


def _objective_smooth(theta, temperature=0.05):
    """
    Objective function for optimization: -log(D) + log(M_smooth)
    Minimizing this is equivalent to maximizing D / M_smooth.
    theta: [center_y, center_x, sigma_y, sigma_x, rho]
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
    
    D, M = _compute_D_M(P)
    if D <= 1e-10 or M <= 1e-10:
        return 1e10
    
    M_smooth = _soft_max_entropy([H_0, H_1, H_2] if False else _get_individual_entropies(P), temperature)
    
    if D <= 1e-10 or M_smooth <= 1e-10:
        return 1e10
    
    return -np.log(D) + np.log(M_smooth)


def _get_individual_entropies(P):
    """Get individual entropies H_0, H_1, H_2 for soft-max."""
    p = P.ravel()
    s = p.sum()
    if s <= 1e-30:
        return [0.0, 0.0, 0.0]
    
    h0_counts = np.zeros(_N_H0)
    np.add.at(h0_counts, _IDX_H0, p)
    h0_counts /= s
    H_0 = _shannon_entropy(h0_counts)
    
    h1_counts = np.zeros(_N_H1)
    np.add.at(h1_counts, _IDX_H1, p)
    h1_counts /= s
    H_1 = _shannon_entropy(h1_counts)
    
    h2_counts = np.zeros(_N_H2)
    np.add.at(h2_counts, _IDX_H2, p)
    h2_counts /= s
    H_2 = _shannon_entropy(h2_counts)
    
    return [H_0, H_1, H_2]


def _objective(theta, temperature=0.05):
    """
    Objective function for optimization: -D/M
    theta: [center_y, center_x, sigma_y, sigma_x, rho]
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
    
    val = _ratio_from_P(P)
    return -val


def _objective_and_P(theta, temperature=0.05):
    """
    Returns (objective_value, P) for a given theta.
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
        P = np.ones((_N, _N)) / (_N * _N)
    else:
        P /= s
    
    val = _ratio_from_P(P)
    return -val, P


def _greedy_sparse_search(P_init, time_limit, n_iterations=500):
    """
    Greedy sparse search: iteratively adjust individual cell weights
    to improve the ratio. This can find non-Gaussian optima.
    """
    t_start = time.time()
    P = P_init.copy()
    best_val = _ratio_from_P(P)
    best_P = P.copy()
    
    # Precompute projection matrices for fast entropy updates
    # For each cell i, we know which bin it contributes to in D, H0, H1, H2
    
    for _ in range(n_iterations):
        if time.time() - t_start > time_limit - 5:
            break
        
        best_delta = 0
        best_cell = -1
        best_scale = 1.0
        
        # Try scaling each cell by a factor
        scales = [0.5, 0.75, 1.5, 2.0]
        
        for cell in range(_N * _N):
            if time.time() - t_start > time_limit - 5:
                break
            
            for scale in scales:
                P_test = P.copy()
                P_test.ravel()[cell] *= scale
                
                # Renormalize
                s = P_test.sum()
                if s <= 1e-10:
                    continue
                P_test /= s
                
                val = _ratio_from_P(P_test)
                delta = val - best_val
                
                if delta > best_delta:
                    best_delta = delta
                    best_cell = cell
                    best_scale = scale
        
        if best_cell >= 0 and best_delta > 1e-12:
            P.ravel()[best_cell] *= best_scale
            s = P.sum()
            if s > 1e-10:
                P /= s
            best_val = _ratio_from_P(P)
            best_P = P.copy()
        else:
            break
    
    return best_P, best_val


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    # Start with uniform
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_val = _ratio_from_P(P)
    best_P = P.copy()
    
    # Deadline guard
    t_start = time.time()
    time_limit = 630.0
    
    # Generate diverse initializations
    inits = []
    
    # 1. Center of grid with various scales
    for sy in [1.0, 2.0, 3.0, 5.0]:
        inits.append([4.5, 4.5, sy, sy, 0.0])
    
    # 2. Diagonals x = y + c (maximize spread in x-y)
    for c in range(-4, 5):
        y_min = max(0, -c)
        y_max = min(9, 9 - c)
        if y_min <= y_max:
            cy = (y_min + y_max) / 2.0
            cx = (y_min + c + y_max + c) / 2.0
            inits.append([cy, cx, 1.5, 1.5, 0.0])
            inits.append([cy, cx, 2.0, 2.0, 0.0])
    
    # 3. Anti-diagonals x + y = c (minimize H1)
    for c in range(0, 19):
        y_min = max(0, c - 9)
        y_max = min(9, c)
        if y_min <= y_max:
            cy = (y_min + y_max) / 2.0
            cx = (c - y_min + c - y_max) / 2.0
            inits.append([cy, cx, 1.5, 1.5, 0.0])
    
    # 4. Corners
    for cy, cx in [(0, 0), (0, 9), (9, 0), (9, 9)]:
        inits.append([cy, cx, 2.0, 2.0, 0.0])
        inits.append([cy, cx, 3.0, 3.0, 0.0])
    
    # 5. Edges
    for cy in [0, 9]:
        inits.append([cy, 4.5, 1.5, 2.0, 0.0])
    for cx in [0, 9]:
        inits.append([4.5, cx, 2.0, 1.5, 0.0])
    
    # 6. With correlation
    for rho in [-0.5, 0.5]:
        inits.append([4.5, 4.5, 2.0, 2.0, rho])
        inits.append([4.5, 4.5, 3.0, 3.0, rho])
    
    # Run optimization with smooth objective first, then polish with raw ratio
    for temp in [0.1, 0.05, 0.01, 0.001]:
        for init in inits:
            if time.time() - t_start > time_limit - 30:
                break
            
            try:
                # Use the smooth objective for optimization
                res = minimize(
                    lambda t: _objective_smooth(t, temperature=temp),
                    init,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                
                # Evaluate with raw ratio
                val_new, P_new = _objective_and_P(res.x)
                val_new = -val_new  # Convert back to positive ratio
                
                if val_new > best_val:
                    best_val = val_new
                    best_P = P_new
                    
            except Exception:
                continue
    
    # Final polish with greedy sparse search
    if time.time() - t_start < time_limit - 60:
        try:
            P_polished, val_polished = _greedy_sparse_search(
                best_P,
                time_limit - (time.time() - t_start) - 10,
                n_iterations=200
            )
            if val_polished > best_val:
                best_val = val_polished
                best_P = P_polished
        except Exception:
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
