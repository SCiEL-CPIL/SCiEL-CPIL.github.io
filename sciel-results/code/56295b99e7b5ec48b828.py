import numpy as np
import time
import random
from scipy.optimize import minimize

N = 10
X_GRID, Y_GRID = np.meshgrid(np.arange(N), np.arange(N), indexing='xy')
X_FLAT = X_GRID.flatten()
Y_FLAT = Y_GRID.flatten()

# Pre-compute projection values
U_VALS = X_FLAT - Y_GRID.flatten()       # x - y
V_VALS = X_FLAT.flatten()                # x
W_VALS = X_FLAT.flatten() + Y_FLAT.flatten()       # x + y
Z_VALS = X_FLAT.flatten() + 2 * Y_FLAT.flatten()   # x + 2y

# Unique values and inverse indices for bincount
U_UNIQ, U_INV = np.unique(U_VALS, return_inverse=True)
V_UNIQ, V_INV = np.unique(V_VALS, return_inverse=True)
W_UNIQ, W_INV = np.unique(W_VALS, return_inverse=True)
Z_UNIQ, Z_INV = np.unique(Z_VALS, return_inverse=True)

def _entropy_from_probs(probs):
    """Calculate Shannon entropy in bits, ignoring zeros."""
    mask = probs > 1e-12
    if not np.any(mask):
        return 0.0
    p = probs[mask]
    p_sum = np.sum(p)
    if p_sum <= 1e-12:
        return 0.0
    p = p / p_sum
    return -np.sum(p * np.log2(p))

def _compute_ratio(P_flat):
    """
    Compute the ratio D / M for a given flat probability vector.
    P_flat: 1D array of length 100, non-negative, summing to 1.
    """
    P_flat = np.clip(P_flat, 0, None)
    total = np.sum(P_flat)
    if total <= 1e-9:
        return 0.0
    P_flat = P_flat / total
    
    p_u = np.bincount(U_INV, weights=P_flat, minlength=len(U_UNIQ))
    p_v = np.bincount(V_INV, weights=P_flat, minlength=len(V_UNIQ))
    p_w = np.bincount(W_INV, weights=P_flat, minlength=len(W_UNIQ))
    p_z = np.bincount(Z_INV, weights=P_flat, minlength=len(Z_UNIQ))
    
    h_u = _entropy_from_probs(p_u)
    h_v = _entropy_from_probs(p_v)
    h_w = _entropy_from_probs(p_w)
    h_z = _entropy_from_probs(p_z)
    
    D = h_u
    M = max(h_v, h_w, h_z)
    
    if M <= 1e-9:
        return 0.0
        
    return D / M

def _objective(params):
    """
    Objective function for minimization.
    We want to maximize ratio, so we minimize -ratio.
    params are raw values (not necessarily normalized).
    """
    P = np.clip(params, 0, None)
    total = np.sum(P)
    if total < 1e-9:
        return 0.0
    P = P / total
    ratio = _compute_ratio(P)
    return -ratio

def _generate_linear_ridge(rng):
    """
    Generate a probability distribution concentrated on a linear ridge.
    This exploits the anisotropic nature of the problem.
    We want to maximize H(x-y) while keeping H(x), H(x+y), H(x+2y) low.
    Lines with slope 0.5 (y = 0.5x + c) keep x-2y constant.
    Wait, the knowledge says: "Anisotropic Gaussian with negative correlation and tuned variance ratio (sigma_x approx 2*sigma_y) minimizes H(X+2Y) while maximizing H(X-Y)".
    This suggests aligning mass along lines where x-2y varies slowly? No, if mass is on a line x-2y = const, H(x-2y) is 0.
    We want H(x-y) high. So we want mass spread out in x-y.
    We want H(x+2y) low. So we want mass concentrated in x+2y.
    
    Let's look at the lines:
    L1: x - y = c  (Slope 1)
    L2: x + 2y = c (Slope -0.5)
    
    If we pick a line with slope -0.5, x+2y is constant -> H(x+2y)=0.
    But then x-y varies along the line.
    Let's test a distribution concentrated on a line with slope -0.5.
    Equation: x + 2y = c.
    Points: (0, c/2), (2, (c-2)/2), ...
    If we pick c such that the line covers many points, H(x-y) will be high.
    
    Let's generate a ridge along x + 2y = c.
    """
    P = np.zeros(100)
    # Pick a random intercept c
    # x + 2y ranges from 0 to 9 + 18 = 27.
    c = rng.uniform(5, 20)
    
    for x in range(10):
        for y in range(10):
            val = x + 2 * y
            # Gaussian around c
            # Sigma controls width. Small sigma = sharp ridge.
            sigma = rng.uniform(0.5, 2.0)
            weight = np.exp(-((val - c)**2) / (2 * sigma**2))
            P[x + 10*y] = weight
            
    # Add some noise to avoid degeneracy
    P += rng.random(100) * 0.01
    
    return P

def _generate_uniform(rng):
    return np.ones(100) * rng.uniform(0.1, 1.0)

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin for 600s limit
    
    best_P = np.ones(100) / 100.0
    best_ratio = _compute_ratio(best_P)
    
    rng = np.random.default_rng(42)
    
    # Generate initial population of diverse candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(_generate_uniform(rng))
    
    # 2. Several linear ridges
    for _ in range(20):
        candidates.append(_generate_linear_ridge(rng))
        
    # 3. Random sparse
    for _ in range(10):
        P = np.zeros(100)
        idx = rng.choice(100, size=10, replace=False)
        P[idx] = 1.0
        candidates.append(P)
        
    # 4. Anisotropic Gaussians (as per knowledge)
    for _ in range(10):
        P = np.zeros(100)
        cx = rng.uniform(0, 9)
        cy = rng.uniform(0, 9)
        # Sigma_x ~ 2 * Sigma_y
        sy = rng.uniform(0.5, 2.0)
        sx = sy * 2.0
        # Negative correlation rho
        rho = rng.uniform(-0.8, -0.2)
        
        for x in range(10):
            for y in range(10):
                dx = x - cx
                dy = y - cy
                # Mahalanobis distance squared
                # (dx, dy) * Inv(Cov) * (dx, dy)^T
                # Cov = [[sx^2, rho*sx*sy], [rho*sx*sy, sy^2]]
                det = sx**2 * sy**2 * (1 - rho**2)
                if det < 1e-9:
                    continue
                inv_c00 = sy**2 / det
                inv_c11 = sx**2 / det
                inv_c01 = -rho * sx * sy / det
                
                d2 = inv_c00 * dx**2 + 2 * inv_c01 * dx * dy + inv_c11 * dy**2
                P[x + 10*y] = np.exp(-0.5 * d2)
                
        P += 1e-6
        candidates.append(P)
        
    # Local optimization using Nelder-Mead
    # We will run Nelder-Mead on the raw vector.
    # To handle the simplex constraint, we rely on the fact that Nelder-Mead
    # might go out of bounds, so we clip in the objective.
    # However, Nelder-Mead on 100D is slow and can get stuck.
    # But with good initial points (ridges), it might refine them well.
    
    # We will use a subset of the best initial candidates to save time.
    # First, evaluate all candidates.
    scored_candidates = []
    for P in candidates:
        P = np.clip(P, 0, None)
        total = np.sum(P)
        if total < 1e-9: continue
        P = P / total
        r = _compute_ratio(P)
        scored_candidates.append((r, P))
        
    # Sort by score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_ratio = scored_candidates[0][0]
    best_P = scored_candidates[0][1]
    
    # Refine the top K candidates using Nelder-Mead
    # Time budget per refinement: ~5s
    # 20 candidates * 5s = 100s.
    # We have ~590s. We can do more.
    
    top_k = 50 # We only have ~40 candidates, so refine all
    for i, (r, P) in enumerate(scored_candidates[:top_k]):
        if time.time() > deadline - 10:
            break
            
        # Nelder-Mead options
        options = {
            'maxiter': 200,
            'maxfev': 500,
            'xatol': 1e-4,
            'fatol': 1e-5
        }
        
        try:
            res = minimize(_objective, P, method='Nelder-Mead', options=options)
            P_opt = np.clip(res.x, 0, None)
            total = np.sum(P_opt)
            if total < 1e-9: continue
            P_opt = P_opt / total
            r_opt = _compute_ratio(P_opt)
            
            if r_opt > best_ratio:
                best_ratio = r_opt
                best_P = P_opt
                
        except Exception:
            pass
            
    # If time remains, try a few more random restarts of the ridge generation and refinement
    while time.time() < deadline - 30:
        P_init = _generate_linear_ridge(rng)
        P_init = np.clip(P_init, 0, None)
        total = np.sum(P_init)
        if total < 1e-9: continue
        P_init = P_init / total
        
        try:
            res = minimize(_objective, P_init, method='Nelder-Mead', 
                           options={'maxiter': 100, 'maxfev': 300})
            P_opt = np.clip(res.x, 0, None)
            total = np.sum(P_opt)
            if total < 1e-9: continue
            P_opt = P_opt / total
            r_opt = _compute_ratio(P_opt)
            
            if r_opt > best_ratio:
                best_ratio = r_opt
                best_P = P_opt
                
        except Exception:
            pass

    P_out = best_P.reshape((10, 10))
    P_out = np.clip(P_out, 0, None)
    total = np.sum(P_out)
    if total <= 1e-9:
        P_out = np.ones((10, 10)) / 100.0
    else:
        P_out = P_out / total
        
    return P_out

# EVOLVE-BLOCK-END
