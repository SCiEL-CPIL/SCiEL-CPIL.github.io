import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE_SECONDS = 140.0
_START = time.time()

def _time_left():
    return DEADLINE_SECONDS - (time.time() - _START)

def _score(points):
    """Calculate the exact packing radius."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Check if points are inside the unit circle
    d2 = np.sum(points * points, axis=1)
    if np.any(d2 > 1.0 + 1e-12):
        return 0.0
        
    # Calculate pairwise distances
    diff = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Calculate distance to boundary
    radii = np.sqrt(d2)
    min_margin = 1.0 - np.max(radii)
    
    r = min(min_pair_dist / 2.0, min_margin)
    return r if r > 0 else 0.0

def _soft_min_obj_and_grad(x, beta):
    """
    Objective: -soft_min(vals) where vals = [pairwise_dists/2, margins]
    We want to maximize min(vals), so we minimize -min(vals).
    soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    -soft_min(x) = 1/beta * log(sum(exp(-beta * x)))
    
    Let f(x) = 1/beta * log(sum(exp(-beta * v_k)))
    grad f = (1/beta) * (sum(-beta * exp(-beta * v_k) * grad v_k) / sum(exp(-beta * v_k)))
           = - sum(exp(-beta * v_k) * grad v_k) / sum(exp(-beta * v_k))
           = - sum(w_k * grad v_k)
    where w_k = exp(-beta * v_k) / sum(exp(-beta * v_k))
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    
    iu = np.triu_indices(N, k=1)
    d2u = d2[iu]
    dists = np.sqrt(d2u)
    
    # Margins
    d2c = np.sum(pts * pts, axis=1)
    margins = 1.0 - np.sqrt(d2c)
    
    # Values to minimize the negative of
    vals = np.concatenate([dists / 2.0, margins])
    
    # Soft min calculation
    xb = beta * vals
    xb = xb - np.min(xb) # for numerical stability
    exp_neg = np.exp(-xb)
    s = np.sum(exp_neg)
    
    # Objective value: -soft_min = 1/beta * log(s)
    val = np.log(s) / beta
    
    # Weights
    w = exp_neg / s
    
    # Gradient calculation
    grad_flat = np.zeros_like(x)
    
    # Pairwise distances contribution
    w_pair = w[:len(iu)]
    for idx in range(len(iu)):
        i, j = iu[0][idx], iu[1][idx]
        dij = diff[i, j]
        d = dists[idx]
        if d < 1e-12:
            continue
        # d (dist/2) / d pts_i = dij / (2 * d)
        # d (dist/2) / d pts_j = -dij / (2 * d)
        coef = w_pair[idx] / (2.0 * d)
        # grad f = - sum(w_k * grad v_k)
        grad_flat[2*i:2*i+2] += -coef * dij
        grad_flat[2*j:2*j+2] += coef * dij
        
    # Margins contribution
    w_marg = w[len(iu):]
    for i in range(N):
        r_i = np.sqrt(d2c[i])
        if r_i < 1e-12:
            continue
        # d margin / d pts_i = -pts_i / r_i
        coef = w_marg[i] / r_i
        grad_flat[2*i:2*i+2] += -coef * (-pts[i]) # -w * grad v = -w * (-pts/r) = w * pts/r
        
    return val, grad_flat

def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit circle."""
    points = []
    # Try different spacings to ensure we get at least N points
    # A spacing of ~0.15 should give plenty of points
    spacing = 0.15
    while len(points) < N:
        # Calculate number of rows/cols needed
        # Hex grid: x = i * spacing + (j % 2) * spacing / 2
        # y = j * spacing * sqrt(3) / 2
        max_coord = 1.0 / spacing + 2
        for j in range(-int(max_coord), int(max_coord) + 1):
            for i in range(-int(max_coord), int(max_coord) + 1):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= (1.0 - spacing/2)**2: # Keep away from boundary
                    points.append([x, y])
        
        if len(points) >= N:
            break
        spacing *= 0.9 # Reduce spacing if not enough points
        
    points = np.array(points)
    # Sort by distance from center to pick the most central ones
    radii = np.sqrt(np.sum(points * points, axis=1))
    idx = np.argsort(radii)
    return points[idx[:N]]

def _random_init(seed):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2*np.pi, N)
    radii = rng.uniform(0, 1.0, N)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _optimize(points, beta_schedule):
    """Run L-BFGS-B with a schedule of beta values."""
    current_points = points.copy()
    for beta in beta_schedule:
        if _time_left() < 2.0:
            break
        def fun_and_grad(x):
            return _soft_min_obj_and_grad(x, beta)
        
        try:
            res = minimize(
                fun_and_grad,
                current_points.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-8}
            )
            current_points = res.x.reshape(N, 2)
        except Exception:
            pass
            
    return current_points

def construct():
    global _START
    _START = time.time()
    
    # Initial configurations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_random_init(42))
    inits.append(_random_init(123))
    inits.append(_random_init(999))
    
    best_points = inits[0]
    best_score = _score(best_points)
    
    # Beta schedule: start smooth, end sharp
    beta_schedule = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    for init in inits:
        if _time_left() < 10.0:
            break
            
        # Optimize this initial configuration
        optimized = _optimize(init, beta_schedule)
        score = _score(optimized)
        
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
            
    # If we have time, do a final high-beta polish on the best result
    if _time_left() > 5.0:
        final_beta = [500.0, 1000.0]
        polished = _optimize(best_points, final_beta)
        score = _score(polished)
        if score > best_score:
            best_score = score
            best_points = polished
            
    # Safety check
    if not np.all(np.isfinite(best_points)):
        best_points = _hex_lattice_init()
        
    return best_points
