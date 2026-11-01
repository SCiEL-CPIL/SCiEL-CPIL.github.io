"""Improved packing algorithm for 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

def _hex_init(N):
    """Generate hexagonal lattice points, select N closest to origin."""
    s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    s *= 0.95
    
    best_points = None
    for attempt in range(30):
        k = int(1.0 / s) + 4
        pts = []
        for j in range(-k, k + 1):
            y = j * s * np.sqrt(3.0) / 2.0
            if abs(y) > 1.0:
                continue
            x_offset = (j % 2) * s / 2.0
            for i in range(-k, k + 1):
                x = (i + x_offset) * s
                r = np.hypot(x, y)
                if r <= 1.0 - s * 0.6:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists, kind='stable')[:N]
            if best_points is None:
                best_points = pts[idx]
        s *= 0.97
    return best_points if best_points is not None else np.random.randn(N, 2) * 0.5

def _compute_radius(pts):
    """Compute the feasible radius for a set of points."""
    n = len(pts)
    if n < 2:
        min_pair = 1.0
    else:
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=1.0)
        if len(pairs) == 0:
            min_pair = 1.0
        else:
            idx1 = np.array([p[0] for p in pairs])
            idx2 = np.array([p[1] for p in pairs])
            dists = np.linalg.norm(pts[idx1] - pts[idx2], axis=1)
            min_pair = np.min(dists) / 2.0
    
    dists_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(dists_boundary)
    
    return min(min_pair, min_boundary)

def _objective_and_grad(pts, temperature):
    """
    Compute objective value and gradient for soft-min packing.
    Objective: minimize -soft_min of all constraints (pair distances/2 and boundary distances).
    """
    n = len(pts)
    
    # Pairwise distances
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=2.0)
    
    pair_vals = np.full(n * (n - 1), 100.0)
    pair_indices = []
    for p in pairs:
        d = np.linalg.norm(pts[p[0]] - pts[p[1]])
        pair_vals.append(d / 2.0)
        pair_indices.append(p)
    
    pair_vals = np.array(pair_vals) if len(pair_vals) > 0 else np.array([100.0])
    
    # Boundary distances
    boundary_vals = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    
    all_vals = np.concatenate([pair_vals, boundary_vals])
    all_vals = np.clip(all_vals, -100, 100)
    
    # Soft min
    min_val = np.min(all_vals)
    exp_vals = np.exp((all_vals - min_val) / temperature)
    weights = exp_vals / np.sum(exp_vals)
    soft_min = min_val - temperature * np.log(np.sum(exp_vals))
    
    obj = -soft_min
    
    # Gradient
    grad = np.zeros_like(pts)
    
    # Pairwise gradient contributions
    pair_weights = weights[:len(pair_vals)]
    for idx, w in zip(pair_indices, pair_weights):
        i, j = idx
        d = np.linalg.norm(pts[i] - pts[j])
        if d > 1e-12:
            diff = (pts[i] - pts[j]) / (2.0 * d)
            grad[i] += diff * w * (-temperature)
            grad[j] -= diff * w * (-temperature)
    
    # Boundary gradient contributions
    boundary_weights = weights[len(pair_vals):]
    for i in range(n):
        r = np.hypot(pts[i, 0], pts[i, 1])
        if r > 1e-12:
            grad[i] += -pts[i] / r * boundary_weights[i] * (-temperature)
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Initial hexagonal packing
    pts = _hex_init(N)
    if pts is None or len(pts) != N:
        pts = np.random.randn(N, 2) * 0.8
    
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    # Multi-stage optimization with decreasing temperature
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002]
    
    for temp in temperatures:
        if time.time() > deadline - 20.0:
            break
        
        def objective(x):
            p = x.reshape(N, 2)
            val, _ = _objective_and_grad(p, temperature=temp)
            return val
        
        def gradient(x):
            p = x.reshape(N, 2)
            _, g = _objective_and_grad(p, temperature=temp)
            return g.flatten()
        
        x0 = pts.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            
            new_pts = result.x.reshape(N, 2)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_pts = new_pts.copy()
                best_r = new_r
                pts = new_pts.copy()
                
        except Exception:
            pass
    
    # Final local search: try small perturbations
    if time.time() < deadline - 10.0:
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            
            # Try a small random perturbation
            perturbation = np.random.randn(N, 2) * 0.001
            candidate = best_pts + perturbation
            
            # Project inside unit circle
            dists = np.hypot(candidate[:, 0], candidate[:, 1])
            mask = dists > 0.999
            if np.any(mask):
                candidate[mask] = candidate[mask] / dists[mask] * 0.999
            
            cand_r = _compute_radius(candidate)
            if cand_r > best_r:
                best_pts = candidate.copy()
                best_r = cand_r
    
    # Ensure validity
    dists = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = dists > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / dists[mask] * 0.999
    
    return best_pts
# EVOLVE-BLOCK-END
