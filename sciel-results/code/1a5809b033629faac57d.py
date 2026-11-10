"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hexagonal_init(n, radius=1.0):
    """Generate initial hexagonal lattice points within a circle."""
    points = []
    # Try different spacings to get enough points
    for spacing in np.linspace(0.05, 0.5, 20):
        k = int(1.0 / spacing) + 2
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= radius - spacing / 2.0:
                    points.append([x, y])
        if len(points) >= n:
            points = np.array(points)
            # Sort by distance from origin and take first n
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists, kind='stable')[:n]
            return points[idx]
    # Fallback: uniform random
    return np.random.uniform(-0.8, 0.8, (n, 2))

def _pack_radius(pts):
    """Compute the packing radius (minimum of pair distances/2 and boundary clearance)."""
    n = len(pts)
    if n < 2:
        return min(1.0 - np.linalg.norm(pts, axis=1)) if n > 0 else 0.0
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary clearance
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    min_boundary = np.min(boundary)
    
    return min(min_pair, min_boundary)

def _soft_min_surrogate(pts, beta=100.0):
    """
    Smooth surrogate for the packing radius using log-sum-exp.
    We want to maximize min(min_pair/2, min_boundary).
    This is equivalent to maximizing -soft_max(-min_pair/2, -min_boundary).
    """
    n = len(pts)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # We want to maximize the minimum of:
    # - dists[i,j]/2 for all i<j
    # - (1 - |pts[i]|) for all i
    # 
    # So we minimize the negative of that, using soft-max for smoothness.
    
    # Collect all values we want to take the min of
    # Pair distances / 2
    pair_vals = []
    for i in range(n):
        for j in range(i+1, n):
            pair_vals.append(dists[i, j] / 2.0)
    
    # Boundary clearances
    boundary_vals = 1.0 - np.linalg.norm(pts, axis=1)
    
    # Combine all values
    all_vals = np.concatenate([np.array(pair_vals), boundary_vals])
    
    # Soft-min = -soft_max(-all_vals)
    # soft_max(x) = (1/beta) * log(sum(exp(beta * x)))
    neg_vals = -all_vals
    max_neg = np.max(neg_vals)
    soft_max_val = max_neg + (1.0 / beta) * np.log(np.sum(np.exp(beta * (neg_vals - max_neg))))
    soft_min_val = -soft_max_val
    
    return soft_min_val

def _neg_soft_min_surrogate(pts, beta=100.0):
    """Negative of the soft-min surrogate for minimization."""
    return -_soft_min_surrogate(pts, beta)

def _grad_neg_soft_min(pts, beta=100.0):
    """
    Gradient of the negative soft-min surrogate.
    """
    n = len(pts)
    if n < 2:
        return np.zeros_like(pts)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Collect all values
    pair_vals = []
    pair_idx = []
    for i in range(n):
        for j in range(i+1, n):
            pair_vals.append(dists[i, j] / 2.0)
            pair_idx.append((i, j))
    
    boundary_vals = 1.0 - np.linalg.norm(pts, axis=1)
    
    all_vals = np.concatenate([np.array(pair_vals), boundary_vals])
    
    # Soft-max weights
    neg_vals = -all_vals
    max_neg = np.max(neg_vals)
    exp_vals = np.exp(beta * (neg_vals - max_neg))
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Gradient of soft_max w.r.t. all_vals is -weights
    # Gradient of neg_soft_min w.r.t. all_vals is weights
    
    # Now propagate to pts
    grad = np.zeros_like(pts)
    
    # Pair contributions
    for k, (i, j) in enumerate(pair_idx):
        w = weights[k]
        # d(pair_val)/d(pts[i]) = (pts[i] - pts[j]) / (2 * dist)
        # d(pair_val)/d(pts[j]) = (pts[j] - pts[i]) / (2 * dist)
        d = dists[i, j]
        if d > 1e-10:
            grad[i] += w * (pts[i] - pts[j]) / (2.0 * d)
            grad[j] += w * (pts[j] - pts[i]) / (2.0 * d)
    
    # Boundary contributions
    for i in range(n):
        w = weights[len(pair_vals) + i]
        # d(boundary_val)/d(pts[i]) = -pts[i] / |pts[i]|
        norm_i = np.linalg.norm(pts[i])
        if norm_i > 1e-10:
            grad[i] += w * (-pts[i] / norm_i)
    
    return grad

def construct():
    """Construct the optimal packing of 50 circles in a unit circle."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    best_pts = None
    best_score = -np.inf
    
    # Try multiple initializations and optimization runs
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial configuration
        if restart == 0:
            pts = _hexagonal_init(N)
        else:
            # Add some randomness
            base = _hexagonal_init(N)
            pts = base + np.random.normal(0, 0.05, base.shape)
        
        # Ensure points are inside the circle
        norms = np.linalg.norm(pts, axis=1)
        mask = norms < 1.0
        pts = pts[mask]
        if len(pts) < N:
            # Add random points
            while len(pts) < N:
                p = np.random.uniform(-0.9, 0.9, 2)
                if np.linalg.norm(p) < 0.95:
                    pts = np.vstack([pts, p])
            pts = pts[:N]
        
        # Optimize using L-BFGS-B
        try:
            # Start with moderate beta, increase for final polish
            betas = [50.0, 100.0, 200.0]
            
            for beta in betas:
                if time.time() > deadline:
                    break
                
                result = minimize(
                    _neg_soft_min_surrogate,
                    pts,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=_grad_neg_soft_min,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < _neg_soft_min_surrogate(pts, beta):
                    pts = result.x
                
                # Check deadline
                if time.time() > deadline:
                    break
            
            # Final polish with high beta
            if time.time() < deadline - 10:
                result = minimize(
                    _neg_soft_min_surrogate,
                    pts,
                    args=(500.0,),
                    method='L-BFGS-B',
                    jac=_grad_neg_soft_min,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if result.success:
                    pts = result.x
            
            # Compute actual score
            score = _pack_radius(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        
        except Exception as e:
            continue
    
    # If we didn't find anything good, use the hexagonal init
    if best_pts is None:
        best_pts = _hexagonal_init(N)
    
    # Final validation: ensure all points are inside the circle
    norms = np.linalg.norm(best_pts, axis=1)
    best_pts = best_pts[norms < 1.0]
    if len(best_pts) < N:
        # Fallback to hexagonal
        best_pts = _hexagonal_init(N)
    
    return best_pts[:N]

# EVOLVE-BLOCK-END
