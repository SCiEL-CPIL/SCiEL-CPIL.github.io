"""Stronger packing algorithm for 300 circles in unit disk."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize, linear_sum_assignment
import time
import math


def _hex_init():
    """Generate initial hexagonal lattice points inside unit disk."""
    # Try different spacings to get close to N points
    best_pts = None
    best_score = -1.0
    for spacing in np.linspace(0.03, 0.12, 40):
        k = int(1.0 / spacing) + 2
        points = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = math.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    points.append((x, y))
        if len(points) < N:
            continue
        pts = np.array(points)
        # Compute score for this initial arrangement
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)
        min_pair_dist = dists[:, 1].min()
        min_contain = (1.0 - np.linalg.norm(pts, axis=1)).min()
        score = min(min_pair_dist / 2.0, min_contain)
        if score > best_score:
            best_score = score
            best_pts = pts
    if best_pts is None:
        # Fallback: uniform grid
        pts = np.random.randn(N, 2) * 0.5
        best_pts = pts
    # Select N points with best local score
    if len(best_pts) > N:
        tree = cKDTree(best_pts)
        dists, _ = tree.query(best_pts, k=2)
        min_pair_dist = dists[:, 1]
        min_contain = (1.0 - np.linalg.norm(best_pts, axis=1))
        point_scores = np.minimum(min_pair_dist / 2.0, min_contain)
        idx = np.argsort(-point_scores)[:N]
        best_pts = best_pts[idx]
    return best_pts


def _soft_min(dists, tau=0.001):
    """Soft minimum using log-sum-exp for differentiability."""
    # Use a scaled soft-min that approaches min as tau -> 0
    m = np.min(dists)
    # Clip to avoid numerical issues
    exp_terms = np.exp(-(dists - m) / tau)
    return m - tau * np.log(np.sum(exp_terms))


def _score(pts):
    """Compute the actual score: min(min pair dist/2, min containment)."""
    if len(pts) < 2:
        return 0.0
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    min_pair_dist = dists[:, 1].min()
    min_contain = (1.0 - np.linalg.norm(pts, axis=1)).min()
    return min(min_pair_dist / 2.0, min_contain)


def _obj_flat(x, tau=0.001):
    """Objective to minimize: negative of soft score."""
    pts = x.reshape(-1, 2)
    # Containment penalty: want all points inside disk
    norms = np.linalg.norm(pts, axis=1)
    contain_dists = 1.0 - norms  # Distance to boundary (positive if inside)
    
    # Pairwise distances using KDTree
    if len(pts) < 2:
        return 100.0
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    pair_dists = dists[:, 1]
    
    # Soft min of all constraints (want them all large)
    soft_min_contain = _soft_min(contain_dists, tau)
    soft_min_pair = _soft_min(pair_dists / 2.0, tau)
    
    # We want to maximize min(soft_min_pair, soft_min_contain)
    # So minimize the negative, but also penalize if any constraint is violated
    # Use a barrier: if any point is outside, add large penalty
    if np.any(contain_dists < 0):
        return 100.0 + np.sum(contain_dists[contain_dists < 0]) ** 2
    
    score = min(soft_min_pair, soft_min_contain)
    return -score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial placement
    pts = _hex_init()
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Local optimization using L-BFGS-B on the smooth surrogate
    tau = 0.002
    n_iter_budget = 300  # Max iterations total
    
    # Try multiple restarts with different tau values
    tau_schedule = [0.005, 0.003, 0.002, 0.001, 0.0005]
    
    for tau in tau_schedule:
        if time.time() > deadline - 20.0:
            break
        
        current_pts = best_pts.copy()
        x0 = current_pts.flatten()
        
        # Use L-BFGS-B with bounds to keep points in a reasonable region
        # Actually, we handle containment via penalty, so no strict bounds needed
        # but let's add loose bounds for stability
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        def obj(x):
            return _obj_flat(x, tau)
        
        # Run optimization
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': n_iter_budget // len(tau_schedule),
                         'ftol': 1e-10, 'gtol': 1e-8,
                         'maxls': 40}
            )
            if result.x is not None and np.all(np.isfinite(result.x)):
                new_pts = result.x.reshape(-1, 2)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    n_iter_budget -= 1  # Use some budget
        except Exception:
            pass
    
    # Final polish with very small tau if time permits
    if time.time() < deadline - 30.0:
        current_pts = best_pts.copy()
        x0 = current_pts.flatten()
        
        def obj_pure(x):
            pts = x.reshape(-1, 2)
            norms = np.linalg.norm(pts, axis=1)
            contain_dists = 1.0 - norms
            if np.any(contain_dists < 0):
                return 100.0 + np.sum(contain_dists[contain_dists < 0]) ** 2
            tree = cKDTree(pts)
            dists, _ = tree.query(pts, k=2)
            pair_dists = dists[:, 1]
            # Use exact min for final polish
            score = min(pair_dists.min() / 2.0, contain_dists.min())
            return -score
        
        bounds = [(-1.0, 1.0)] * (2 * N)
        try:
            result = minimize(
                obj_pure, x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10,
                         'maxls': 40}
            )
            if result.x is not None and np.all(np.isfinite(result.x)):
                new_pts = result.x.reshape(-1, 2)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
        except Exception:
            pass
    
    # Return the best valid configuration
    # Ensure all points are strictly inside
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms >= 1.0):
        # Clip to inside
        scale = np.ones(N)
        mask = norms >= 1.0
        scale[mask] = (1.0 - 1e-10) / norms[mask]
        best_pts = best_pts * scale[:, np.newaxis]
    
    return best_pts

# EVOLVE-BLOCK-END
