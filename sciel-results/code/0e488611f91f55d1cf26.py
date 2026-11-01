import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import itertools

N = 14
D = 3
DEADLINE = 115.0  # Safety margin from 120s

def _pairwise_distances(pts):
    """Compute pairwise distances between all points."""
    d = cdist(pts, pts)
    mask = ~np.eye(N, dtype=bool)
    return d[mask]

def _true_score(pts):
    """Calculate the actual score: (max/min)^2."""
    d = _pairwise_distances(pts)
    if len(d) == 0:
        return np.inf
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _soft_log_ratio(dist, tau):
    """
    Smooth approximation of log(max_d) - log(min_d) using soft min/max.
    This is a smooth surrogate for the actual objective.
    """
    mask = ~np.eye(N, dtype=bool)
    d = dist[mask]
    
    # Stable soft max: tau * log(sum(exp(d/tau)))
    max_d = np.max(d)
    soft_max = max_d + tau * np.log(np.sum(np.exp((d - max_d) / tau)))
    
    # Stable soft min: -tau * log(sum(exp(-d/tau)))
    min_d = np.min(d)
    soft_min = min_d - tau * np.log(np.sum(np.exp((min_d - d) / tau)))
    
    return soft_max - soft_min

def _objective(x, tau):
    """
    Objective function to minimize.
    Uses a soft log-ratio to encourage uniformity of pairwise distances.
    """
    pts = x.reshape(N, D)
    
    # Center the points to remove translation invariance
    pts = pts - np.mean(pts, axis=0)
    
    d = _pairwise_distances(pts)
    
    # Avoid log(0) and penalize close points
    if np.any(d < 1e-12):
        return 1e10
        
    return _soft_log_ratio(d, tau)

def _generate_fib_sphere_points():
    """Generate points on a sphere using Fibonacci distribution."""
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))

def _generate_icosahedron_plus_2():
    """
    Generate a configuration based on an icosahedron (12 vertices) plus 2 points.
    The icosahedron is a known optimal packing for 12 points on a sphere.
    Adding 2 points near the poles or center might yield good local minima.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Add 2 points. Let's try adding them near the poles (0,0,±1) but slightly off to avoid symmetry breaking issues
    # Or simply add them at the origin? No, distinct points.
    # Let's add them at (0, 0, 1) and (0, 0, -1) but shifted slightly?
    # Actually, the icosahedron has 5-fold symmetry. Adding points on the axis is a natural symmetry extension.
    # Let's try points at (0,0, h) and (0,0, -h).
    # h=1 is the pole. The icosahedron vertices are at distance 1 from center.
    # Distance from pole (0,0,1) to nearest icosahedron vertex (0,1,phi)/norm is:
    # norm = sqrt(1+phi^2) = sqrt(1+2.618) = sqrt(3.618) = 1.902
    # Vertex: (0, 0.5257, 0.8506). Pole: (0,0,1).
    # dist^2 = 0 + 0.5257^2 + (1-0.8506)^2 = 0.276 + 0.022 = 0.298. dist=0.546.
    # This seems reasonable.
    
    h = 1.0
    extra = np.array([[0, 0, h], [0, 0, -h]])
    return np.vstack([verts, extra])

def _generate_rings():
    """
    Generate points in two rings plus poles.
    14 points: 6 on top ring, 6 on bottom ring, 2 poles? Or 7 and 7?
    Let's try 6 top, 6 bottom, 2 poles.
    """
    # Top ring
    angles_top = 2 * np.pi * np.arange(6) / 6
    r_top = 0.8
    h_top = 0.6
    top_pts = np.column_stack([r_top * np.cos(angles_top), r_top * np.sin(angles_top), np.full(6, h_top)])
    
    # Bottom ring
    angles_bot = 2 * np.pi * np.arange(6) / 6 + np.pi / 6
    r_bot = 0.8
    h_bot = -0.6
    bot_pts = np.column_stack([r_bot * np.cos(angles_bot), r_bot * np.sin(angles_bot), np.full(6, h_bot)])
    
    # Poles
    poles = np.array([[0, 0, 1.2], [0, 0, -1.2]])
    
    return np.vstack([top_pts, bot_pts, poles])

def construct():
    t_start = time.time()
    
    best_score = np.inf
    best_pts = None
    
    # Initial candidates
    candidates = []
    candidates.append(_generate_fib_sphere_points())
    candidates.append(_generate_icosahedron_plus_2())
    candidates.append(_generate_rings())
    candidates.append(np.random.randn(N, D))
    
    # Two-stage optimization: coarse (large tau) then fine (small tau)
    tau_stages = [0.3, 0.1, 0.05, 0.02]
    
    # First pass: coarse optimization from all starts
    refined_candidates = []
    
    for guess in candidates:
        if time.time() - t_start > DEADLINE - 10:
            break
            
        x0 = guess.flatten()
        
        try:
            res = minimize(
                _objective, 
                x0, 
                args=(tau_stages[0],),
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if res.fun < 1e5:
                refined_candidates.append(res.x)
        except Exception:
            continue

    # Keep top candidates by objective value
    if refined_candidates:
        cand_scores = []
        for cx in refined_candidates:
            pts = cx.reshape(N, D)
            pts = pts - np.mean(pts, axis=0)
            
            d = _pairwise_distances(pts)
            if d.min() < 1e-6:
                continue
                
            score = _true_score(pts)
            cand_scores.append((score, cx))
        
        cand_scores.sort(key=lambda x: x[0])
        # Keep top 6 for fine-tuning
        top_candidates = [c[1] for c in cand_scores[:6]]
    else:
        top_candidates = [g.flatten() for g in candidates[:3]]
    
    # Second pass: fine-tune top candidates with decreasing tau
    for cx in top_candidates:
        if time.time() - t_start > DEADLINE - 5:
            break
        
        x_curr = cx.copy()
        
        for tau in tau_stages[1:]:
            if time.time() - t_start > DEADLINE - 2:
                break
            
            try:
                res = minimize(
                    _objective,
                    x_curr,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                # Only update if improved
                if res.fun < _objective(x_curr, tau) + 1e-6:
                    x_curr = res.x
            except Exception:
                continue
        
        # Evaluate final score
        pts = x_curr.reshape(N, D)
        pts = pts - np.mean(pts, axis=0)
        
        d = _pairwise_distances(pts)
        if d.min() < 1e-6:
            continue
            
        score = _true_score(pts)
        if score < best_score:
            best_score = score
            best_pts = pts
    
    # If we have time, try a few more random starts with fine tau
    if time.time() - t_start < DEADLINE - 10:
        rng = np.random.default_rng(123)
        for _ in range(10):
            if time.time() - t_start > DEADLINE - 5:
                break
            
            x0 = rng.normal(0, 1, size=(N, D)).flatten()
            
            try:
                # Quick coarse
                res1 = minimize(
                    _objective, x0, args=(0.2,),
                    method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-9}
                )
                if res1.fun > 1e5:
                    continue
                
                # Fine
                res2 = minimize(
                    _objective, res1.x, args=(0.05,),
                    method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-11}
                )
                
                pts = res2.x.reshape(N, D)
                pts = pts - np.mean(pts, axis=0)
                
                d = _pairwise_distances(pts)
                if d.min() < 1e-6:
                    continue
                    
                score = _true_score(pts)
                if score < best_score:
                    best_score = score
                    best_pts = pts
                    
            except Exception:
                continue

    # If no valid points were found (unlikely), fallback to grid
    if best_pts is None:
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
        
    return best_pts
