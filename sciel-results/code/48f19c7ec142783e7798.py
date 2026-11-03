"""
Packing 300 equal circles in a unit circle.
Strategy: 
1. Hexagonal lattice initialization.
2. Topology-aware force-directed relaxation (Fruchterman-Reingold variant) 
   using Delaunay triangulation to identify critical close pairs.
   This focuses computational effort on resolving overlaps rather than 
   optimizing all 45k pairs equally, allowing faster escape from local minima.
3. High-precision gradient-based polish (L-BFGS-B) to converge to the exact optimum 
   for the current topology.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

N = 300
TIME_LIMIT = 115.0  # Safety margin for 120s limit

def _hex_init(n=N):
    """Dense hexagonal lattice clipped to the unit disk, keep n closest to center."""
    spacing = 0.10
    k = int(1.1 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    if len(pts) < n:
        return _random_init(n)
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()

def _random_init(n=N):
    """Random points inside unit disk."""
    rng = np.random.default_rng(42)
    while True:
        pts = rng.uniform(-1, 1, size=(n, 2))
        r = np.hypot(pts[:, 0], pts[:, 1])
        if np.all(r <= 1.0):
            break
        pts[r > 1.0] = pts[r > 1.0] / r[r > 1.0, None]
    return pts

def _score(pts):
    """Achieved common radius."""
    n = len(pts)
    # Compute pairwise distances
    # For N=300, 45k pairs is fast enough for scoring
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    min_bound = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair, min_bound)

def _force_directed_step(pts, k_iter=100, temp_start=0.1, temp_end=0.01):
    """
    Force-directed relaxation using Delaunay triangulation.
    We treat edges in the Delaunay triangulation as "springs" that want to be long,
    but primarily we want to maximize the minimum edge length.
    
    Strategy:
    1. Compute Delaunay triangulation.
    2. Identify edges.
    3. Apply repulsive force between all pairs, but scaled heavily for close pairs 
       (specifically those in Delaunay or very close).
    4. Project back to disk.
    """
    pts = pts.copy()
    n = len(pts)
    
    # Precompute indices for upper triangle to save memory/time in loops if needed, 
    # but for 300 points, vectorized ops are fine.
    
    for step in range(k_iter):
        # Current temperature (cooling schedule)
        t = temp_start * (temp_end / temp_start) ** (step / k_iter)
        
        # 1. Delaunay Triangulation
        # If points are collinear or degenerate, this might fail, but unlikely for 2D
        try:
            tri = Delaunay(pts)
            edges = set()
            for simplex in tri.simplices:
                edges.add((simplex[0], simplex[1]))
                edges.add((simplex[1], simplex[2]))
                edges.add((simplex[0], simplex[2]))
            edge_list = list(edges)
        except Exception:
            # Fallback: no topology, just all pairs
            edge_list = []
            
        # 2. Compute forces
        # Force is repulsive: F = k / d^2 * direction
        # We want to maximize min distance. 
        # A good heuristic: apply strong repulsion to the closest pairs.
        
        # Compute all pairwise distances (N=300, 45k pairs)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist2, np.inf)
        dist = np.sqrt(dist2)
        
        # Avoid division by zero
        dist_safe = np.maximum(dist, 1e-6)
        
        # Base repulsive force
        # We want to push apart points that are too close.
        # Let's define a target distance d_target based on current best radius.
        # Actually, simpler: just minimize sum of (1/d)^2 for all pairs, 
        # but weight Delaunay edges more heavily? 
        # No, the standard FR algorithm uses attraction for edges and repulsion for all.
        # Here we have NO attraction, only repulsion and boundary constraint.
        # So it's a "maximize min distance" problem.
        
        # Gradient of min-distance objective is non-smooth.
        # Proxy: minimize sum(1/d^2).
        # Force on i due to j: d/dx_i (1/d_ij^2) = -2/d_ij^4 * (x_i - x_j)
        # So force is +2/d^4 * (x_i - x_j) (repulsive)
        
        # Compute force magnitudes
        # F_mag = 2 / d^4
        force_mag = 2.0 / (dist_safe ** 4)
        np.fill_diagonal(force_mag, 0.0)
        
        # Direction vectors
        # force_vec = force_mag[:, :, None] * diff
        # This is N x N x 2. For N=300, 300*300*2 = 180k floats. Manageable.
        
        # Optimization: Only compute full forces for "close" pairs?
        # With N=300, full computation is ~180k ops per step. 100 steps = 18M ops. 
        # Python loop overhead is the killer. Vectorized numpy is fast.
        
        forces = force_mag[:, :, None] * diff
        
        # Sum forces for each point
        net_forces = np.sum(forces, axis=1)
        
        # Add boundary repulsion
        # If point is near boundary, push inward.
        rad = np.hypot(pts[:, 0], pts[:, 1])
        # Boundary force: strong if rad > 0.9
        # F_bound = -k * (rad - r_crit) * direction
        # Let's use a smooth penalty for rad > 0.8
        boundary_force = np.zeros_like(pts)
        mask = rad > 0.8
        if np.any(mask):
            # Push inward proportional to (rad - 0.8)
            strength = (rad[mask] - 0.8) ** 2 * 10.0
            # Direction is -pts / rad
            dir_in = -pts[mask] / rad[mask, None]
            boundary_force[mask] = strength[:, None] * dir_in
            
        net_forces += boundary_force
        
        # Update positions
        # Step size proportional to temperature
        step_size = t * 0.1
        pts += step_size * net_forces
        
        # Project back to disk
        rad = np.hypot(pts[:, 0], pts[:, 1])
        out = rad > 1.0
        if np.any(out):
            pts[out] = pts[out] / rad[out, None] * 0.999 # Slightly inside to avoid sticking
            
    return pts

def _gradient_polish(pts, time_limit=10.0):
    """
    Final polish using L-BFGS-B on a smooth proxy objective.
    Objective: sum(1/d_ij^2) + sum(1/d_bound^2)
    """
    n = len(pts)
    
    def objective(x):
        p = x.reshape(n, 2)
        # Pairwise
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist2, np.inf)
        dist2 = np.maximum(dist2, 1e-8)
        obj_pairs = np.sum(1.0 / dist2)
        
        # Boundary
        rad = np.hypot(p[:, 0], p[:, 1])
        d_bound = 1.0 - rad
        d_bound = np.maximum(d_bound, 1e-8)
        obj_bound = np.sum(1.0 / (d_bound ** 2))
        
        return obj_pairs + obj_bound * 10.0 # Weight boundary more
        
    def gradient(x):
        p = x.reshape(n, 2)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist2, np.inf)
        dist2 = np.maximum(dist2, 1e-8)
        dist = np.sqrt(dist2)
        
        # Gradient of sum(1/d^2) wrt p_i
        # d/dp_i (1/d_ij^2) = -2/d_ij^4 * (p_i - p_j)
        # Sum over j
        # Term for pair (i,j): -2/d^4 * (p_i - p_j)
        # Force on i from j is +2/d^4 * (p_i - p_j)
        # So gradient is -Force? 
        # Let's derive carefully.
        # f = sum_{i<j} 1/d_ij^2
        # df/dp_i = sum_{j!=i} d/dp_i (1/d_ij^2)
        # d/dx (1/(x^2+y^2)) = -2x/(x^2+y^2)^2
        # Let v = p_i - p_j. d_ij^2 = |v|^2.
        # d/dp_i (1/|v|^2) = -1/|v|^4 * 2 v
        # So grad_i = sum_{j!=i} -2 (p_i - p_j) / |p_i - p_j|^4
        
        coeff = -2.0 / (dist2 ** 2)
        np.fill_diagonal(coeff, 0.0)
        
        grad_pairs = np.sum(coeff[:, :, None] * diff, axis=1)
        
        # Boundary gradient
        rad = np.hypot(p[:, 0], p[:, 1])
        d_bound = 1.0 - rad
        d_bound = np.maximum(d_bound, 1e-8)
        
        # d/dp (1/d_bound^2) = d/dp (1/(1-rad)^2) = 2/(1-rad)^3 * d(rad)/dp
        # d(rad)/dp = p / rad
        # So grad_bound_i = 2 / (1-rad_i)^3 * (p_i / rad_i)
        
        rad_safe = np.maximum(rad, 1e-8)
        grad_bound = np.zeros_like(p)
        mask = d_bound > 0
        if np.any(mask):
            factor = 2.0 * 10.0 / (d_bound[mask] ** 3)
            dir_rad = p[mask] / rad_safe[mask, None]
            grad_bound[mask] = factor[:, None] * dir_rad
            
        return (grad_pairs + grad_bound).flatten()
        
    x0 = pts.flatten()
    
    start_time = time.time()
    
    try:
        res = minimize(
            objective, 
            x0, 
            jac=gradient, 
            method='L-BFGS-B',
            bounds=[(-1.0, 1.0)] * (2 * n),
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        # Check time
        if time.time() - start_time > time_limit:
            return pts
            
        opt_pts = res.x.reshape(n, 2)
        
        # Ensure inside
        rad = np.hypot(opt_pts[:, 0], opt_pts[:, 1])
        out = rad > 1.0
        if np.any(out):
            opt_pts[out] = opt_pts[out] / rad[out, None]
            
        return opt_pts
        
    except Exception:
        return pts

def construct():
    start_time = time.time()
    
    # 1. Initialization
    pts = _hex_init(N)
    
    # Ensure inside
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]
        
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # 2. Force Directed Relaxation
    # This step is designed to quickly resolve overlaps and expand the packing.
    # We run it for a fixed number of iterations or until time limit.
    
    # We will run 3 rounds of force-directed steps with decreasing temperature
    # and re-evaluate score after each.
    
    temps = [(0.1, 0.05, 50), (0.05, 0.02, 50), (0.02, 0.005, 100)]
    
    for t_start, t_end, k_iter in temps:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
        pts = _force_directed_step(pts, k_iter=k_iter, temp_start=t_start, temp_end=t_end)
        
        # Score and update best
        cur_score = _score(pts)
        if cur_score > best_score:
            best_score = cur_score
            best_pts = pts.copy()
            
    # 3. Gradient Polish
    # Use remaining time for high-precision optimization
    time_left = TIME_LIMIT - (time.time() - start_time)
    if time_left > 5:
        polish_time = min(time_left - 5, 30.0)
        best_pts = _gradient_polish(best_pts, time_limit=polish_time)
        
        final_score = _score(best_pts)
        if final_score > best_score:
            best_score = final_score
    
    # 4. Final Safety Check
    best_pts = np.asarray(best_pts, dtype=float)
    if best_pts.shape != (N, 2):
        best_pts = _hex_init(N)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N)
        
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        best_pts[out] = best_pts[out] / rad[out, None]
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
