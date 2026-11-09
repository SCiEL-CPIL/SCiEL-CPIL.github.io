"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import time


def _score(pts):
    """Compute the common radius: min of half min pairwise dist and min boundary margin."""
    n = len(pts)
    if n == 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Boundary
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    min_bound = 1.0 - np.max(norms)
    return min(min_pair, min_bound)


def _hex_lattice_init(n):
    """Generate a hexagonal lattice point cloud and pick n points near center."""
    # Try various spacings to get at least n points in the unit disk
    for spacing in np.linspace(0.05, 0.3, 200):
        k = int(1.0 / spacing) + 2
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                    pts.append([x, y])
        if len(pts) >= n:
            pts = np.array(pts)
            # Sort by distance from center and take n
            d = np.sqrt(np.sum(pts ** 2, axis=1))
            idx = np.argsort(d, kind='stable')[:n]
            return pts[idx]
    # Fallback: random points
    return np.random.uniform(-0.8, 0.8, size=(n, 2))


def _build_constraints(pts):
    """Build pairwise and boundary constraints for SLSQP.
    
    Variables: x = [x0, y0, x1, y1, ..., x_{n-1}, y_{n-1}]
    Constraints:
      - For each pair (i,j): x_i^2 + y_i^2 + x_j^2 + y_j^2 - 2*x_i*x_j - 2*y_i*y_j >= r^2
        where r is the target radius. But we want to maximize r, so we use a different approach.
    
    Actually, let's use a simpler formulation. We'll optimize positions to maximize the minimum distance.
    Use penalty-based approach with SLSQP.
    """
    pass


def _penalized_objective(x, n, r_target):
    """Penalized objective: minimize negative radius + penalty for violations."""
    pts = x.reshape(n, 2)
    
    # Compute current radius
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    min_bound = 1.0 - np.max(norms)
    
    current_r = min(min_pair, min_bound)
    
    # Penalty for being below target
    penalty = 0.0
    if current_r < r_target:
        # Pairwise violations
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((pts[i] - pts[j]) ** 2))
                if d < 2 * r_target:
                    penalty += (2 * r_target - d) ** 2
        # Boundary violations
        for i in range(n):
            norm = np.sqrt(np.sum(pts[i] ** 2))
            if norm > 1.0 - r_target:
                penalty += (norm - (1.0 - r_target)) ** 2
    
    return -current_r + 0.1 * penalty


def _pso_optimize(pts, n, max_iters=200, swarm_size=20, deadline=None):
    """Particle Swarm Optimization for circle packing."""
    if deadline is None:
        deadline = time.time() + 120
    
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Initialize swarm
    swarm = []
    for _ in range(swarm_size):
        p = pts + np.random.normal(0, 0.05, size=(n, 2))
        # Keep within bounds
        norms = np.sqrt(np.sum(p ** 2, axis=1))
        mask = norms > 0.95
        p[mask] *= 0.95 / norms[mask, np.newaxis]
        swarm.append(p)
    
    v_max = 0.1
    c1 = 1.5
    c2 = 1.5
    
    for it in range(max_iters):
        if time.time() > deadline - 10:
            break
        
        for i in range(swarm_size):
            score = _score(swarm[i])
            
            # Update personal best
            if score > best_score:
                best_score = score
                best_pts = swarm[i].copy()
            
            # Randomly pick a neighbor's best (or use global best for simplicity)
            # Use global best
            r1 = np.random.rand(n, 2)
            r2 = np.random.rand(n, 2)
            
            v = c1 * r1 * (best_pts - swarm[i]) + c2 * r2 * (best_pts - swarm[i])
            
            # Clamp velocity
            v_norms = np.sqrt(np.sum(v ** 2, axis=1))
            mask = v_norms > v_max
            v[mask] *= v_max / v_norms[mask, np.newaxis]
            
            new_p = swarm[i] + v
            
            # Keep within unit disk
            norms = np.sqrt(np.sum(new_p ** 2, axis=1))
            mask = norms > 0.98
            new_p[mask] *= 0.98 / norms[mask, np.newaxis]
            
            swarm[i] = new_p
        
        # Occasionally add diversity
        if it % 20 == 19:
            idx = np.random.randint(swarm_size)
            swarm[idx] = best_pts + np.random.normal(0, 0.1, size=(n, 2))
            norms = np.sqrt(np.sum(swarm[idx] ** 2, axis=1))
            mask = norms > 0.98
            swarm[idx][mask] *= 0.98 / norms[mask, np.newaxis]
    
    return best_pts, best_score


def _slsqp_optimize(pts, n, deadline=None):
    """SLSQP optimization with explicit constraints."""
    if deadline is None:
        deadline = time.time() + 120
    
    x0 = pts.flatten()
    
    def objective(x):
        pts = x.reshape(n, 2)
        # We want to maximize the minimum radius, so minimize negative
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        norms = np.sqrt(np.sum(pts ** 2, axis=1))
        min_bound = 1.0 - np.max(norms)
        return -min(min_pair, min_bound)
    
    # Boundary constraints: each point must be inside unit disk
    # x_i^2 + y_i^2 <= 1 - r^2, but r is variable, so use x_i^2 + y_i^2 <= 0.81 (r >= 0.1)
    bounds = []
    for i in range(n):
        bounds.append(((-0.9, 0.9), (-0.9, 0.9)))
    
    # Pairwise distance constraints: ||p_i - p_j|| >= 2*r_min
    # This is tricky with SLSQP since r is implicit. Use penalty instead.
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-8})
        if res.success or res.fun < objective(x0):
            pts_opt = res.x.reshape(n, 2)
            score = _score(pts_opt)
            if score > _score(pts):
                return pts_opt, score
    except Exception:
        pass
    
    return pts, _score(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Initial guess from hexagonal lattice
    pts = _hex_lattice_init(N)
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # PSO refinement
    if time.time() < deadline - 30:
        pts_pso, score_pso = _pso_optimize(best_pts, N, max_iters=100, swarm_size=15,
                                           deadline=deadline - 30)
        if score_pso > best_score:
            best_pts = pts_pso
            best_score = score_pso
    
    # SLSQP refinement
    if time.time() < deadline - 10:
        pts_slsqp, score_slsqp = _slsqp_optimize(best_pts, N, deadline=deadline - 5)
        if score_slsqp > best_score:
            best_pts = pts_slsqp
            best_score = score_slsqp
    
    # Final PSO polish with smaller steps
    if time.time() < deadline - 5:
        pts_final, score_final = _pso_optimize(best_pts, N, max_iters=50, swarm_size=10,
                                               deadline=deadline - 2)
        if score_final > best_score:
            best_pts = pts_final
            best_score = score_final
    
    # Ensure all points are strictly inside
    norms = np.sqrt(np.sum(best_pts ** 2, axis=1))
    mask = norms > 0.999
    if np.any(mask):
        best_pts[mask] *= 0.999 / norms[mask, np.newaxis]
    
    return best_pts
# EVOLVE-BLOCK-END
