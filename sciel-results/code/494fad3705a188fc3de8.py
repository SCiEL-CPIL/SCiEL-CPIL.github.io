"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)
TRI_AREA = S3 / 4.0

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I = _IDX_TRIPLES[:, 0]
_J = _IDX_TRIPLES[:, 1]
_K = _IDX_TRIPLES[:, 2]


def _bary_to_cart(bc):
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * (S3 / 2)
    return np.column_stack([x, y])


def _cart_to_bary(points):
    x = points[:, 0]
    y = points[:, 1]
    c = y / (S3 / 2)
    b = x - 0.5 * c
    return np.column_stack([b, c])


def _project_to_bary_simplex(bc):
    """Project barycentric coordinates to the valid simplex."""
    b = bc[:, 0].copy()
    c = bc[:, 1].copy()
    b = np.clip(b, 0, None)
    c = np.clip(c, 0, None)
    excess = b + c - 1.0
    if np.any(excess > 0):
        scale = 1.0 / (1.0 + excess)
        b = b * scale
        c = c * scale
    return np.column_stack([b, c])


def _clip_to_triangle(points):
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    y_max = S3 / 2
    y = np.clip(y, 0, y_max)
    x_min = y / S3
    x_max = 1 - y / S3
    x = np.clip(x, x_min, x_max)
    return np.column_stack([x, y])


def _dets_from_bary(bary):
    """Compute all 165 determinants (twice the area in barycentric coords)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    Pi = P[_I]
    Pj = P[_J]
    Pk = P[_K]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
    return det


def _min_area_ratio_from_bary(bary):
    """Compute min area ratio (min det / TRI_AREA) from barycentric coords."""
    dets = _dets_from_bary(bary)
    min_det = np.min(np.abs(dets))
    return min_det / TRI_AREA


def _structured_init_fps(rng, n_points=N):
    """
    Farthest Point Sampling based structured initialization.
    Starts with vertices and iteratively adds the point farthest from all existing points.
    """
    # Start with the three vertices
    bary = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    
    for _ in range(n_points - 3):
        # Compute distance from each candidate to all existing points
        # Use a grid of candidate points
        grid_size = 50
        us = np.linspace(0, 1, grid_size)
        vs = np.linspace(0, 1, grid_size)
        U, V = np.meshgrid(us, vs)
        candidates = np.column_stack([U.ravel(), V.ravel()])
        
        # Filter to valid barycentric coords
        valid_mask = (candidates[:, 0] >= 0) & (candidates[:, 1] >= 0) & (candidates[:, 0] + candidates[:, 1] <= 1)
        candidates = candidates[valid_mask]
        
        # Compute minimum distance to each existing point
        min_dists = np.inf * np.ones(len(candidates))
        for existing in bary:
            dists = np.sqrt(np.sum((candidates - existing) ** 2, axis=1))
            min_dists = np.minimum(min_dists, dists)
        
        # Pick the candidate with maximum minimum distance
        best_idx = np.argmax(min_dists)
        bary = np.vstack([bary, candidates[best_idx]])
    
    # Add small random perturbation to avoid degenerate configurations
    noise = rng.normal(0, 0.01, size=bary.shape)
    bary = bary + noise
    bary = _project_to_bary_simplex(bary)
    
    return bary


def _epigraph_objective(x, t_init, mu=1.0):
    """
    Epigraph formulation with interior-point barrier.
    x: (2N+1,) array, first 2N are barycentric coords, last is t.
    t_init: initial value of t for scaling.
    mu: barrier parameter.
    
    We want to maximize t such that |det_i| >= t for all i, and points in triangle.
    This is equivalent to minimizing -t subject to:
      |det_i| - t >= 0 for all i (non-convex!)
      b_i >= 0, c_i >= 0, b_i + c_i <= 1 for all i
    
    For the barrier method, we penalize violations of these constraints.
    """
    bary = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    # Clamp barycentric coords
    b = np.clip(bary[:, 0], 0, None)
    c = np.clip(bary[:, 1], 0, None)
    excess = b + c - 1.0
    if np.any(excess > 0):
        scale = 1.0 / (1.0 + excess)
        b = b * scale
        c = c * scale
    
    bary_clamped = np.column_stack([b, c])
    
    # Compute determinants
    dets = _dets_from_bary(bary_clamped)
    areas = np.abs(dets)
    
    # Barrier for area constraints: |det_i| >= t
    # Violation: t - |det_i| > 0
    violations = np.maximum(0, t - areas)
    barrier_area = np.sum(violations ** 2)
    
    # Barrier for simplex constraints (should be handled by clamping, but add for smoothness)
    excess_b = np.maximum(0, -b)
    excess_c = np.maximum(0, -c)
    excess_sum = np.maximum(0, b + c - 1)
    barrier_simplex = np.sum(excess_b ** 2) + np.sum(excess_c ** 2) + np.sum(excess_sum ** 2)
    
    # Objective: minimize -t + barrier terms
    objective = -t + mu * (barrier_area + 10.0 * barrier_simplex)
    
    return objective


def _epigraph_gradient(x, t_init, mu=1.0):
    """
    Gradient of the epigraph objective.
    """
    bary = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    b = np.clip(bary[:, 0], 0, None)
    c = np.clip(bary[:, 1], 0, None)
    excess = b + c - 1.0
    if np.any(excess > 0):
        scale = 1.0 / (1.0 + excess)
        b = b * scale
        c = c * scale
    
    bary_clamped = np.column_stack([b, c])
    
    u = bary_clamped[:, 0]
    v = bary_clamped[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    dets = _dets_from_bary(bary_clamped)
    areas = np.abs(dets)
    
    # Gradient of barrier_area = sum(max(0, t - areas)^2)
    # d(barrier_area)/d(t) = -2 * sum(max(0, t - areas))
    # d(barrier_area)/d(point_k) involves chain rule through areas
    
    violations = np.maximum(0, t - areas)
    grad_t = -1.0 - 2.0 * mu * np.sum(violations)
    
    # For the point gradients, we need d(areas)/d(bary)
    # This is complex; use finite differences for simplicity in this implementation
    # or return None to let scipy use numerical gradients
    
    return None  # Let scipy use numerical gradient


def _hard_feasibility_check(bary, t):
    """Check if barycentric coords satisfy |det_i| >= t for all i."""
    dets = _dets_from_bary(bary)
    areas = np.abs(dets)
    return np.all(areas >= t - 1e-10)


def _local_refine_slsqp(bary_init, t_target, max_iter=100):
    """
    Local refinement using SLSQP to maximize t subject to constraints.
    """
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]})
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    def objective(x):
        b = x[:2*N].reshape(N, 2)
        b = _project_to_bary_simplex(b)
        dets = _dets_from_bary(b)
        return -np.min(np.abs(dets))
    
    x0 = bary_init.flatten()
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-8}
        )
        bary_opt = _project_to_bary_simplex(result.x.reshape(N, 2))
        return bary_opt
    except:
        return bary_init


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -float('inf')
    
    # Phase 1: FPS-based structured initialization
    bary_fps = _structured_init_fps(rng)
    points_fps = _bary_to_cart(bary_fps)
    score_fps = _min_area_ratio_from_bary(bary_fps)
    
    if score_fps > best_score:
        best_score = score_fps
        best_points = points_fps.copy()
    
    # Phase 2: Multi-start with interior-point barrier method
    n_starts = 12
    
    for i in range(n_starts):
        if time.time() > deadline - 20.0:
            break
        
        # Generate different initializations
        if i == 0:
            bary_init = bary_fps.copy()
        elif i == 1:
            # Random initialization
            bary_init = np.random.uniform(0, 1, size=(N, 2))
            bary_init = _project_to_bary_simplex(bary_init)
        else:
            # Perturbed FPS
            bary_init = bary_fps + rng.normal(0, 0.05, size=(N, 2))
            bary_init = _project_to_bary_simplex(bary_init)
        
        # Start with a small t and increase
        current_bary = bary_init.copy()
        t_current = 0.01
        
        # Adaptive barrier parameter annealing
        mu = 10.0
        for mu_iter in range(8):
            if time.time() > deadline - 15.0:
                break
            
            # Solve for optimal t given current mu
            x0 = np.concatenate([current_bary.flatten(), [t_current]])
            
            def obj(x):
                return _epigraph_objective(x, t_current, mu)
            
            try:
                result = minimize(
                    obj,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * (2*N) + [(0.0, 1.0)],
                    options={'maxiter': 150, 'ftol': 1e-10}
                )
                
                x_opt = result.x
                current_bary = _project_to_bary_simplex(x_opt[:2*N].reshape(N, 2))
                t_candidate = x_opt[2*N]
                
                # Check feasibility
                if _hard_feasibility_check(current_bary, t_candidate * 0.99):
                    t_current = t_candidate
                else:
                    # Reduce t if infeasible
                    t_current = np.min(np.abs(_dets_from_bary(current_bary))) * 0.95
                
            except:
                break
            
            # Reduce barrier parameter
            mu *= 0.5
        
        # Final local refinement with SLSQP
        current_bary = _local_refine_slsqp(current_bary, t_current)
        points_opt = _bary_to_cart(current_bary)
        score_opt = _min_area_ratio_from_bary(current_bary)
        
        if score_opt > best_score:
            best_score = score_opt
            best_points = points_opt.copy()
    
    # Phase 3: Final Nelder-Mead refinement if time permits
    if best_points is not None and time.time() < deadline - 5.0:
        x_cart = best_points.flatten()
        
        def _objective_cart(x):
            points = x.reshape(-1, 2)
            points = _clip_to_triangle(points)
            bary = _cart_to_bary(points)
            return -_min_area_ratio_from_bary(bary)
        
        try:
            result = minimize(
                _objective_cart,
                x_cart,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            points_opt = _clip_to_triangle(result.x.reshape(N, 2))
            bary_opt = _cart_to_bary(points_opt)
            score_opt = _min_area_ratio_from_bary(bary_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt.copy()
        except:
            pass
    
    if best_points is None:
        best_points = _bary_to_cart(_project_to_bary_simplex(rng.uniform(0, 1, size=(N, 2))))
    
    return _clip_to_triangle(best_points)
