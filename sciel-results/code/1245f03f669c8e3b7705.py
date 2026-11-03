"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the smallest triangle area."""
import time
import numpy as np
from scipy.optimize import minimize, root

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0
H = np.sqrt(3) / 2.0

# Precompute indices for all C(11,3) = 165 triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRIPLES = _IDX_TRIPLES.shape[0]

def _min_triangle_area(points):
    """Compute the smallest area among all triangles formed by triples of points."""
    pi = points[_IDX_TRIPLES[:, 0]]
    pj = points[_IDX_TRIPLES[:, 1]]
    pk = points[_IDX_TRIPLES[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _min_triangle_area_grad(points):
    """Compute the min area and its gradient with respect to points."""
    pi = points[_IDX_TRIPLES[:, 0]]
    pj = points[_IDX_TRIPLES[:, 1]]
    pk = points[_IDX_TRIPLES[:, 2]]
    
    # d1 = pj - pi, d2 = pk - pi
    d1 = pj - pi
    d2 = pk - pi
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_idx = np.argmin(areas)
    min_area = areas[min_idx]
    
    # Gradient of area with respect to each point
    # For a triple (i, j, k), area = 0.5 * |cross(pj-pi, pk-pi)|
    # Gradient w.r.t. pi: -0.5 * (cross derivative w.r.t. pi)
    # Let's compute gradient for the min triple
    i, j, k = _IDX_TRIPLES[min_idx]
    pi_pt = points[i]
    pj_pt = points[j]
    pk_pt = points[k]
    
    d1 = pj_pt - pi_pt
    d2 = pk_pt - pi_pt
    c = d1[0] * d2[1] - d1[1] * d2[0]
    
    # sign of cross
    sgn = 1.0 if c >= 0 else -1.0
    
    grad = np.zeros((N, 2))
    # Area = 0.5 * sgn * (d1[0]*d2[1] - d1[1]*d2[0])
    # d(Area)/d(pi) = 0.5 * sgn * (-d2[1], d2[0])
    # d(Area)/d(pj) = 0.5 * sgn * (d2[1], -d2[0])
    # d(Area)/d(pk) = 0.5 * sgn * (-d1[1], d1[0])
    
    grad[i] = 0.5 * sgn * np.array([-d2[1], d2[0]])
    grad[j] = 0.5 * sgn * np.array([d2[1], -d2[0]])
    grad[k] = 0.5 * sgn * np.array([-d1[1], d1[0]])
    
    return min_area, grad

def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    y = np.clip(y, 0, H)
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)
    
    points[:, 0] = x
    points[:, 1] = y
    return points

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) to cartesian."""
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to valid region."""
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _objective_bary(x):
    """Objective: negative min triangle area. x is flat array of 2N barycentric coords."""
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area(points)

def _make_constraints():
    """Create linear inequality constraints for barycentric coords."""
    constraints = []
    for i in range(N):
        u_idx = 2 * i
        v_idx = 2 * i + 1
        A_u = np.zeros(2 * N)
        A_u[u_idx] = 1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A_u: A @ x})
        A_v = np.zeros(2 * N)
        A_v[v_idx] = 1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A_v: A @ x})
        A_sum = np.zeros(2 * N)
        A_sum[u_idx] = -1.0
        A_sum[v_idx] = -1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A_sum: 1.0 + A @ x})
    return constraints

def _make_initial_bary_structured():
    """A well-structured initial configuration."""
    bary_list = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1.0/3, 1.0/3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.5, 0.25],
    ]
    return np.array(bary_list[:N])

def _make_random_bary(seed):
    """Generate random barycentric coordinates."""
    rng = np.random.RandomState(seed)
    bary_list = []
    for _ in range(1000):
        u = rng.uniform(0.0, 1.0)
        v = rng.uniform(0.0, 1.0)
        if u + v <= 1.0:
            bary_list.append([u, v])
            if len(bary_list) == N:
                break
    while len(bary_list) < N:
        u = rng.uniform(0.1, 0.6)
        v = rng.uniform(0.1, 0.6)
        if u + v < 0.8:
            bary_list.append([u, v])
    return np.array(bary_list[:N])

def _lagrange_refinement(points, max_iter=50, tol=1e-10):
    """
    Refine using a projected gradient / Newton-like step.
    At the optimum, the gradient of the min-area w.r.t. the active constraint
    should be balanced by the constraint gradients. We use a simple iterative
    approach: move points in the direction that increases the min area,
    projected to stay in the triangle.
    """
    points = points.copy()
    current_min = _min_triangle_area(points)
    
    for _ in range(max_iter):
        min_area, grad = _min_triangle_area_grad(points)
        
        # We want to increase min_area, so move in direction of grad
        # But grad is w.r.t. the min triple. Other constraints may be violated.
        # Simple approach: move each point slightly in the direction of its gradient component
        # This is a subgradient method.
        
        # Scale step size
        step_size = 0.01
        new_points = points + step_size * grad
        
        # Project back to triangle
        new_points = _clip_to_triangle(new_points)
        
        # Check if we improved
        new_min = _min_triangle_area(new_points)
        if new_min > current_min:
            points = new_points
            current_min = new_min
            # Decrease step size as we converge
            step_size *= 0.95
        else:
            # Try smaller step
            step_size *= 0.5
            if step_size < tol:
                break
    
    return points, current_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def time_left():
        return deadline - time.time()
    
    constraints = _make_constraints()
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_score = -np.inf
    best_bary = None
    best_points = None
    
    # Generate diverse initializations
    inits = []
    
    bary_struct = _make_initial_bary_structured()
    inits.append(bary_struct)
    
    for seed in range(15):
        if time_left() < 5.0:
            break
        bary_r = _make_random_bary(seed)
        inits.append(bary_r)
    
    rng_pert = np.random.RandomState(42)
    for scale in [0.02, 0.05, 0.10, 0.20]:
        if time_left() < 5.0:
            break
        bary_pert = bary_struct + rng_pert.normal(0, scale, size=bary_struct.shape)
        bary_pert = _clamp_bary(bary_pert)
        inits.append(bary_pert)
    
    # Optimize each initialization with SLSQP
    for i, bary0 in enumerate(inits):
        if time_left() < 15.0:
            break
        
        x0 = bary0.flatten()
        
        try:
            result = minimize(
                _objective_bary,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            bary_opt = _clamp_bary(result.x.reshape(-1, 2))
            points_opt = _bary_to_cart(bary_opt)
            points_opt = _clip_to_triangle(points_opt)
            score_opt = _min_triangle_area(points_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_bary = bary_opt.copy()
                best_points = points_opt.copy()
                
        except Exception:
            pass
    
    # Lagrange-based refinement from the best solution
    if time_left() > 20.0 and best_points is not None:
        # Try multiple refinement scales
        rng_refine = np.random.RandomState(123)
        for scale in [0.01, 0.03, 0.08, 0.02]:
            if time_left() < 15.0:
                break
            
            bary_pert = best_bary + rng_refine.normal(0, scale, size=best_bary.shape)
            bary_pert = _clamp_bary(bary_pert)
            points_pert = _bary_to_cart(bary_pert)
            points_pert = _clip_to_triangle(points_pert)
            
            # First SLSQP refinement
            x0 = bary_pert.flatten()
            try:
                result = minimize(
                    _objective_bary,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 150, 'ftol': 1e-13}
                )
                bary_opt = _clamp_bary(result.x.reshape(-1, 2))
                points_opt = _bary_to_cart(bary_opt)
                points_opt = _clip_to_triangle(points_opt)
            except Exception:
                points_opt = points_pert
            
            # Then Lagrange/subgradient refinement
            points_ref, score_ref = _lagrange_refinement(points_opt, max_iter=30)
            
            if score_ref > best_score:
                best_score = score_ref
                best_points = points_ref.copy()
                # Update best_bary by inverse transforming (approximate)
                # We can also just keep the points
            
            # Also check the SLSQP result before Lagrange
            score_slsqp = _min_triangle_area(points_opt)
            if score_slsqp > best_score:
                best_score = score_slsqp
                best_points = points_opt.copy()
                bary_opt = _clamp_bary(bary_opt)
                best_bary = bary_opt.copy()
    
    # Final polish: one more SLSQP from the best
    if time_left() > 10.0 and best_bary is not None:
        try:
            x0 = best_bary.flatten()
            result = minimize(
                _objective_bary,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-14}
            )
            bary_opt = _clamp_bary(result.x.reshape(-1, 2))
            points_opt = _bary_to_cart(bary_opt)
            points_opt = _clip_to_triangle(points_opt)
            score_opt = _min_triangle_area(points_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt.copy()
                best_bary = bary_opt.copy()
        except Exception:
            pass
    
    if best_points is None:
        best_points = _bary_to_cart(_make_initial_bary_structured())
        best_points = _clip_to_triangle(best_points)
    
    return best_points
