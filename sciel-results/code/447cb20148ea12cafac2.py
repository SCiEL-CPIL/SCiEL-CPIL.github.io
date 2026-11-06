import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 130

def _thomson_init(n, iters=5000, lr_start=0.01, lr_end=0.0001):
    """Initialize n points on unit sphere via Thomson (electrostatic) relaxation."""
    # Start with Fibonacci spiral as initial guess
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Simulated annealing / gradient descent to minimize Coulomb energy
    for it in range(iters):
        # Compute pairwise forces
        # Force on point i from point j: k * (p_i - p_j) / |p_i - p_j|^3
        # This is O(n^2) but n=130 so it's fine
        
        # Vectorized computation
        # diff[i,j] = p_i - p_j
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
        dist = np.linalg.norm(diff, axis=2)  # (n, n)
        dist[dist < 1e-10] = 1e-10  # avoid division by zero
        
        # Force: sum_j (p_i - p_j) / |p_i - p_j|^3
        # We need to compute this for all i
        inv_dist3 = 1.0 / (dist ** 3)
        inv_dist3[np.arange(n)[:, None] == np.arange(n)[None, :]] = 0.0  # no self-force
        
        forces = np.sum(diff * inv_dist3[:, :, np.newaxis], axis=1)  # (n, 3)
        
        # Learning rate schedule
        t = it / iters
        lr = lr_start * (lr_end / lr_start) ** t
        
        # Update and normalize
        pts = pts + lr * forces
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        pts = pts / norms
    
    return pts

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return pts / norms

def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_facets(points):
    """Compute exact covering radius using convex hull facets."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: grid-based estimate
        return _covering_radius_grid(points)
    
    max_cr = 0.0
    for simplex in hull.simplices:
        verts = points[simplex]
        if len(verts) < 3:
            continue
        
        # For a facet, find the point on the sphere farthest from all vertices
        # within the spherical polygon defined by the facet
        # This is the circumradius of the spherical polygon
        
        # For a triangle: compute circumcenter of the planar triangle,
        # then find the corresponding point on the sphere
        if len(verts) == 3:
            v1, v2, v3 = verts
            e1 = v2 - v1
            e2 = v3 - v1
            A = np.array([[np.dot(e1, e1), np.dot(e2, e1)],
                           [np.dot(e1, e2), np.dot(e2, e2)]])
            b = np.array([np.dot(e1, e1) / 2, np.dot(e2, e2) / 2])
            
            try:
                alpha, beta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            
            pc = v1 + alpha * e1 + beta * e2
            pc_norm = np.linalg.norm(pc)
            if pc_norm < 1e-12:
                continue
            
            # Check if circumcenter is inside the triangle (barycentric coords)
            # alpha and beta should be in [0,1] and alpha+beta <= 1 for inside
            # For convex hull facets, it should be inside
            cos_R = np.dot(v1, pc) / pc_norm
            cos_R = np.clip(cos_R, -1.0, 1.0)
            R = np.arccos(cos_R)
            max_cr = max(max_cr, R)
        else:
            # For higher-dimensional facets, use the first 3 vertices as approximation
            v1, v2, v3 = verts[0], verts[1], verts[2]
            e1 = v2 - v1
            e2 = v3 - v1
            A = np.array([[np.dot(e1, e1), np.dot(e2, e1)],
                           [np.dot(e1, e2), np.dot(e2, e2)]])
            b = np.array([np.dot(e1, e1) / 2, np.dot(e2, e2) / 2])
            
            try:
                alpha, beta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            
            pc = v1 + alpha * e1 + beta * e2
            pc_norm = np.linalg.norm(pc)
            if pc_norm < 1e-12:
                continue
            
            cos_R = np.dot(v1, pc) / pc_norm
            cos_R = np.clip(cos_R, -1.0, 1.0)
            R = np.arccos(cos_R)
            max_cr = max(max_cr, R)
    
    return np.degrees(max_cr)

def _covering_radius_grid(points, n_theta=180, n_phi=360):
    """Fallback: estimate covering radius using a grid on the sphere."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    X = np.sin(TH) * np.cos(PH)
    Y = np.sin(TH) * np.sin(PH)
    Z = np.cos(TH)
    
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # For each grid point, find nearest configuration point
    dots = grid_points @ points.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # (M, N)
    min_dists = np.min(dists, axis=1)  # (M,)
    max_dist = np.max(min_dists)
    
    return np.degrees(max_dist)

def _soft_soft_objective_and_grad(flat_pts, test_pts, beta, gamma):
    """
    Computes the soft-soft covering radius and its gradient.
    """
    pts = _normalize(flat_pts.reshape(N, 3))
    
    dots = test_pts @ pts.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # (M, N)
    
    # Inner soft-min
    max_d = np.max(dists, axis=1, keepdims=True)
    exp_term = np.exp(-gamma * (dists - max_d))
    S = np.sum(exp_term, axis=1, keepdims=True)
    m_gamma = -1.0 / gamma * (np.log(S) - gamma * max_d)
    m_gamma = m_gamma.flatten()
    
    w = exp_term / S  # (M, N)
    
    # Outer soft-max
    max_m = np.max(m_gamma)
    exp_m = np.exp(beta * (m_gamma - max_m))
    T = np.sum(exp_m)
    J = max_m + np.log(T) / beta
    
    v = exp_m / T  # (M,)
    
    # Gradient
    denom = np.sqrt(np.maximum(1e-10, 1.0 - dots * dots))
    
    y_b = test_pts[:, np.newaxis, :]  # (M, 1, 3)
    p_b = pts[np.newaxis, :, :]  # (1, N, 3)
    dot_b = dots[:, :, np.newaxis]  # (M, N, 1)
    
    diff = y_b - dot_b * p_b  # (M, N, 3)
    coeff = (v[:, np.newaxis] * w / denom[:, :, np.newaxis])  # (M, N, 1)
    grad_contrib = coeff * diff  # (M, N, 3)
    grad_pts = np.sum(grad_contrib, axis=0)  # (N, 3)
    
    grad_flat = grad_pts.flatten()
    
    return J, grad_flat

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Phase 0: Thomson initialization
    pts = _thomson_init(N, iters=3000, lr_start=0.01, lr_end=0.0001)
    best_pts = pts.copy()
    
    # Compute initial score
    best_score = _covering_radius_facets(best_pts)
    
    if time.time() > deadline - 60:
        return best_pts
    
    # Test points for soft objective
    n_test = 15000
    test_points = _fibonacci_spiral(n_test)
    
    # Phase 1: Coarse optimization with soft-soft objective
    beta = 5.0
    gamma = 5.0
    
    def objective_coarse(flat_pts):
        val, _ = _soft_soft_objective_and_grad(flat_pts, test_points, beta, gamma)
        return val
        
    def grad_coarse(flat_pts):
        _, g = _soft_soft_objective_and_grad(flat_pts, test_points, beta, gamma)
        return g
        
    try:
        res = minimize(
            objective_coarse,
            grad_coarse,
            pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 80, 'gtol': 1e-5, 'maxfun': 150}
        )
        candidate = _normalize(res.x.reshape(N, 3))
        candidate_score = _covering_radius_facets(candidate)
        if candidate_score < best_score:
            best_score = candidate_score
            best_pts = candidate
    except Exception:
        pass
    
    if time.time() > deadline - 40:
        return best_pts
    
    # Phase 2: Fine optimization
    beta = 20.0
    gamma = 20.0
    
    def objective_fine(flat_pts):
        val, _ = _soft_soft_objective_and_grad(flat_pts, test_points, beta, gamma)
        return val
        
    def grad_fine(flat_pts):
        _, g = _soft_soft_objective_and_grad(flat_pts, test_points, beta, gamma)
        return g

    try:
        res2 = minimize(
            objective_fine,
            grad_fine,
            best_pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 80, 'gtol': 1e-6, 'maxfun': 150}
        )
        candidate = _normalize(res2.x.reshape(N, 3))
        candidate_score = _covering_radius_facets(candidate)
        if candidate_score < best_score:
            best_score = candidate_score
            best_pts = candidate
    except Exception:
        pass
    
    if time.time() > deadline - 20:
        return best_pts
    
    # Phase 3: Very fine polish
    beta = 50.0
    gamma = 50.0
    
    def objective_vf(flat_pts):
        val, _ = _soft_soft_objective_and_grad(flat_pts, test_points, beta, gamma)
        return val
        
    def grad_vf(flat_pts):
        _, g = _soft_soft_objective_and_grad(flat_pts, test_points, beta, gamma)
        return g

    try:
        res3 = minimize(
            objective_vf,
            grad_vf,
            best_pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 60, 'gtol': 1e-7, 'maxfun': 100}
        )
        candidate = _normalize(res3.x.reshape(N, 3))
        candidate_score = _covering_radius_facets(candidate)
        if candidate_score < best_score:
            best_score = candidate_score
            best_pts = candidate
    except Exception:
        pass
    
    # Phase 4: Local facet-based refinement
    # For the worst facet, try to move the opposite vertex slightly
    for _ in range(10):
        if time.time() > deadline - 5:
            break
            
        points = _normalize(best_pts)
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Find the facet with largest circumradius
        worst_simplex = None
        worst_R = 0.0
        for simplex in hull.simplices:
            verts = points[simplex]
            if len(verts) == 3:
                v1, v2, v3 = verts
                e1 = v2 - v1
                e2 = v3 - v1
                A = np.array([[np.dot(e1, e1), np.dot(e2, e1)],
                               [np.dot(e1, e2), np.dot(e2, e2)]])
                b = np.array([np.dot(e1, e1) / 2, np.dot(e2, e2) / 2])
                try:
                    alpha, beta_c = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue
                pc = v1 + alpha * e1 + beta_c * e2
                pc_norm = np.linalg.norm(pc)
                if pc_norm < 1e-12:
                    continue
                cos_R = np.dot(v1, pc) / pc_norm
                cos_R = np.clip(cos_R, -1.0, 1.0)
                R = np.arccos(cos_R)
                if R > worst_R:
                    worst_R = R
                    worst_simplex = simplex
        
        if worst_simplex is None:
            break
        
        # Try small perturbations of points not in the worst facet
        improved = False
        for idx in range(N):
            if idx in worst_simplex:
                continue
            
            for step in [0.001, 0.005, 0.01]:
                trial = best_pts.copy()
                # Move point idx slightly toward the circumcenter of the worst facet
                verts = points[worst_simplex]
                v1, v2, v3 = verts
                e1 = v2 - v1
                e2 = v3 - v1
                A = np.array([[np.dot(e1, e1), np.dot(e2, e1)],
                               [np.dot(e1, e2), np.dot(e2, e2)]])
                b = np.array([np.dot(e1, e1) / 2, np.dot(e2, e2) / 2])
                try:
                    alpha, beta_c = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue
                pc = v1 + alpha * e1 + beta_c * e2
                pc_norm = np.linalg.norm(pc)
                if pc_norm < 1e-12:
                    continue
                pc_dir = pc / pc_norm
                
                # Move point idx toward pc_dir
                trial[idx] = trial[idx] + step * (pc_dir - trial[idx])
                trial = _normalize(trial)
                
                trial_score = _covering_radius_facets(trial)
                if trial_score < best_score:
                    best_score = trial_score
                    best_pts = trial
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return _normalize(best_pts)
