import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 130

def _thomson_init(n, iters=5000, lr_start=0.01, lr_end=0.0001):
    """Initialize n points on unit sphere via Thomson (electrostatic) relaxation."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    for it in range(iters):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist[dist < 1e-10] = 1e-10
        
        inv_dist3 = 1.0 / (dist ** 3)
        inv_dist3[np.arange(n)[:, None] == np.arange(n)[None, :]] = 0.0
        
        forces = np.sum(diff * inv_dist3[:, :, np.newaxis], axis=1)
        
        t = it / iters
        lr = lr_start * (lr_end / lr_start) ** t
        
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
        return _covering_radius_grid(points)
    
    max_cr = 0.0
    for simplex in hull.simplices:
        verts = points[simplex]
        if len(verts) < 3:
            continue
        
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
            
            cos_R = np.dot(v1, pc) / pc_norm
            cos_R = np.clip(cos_R, -1.0, 1.0)
            R = np.arccos(cos_R)
            max_cr = max(max_cr, R)
        else:
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
    
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    max_dist = np.max(min_dists)
    
    return np.degrees(max_dist)

def _riemannian_gradient(pts, test_pts, beta, gamma):
    """Compute Riemannian gradient of soft-soft covering radius on the sphere."""
    dots = test_pts @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    max_d = np.max(dists, axis=1, keepdims=True)
    exp_term = np.exp(-gamma * (dists - max_d))
    S = np.sum(exp_term, axis=1, keepdims=True)
    m_gamma = (-1.0 / gamma * (np.log(S) - gamma * max_d)).flatten()
    
    w = exp_term / S
    
    max_m = np.max(m_gamma)
    exp_m = np.exp(beta * (m_gamma - max_m))
    T = np.sum(exp_m)
    v = exp_m / T
    
    denom = np.sqrt(np.maximum(1e-10, 1.0 - dots * dots))
    
    y_b = test_pts[:, np.newaxis, :]
    p_b = pts[np.newaxis, :, :]
    dot_b = dots[:, :, np.newaxis]
    
    diff = y_b - dot_b * p_b
    coeff = (v[:, np.newaxis] * w / denom[:, :, np.newaxis])
    grad_contrib = coeff * diff
    grad_pts = np.sum(grad_contrib, axis=0)
    
    # Riemannian gradient: project to tangent space
    # Tangent space at pts[i] is orthogonal to pts[i]
    # grad_tangent = grad - (grad . pts) * pts
    grad_tangent = grad_pts - np.sum(grad_pts * pts, axis=1, keepdims=True) * pts
    
    return grad_tangent

def _riemannian_l_bfgs_b(pts, test_pts, beta, gamma, maxiter=100, m=10, tol=1e-6):
    """
    Riemannian L-BFGS-B optimization on the sphere.
    Uses retraction (normalization) to stay on the manifold.
    """
    pts = _normalize(pts)
    n = pts.shape[0]
    flat_dim = n * 3
    
    # L-BFGS memory
    S_list = []  # step directions
    Y_list = []  # gradient differences
    W_list = []  # inverse Hessian approx coefficients
    c_list = []  # rho coefficients
    
    g = _riemannian_gradient(pts, test_pts, beta, gamma)
    
    for it in range(maxiter):
        # Check convergence
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Line search with Riemannian retraction
        alpha = 1.0
        f0 = _soft_soft_value(pts, test_pts, beta, gamma)
        
        # Backtracking line search
        for ls_iter in range(30):
            # Retraction: move in direction -alpha * g, then normalize
            new_pts = pts - alpha * g
            new_pts = _normalize(new_pts)
            
            f1 = _soft_soft_value(new_pts, test_pts, beta, gamma)
            
            # Armijo condition
            if f1 <= f0 - 0.01 * alpha * np.sum(g * g):
                break
            alpha *= 0.5
        
        # Update L-BFGS memory
        g_new = _riemannian_gradient(new_pts, test_pts, beta, gamma)
        S = (new_pts - pts).flatten()
        Y = (g_new - g).flatten()
        
        S_list.append(S)
        Y_list.append(Y)
        
        rho = 1.0 / (np.dot(Y, S) + 1e-12)
        c_list.append(rho)
        
        # Keep only last m pairs
        if len(S_list) > m:
            S_list.pop(0)
            Y_list.pop(0)
            c_list.pop(0)
        
        pts = new_pts
        g = g_new
    
    return _normalize(pts)

def _soft_soft_value(pts, test_pts, beta, gamma):
    """Compute soft-soft covering radius value."""
    dots = test_pts @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    max_d = np.max(dists, axis=1, keepdims=True)
    exp_term = np.exp(-gamma * (dists - max_d))
    S = np.sum(exp_term, axis=1, keepdims=True)
    m_gamma = (-1.0 / gamma * (np.log(S) - gamma * max_d)).flatten()
    
    max_m = np.max(m_gamma)
    exp_m = np.exp(beta * (m_gamma - max_m))
    T = np.sum(exp_m)
    J = max_m + np.log(T) / beta
    
    return J

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Phase 0: Thomson initialization
    pts = _thomson_init(N, iters=3000, lr_start=0.01, lr_end=0.0001)
    best_pts = pts.copy()
    best_score = _covering_radius_facets(best_pts)
    
    if time.time() > deadline - 60:
        return best_pts
    
    # Test points for soft objective
    n_test = 15000
    test_points = _fibonacci_spiral(n_test)
    
    # Phase 1: Riemannian L-BFGS-B with moderate beta/gamma
    beta = 5.0
    gamma = 5.0
    
    try:
        candidate = _riemannian_l_bfgs_b(pts, test_points, beta, gamma, maxiter=80, m=10, tol=1e-5)
        candidate_score = _covering_radius_facets(candidate)
        if candidate_score < best_score:
            best_score = candidate_score
            best_pts = candidate
    except Exception:
        pass
    
    if time.time() > deadline - 40:
        return best_pts
    
    # Phase 2: Fine optimization with higher beta/gamma
    beta = 20.0
    gamma = 20.0
    
    try:
        candidate = _riemannian_l_bfgs_b(best_pts, test_points, beta, gamma, maxiter=80, m=10, tol=1e-6)
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
    
    try:
        candidate = _riemannian_l_bfgs_b(best_pts, test_points, beta, gamma, maxiter=60, m=10, tol=1e-7)
        candidate_score = _covering_radius_facets(candidate)
        if candidate_score < best_score:
            best_score = candidate_score
            best_pts = candidate
    except Exception:
        pass
    
    # Phase 4: Local facet-based refinement
    for _ in range(10):
        if time.time() > deadline - 5:
            break
            
        points = _normalize(best_pts)
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
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
        
        improved = False
        for idx in range(N):
            if idx in worst_simplex:
                continue
            
            for step in [0.001, 0.005, 0.01]:
                trial = best_pts.copy()
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
