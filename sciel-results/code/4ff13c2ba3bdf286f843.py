# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

def _precompute_indices(n):
    """Precompute triangle indices for efficiency."""
    return np.array(list(itertools.combinations(range(n), 3)))

def _compute_dets(points, idx):
    """Compute 2 * signed area (determinant) for all triangles."""
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return det

def _compute_hull_area(points):
    """Compute convex hull area safely."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
        
    A_hull = _compute_hull_area(points)
    if A_hull <= 1e-12:
        return 0.0
        
    idx = _precompute_indices(n)
    dets = _compute_dets(points, idx)
    A_min = 0.5 * np.min(np.abs(dets))
    
    return A_min / A_hull

def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    idx = _precompute_indices(n)
    
    # Initializations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Concentric polygons (1 center, 5 inner, 7 outer)
    center = np.array([[0, 0]])
    angles_inner = np.linspace(0, 2*np.pi, 5, endpoint=False)
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    inner_pts = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    outer_pts = np.column_stack([1.0 * np.cos(angles_outer), 1.0 * np.sin(angles_outer)])
    inits.append(np.vstack([center, inner_pts, outer_pts]))
    
    # 3. Random
    rng = np.random.default_rng(seed=42)
    inits.append(rng.random((n, 2)) * 2 - 1)
    
    # 4. Grid
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts = np.column_stack([gx.ravel(), gy.ravel()])[:13]
    pts += rng.random((13, 2)) * 0.05
    inits.append(pts)
    
    # 5. Hexagonal lattice snippet
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    # Normalize to fit in [-1, 1]
    hex_pts = (hex_pts - hex_pts.mean(axis=0)) / (np.abs(hex_pts).max() + 1e-9)
    # Take 13 points
    hex_pts = hex_pts[:13]
    inits.append(hex_pts)
    
    # 6. Fibonacci spiral
    angles_fib = np.arange(n) * 2 * np.pi * 0.6180339887
    radii_fib = np.sqrt(np.arange(1, n + 1) / n)
    inits.append(np.column_stack([radii_fib * np.cos(angles_fib), radii_fib * np.sin(angles_fib)]))
    
    for i_init, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
            
        x0 = init_pts.flatten()
        
        def soft_min_objective(x, beta=50.0):
            """
            Objective: Minimize soft_max( |det| / A_hull ) + penalty * A_hull
            This pushes all triangles to have similar, large areas relative to the hull.
            We minimize the maximum area to force the minimum area up? 
            Wait. We want to MAXIMIZE min_area / hull_area.
            This is equivalent to MINIMIZING hull_area / min_area.
            Or MINIMIZING max( hull_area / area_i ) over all triangles.
            Let z_i = hull_area / (0.5 * |det_i|). We want to minimize max(z_i).
            Soft max of z_i is a smooth upper bound.
            
            z_i = A_hull / a_i.
            log(z_i) = log(A_hull) - log(a_i).
            soft_max(z) = exp( sum(exp(z_i)) / N ).
            log(soft_max(z)) = log_sum_exp(z_i).
            
            So we minimize: log_sum_exp( log(A_hull) - log(a_i) )
            
            Gradient:
            Let L = log_sum_exp(y_i) where y_i = log(A_hull) - log(a_i).
            dL/dy_i = softmax(y_i)_i.
            dy_i/dA_hull = 1/A_hull.
            dy_i/da_i = -1/a_i.
            
            da_i/dx = 0.5 * sign(det_i) * d(det_i)/dx.
            
            This looks complicated but manageable.
            """
            points = x.reshape((n, 2))
            if not np.all(np.isfinite(points)):
                return 1e6, np.zeros_like(x)
                
            A_hull = _compute_hull_area(points)
            if A_hull <= 1e-12:
                return 1e6, np.zeros_like(x)
                
            dets = _compute_dets(points, idx)
            abs_dets = np.abs(dets)
            # Avoid log(0)
            abs_dets = np.maximum(abs_dets, 1e-12)
            
            a = 0.5 * abs_dets
            y = np.log(A_hull) - np.log(a)
            
            # log_sum_exp(y)
            m = np.max(y)
            L = m + np.log(np.sum(np.exp(y - m)))
            
            # Gradient
            # p = softmax(y)
            p = np.exp(y - m)
            p /= np.sum(p)
            
            # dL/dy_i = p_i
            # dy_i/dA_hull = 1/A_hull
            # dy_i/da_i = -1/a_i
            
            # dA_hull/dx: subgradient from hull edges
            # da_i/dx: from triangle geometry
            
            grad_x = np.zeros_like(x)
            
            # Contribution from A_hull
            # dL/dA_hull = sum_i (p_i * (1/A_hull)) = (1/A_hull) * sum(p_i) = 1/A_hull
            # So grad_x += (1/A_hull) * grad_A_hull
            
            # Contribution from a_i
            # dL/da_i = p_i * (-1/a_i)
            # dL/dx = sum_i (dL/da_i * da_i/dx)
            
            # da_i/dx_j (for point j in triangle i)
            # a_i = 0.5 * |det_i|
            # da_i/dx = 0.5 * sign(det_i) * d(det_i)/dx
            
            # We can vectorize the triangle gradient part.
            # d(det_i)/dx_k is non-zero only if k is in triangle i.
            # Let's accumulate gradients.
            
            # grad_det_i is a vector of shape (3, 2) for the 3 points in triangle i.
            # We need to sum over all i: p_i * (-1/a_i) * 0.5 * sign(det_i) * grad_det_i
            
            # To avoid a loop over 286 triangles (which is fine, 286 is small), 
            # or we can do it vectorized.
            
            # Let's do it semi-vectorized for speed and clarity.
            # p_i is scalar weight.
            # w_i = -p_i / (2 * a_i) * sign(det_i)
            
            signs = np.sign(dets)
            weights = -p / (2 * a) * signs
            
            # Now we need to add weights[i] * grad_det_i to the global gradient.
            # grad_det_i for points p1, p2, p3:
            # d(det)/dp1 = [y2-y3, x3-x2]
            # d(det)/dp2 = [y3-y1, x1-x3]
            # d(det)/dp3 = [y1-y2, x2-x1]
            
            p1 = points[idx[:, 0]]
            p2 = points[idx[:, 1]]
            p3 = points[idx[:, 2]]
            
            g1 = np.column_stack([p2[:, 1] - p3[:, 1], p3[:, 0] - p2[:, 0]])
            g2 = np.column_stack([p3[:, 1] - p1[:, 1], p1[:, 0] - p3[:, 0]])
            g3 = np.column_stack([p1[:, 1] - p2[:, 1], p2[:, 0] - p1[:, 0]])
            
            # Weighted sum
            # We need to scatter-add these to the global gradient array.
            # grad_x[2*idx[:,0] : 2*idx[:,0]+2] += weights[:, None] * g1
            # This is tricky with numpy. Let's use a loop or np.add.at
            
            # np.add.at is efficient enough for 286 items.
            idx_flat = idx.flatten()
            # We have 3 vectors per triangle.
            # Let's create a combined array of shape (286*3, 2) and indices of shape (286*3,)
            
            # Actually, simpler:
            # grad_x[2*idx[:,0]] += weights[:, 0] * g1[:, 0]
            # grad_x[2*idx[:,0]+1] += weights[:, 0] * g1[:, 1]
            # ... and so on.
            
            np.add.at(grad_x, 2*idx[:, 0], weights * g1[:, 0])
            np.add.at(grad_x, 2*idx[:, 0]+1, weights * g1[:, 1])
            np.add.at(grad_x, 2*idx[:, 1], weights * g2[:, 0])
            np.add.at(grad_x, 2*idx[:, 1]+1, weights * g2[:, 1])
            np.add.at(grad_x, 2*idx[:, 2], weights * g3[:, 0])
            np.add.at(grad_x, 2*idx[:, 2]+1, weights * g3[:, 1])
            
            # Now add hull gradient
            # dL/dA_hull = 1/A_hull
            # grad_A_hull
            try:
                hull = ConvexHull(points)
                hull_vertices = hull.vertices
                hull_grad = np.zeros_like(points)
                
                def r90(v):
                    return np.array([-v[1], v[0]])
                    
                for idx_h, v in enumerate(hull_vertices):
                    prev_v = hull_vertices[(idx_h - 1) % len(hull_vertices)]
                    next_v = hull_vertices[(idx_h + 1) % len(hull_vertices)]
                    hull_grad[v] = 0.5 * (r90(points[prev_v]) - r90(points[next_v]))
                    
                grad_x += (1.0 / A_hull) * hull_grad.flatten()
                
            except Exception:
                pass
            
            return L, grad_x

        # Run optimization with multiple betas (annealing)
        betas = [10, 50, 200, 1000]
        current_x = x0.copy()
        
        for beta in betas:
            if time.time() > deadline:
                break
            
            # Note: The objective function defined above doesn't use beta directly in the log-sum-exp 
            # if we just do log_sum_exp(y). To anneal, we should scale y by beta?
            # log_sum_exp(beta * y) approximates beta * max(y).
            # So we modify the objective to use beta.
            
            def objective_beta(x, beta_val=beta):
                points = x.reshape((n, 2))
                if not np.all(np.isfinite(points)):
                    return 1e6, np.zeros_like(x)
                    
                A_hull = _compute_hull_area(points)
                if A_hull <= 1e-12:
                    return 1e6, np.zeros_like(x)
                    
                dets = _compute_dets(points, idx)
                abs_dets = np.maximum(np.abs(dets), 1e-12)
                a = 0.5 * abs_dets
                y = np.log(A_hull) - np.log(a)
                y_scaled = beta_val * y
                
                m = np.max(y_scaled)
                L = m + np.log(np.sum(np.exp(y_scaled - m)))
                
                p = np.exp(y_scaled - m)
                p /= np.sum(p)
                
                grad_x = np.zeros_like(x)
                
                signs = np.sign(dets)
                # dL/da_i = beta * p_i * (-1/a_i)
                # dL/dx from a_i: sum_i (beta * p_i * (-1/a_i) * 0.5 * sign(det_i) * grad_det_i)
                weights = -beta_val * p / (2 * a) * signs
                
                p1 = points[idx[:, 0]]
                p2 = points[idx[:, 1]]
                p3 = points[idx[:, 2]]
                
                g1 = np.column_stack([p2[:, 1] - p3[:, 1], p3[:, 0] - p2[:, 0]])
                g2 = np.column_stack([p3[:, 1] - p1[:, 1], p1[:, 0] - p3[:, 0]])
                g3 = np.column_stack([p1[:, 1] - p2[:, 1], p2[:, 0] - p1[:, 0]])
                
                np.add.at(grad_x, 2*idx[:, 0], weights * g1[:, 0])
                np.add.at(grad_x, 2*idx[:, 0]+1, weights * g1[:, 1])
                np.add.at(grad_x, 2*idx[:, 1], weights * g2[:, 0])
                np.add.at(grad_x, 2*idx[:, 1]+1, weights * g2[:, 1])
                np.add.at(grad_x, 2*idx[:, 2], weights * g3[:, 0])
                np.add.at(grad_x, 2*idx[:, 2]+1, weights * g3[:, 1])
                
                # Hull gradient
                # dL/dA_hull = beta * sum(p_i * 1/A_hull) = beta / A_hull
                try:
                    hull = ConvexHull(points)
                    hull_vertices = hull.vertices
                    hull_grad = np.zeros_like(points)
                    
                    def r90(v):
                        return np.array([-v[1], v[0]])
                        
                    for idx_h, v in enumerate(hull_vertices):
                        prev_v = hull_vertices[(idx_h - 1) % len(hull_vertices)]
                        next_v = hull_vertices[(idx_h + 1) % len(hull_vertices)]
                        hull_grad[v] = 0.5 * (r90(points[prev_v]) - r90(points[next_v]))
                        
                    grad_x += (beta_val / A_hull) * hull_grad.flatten()
                    
                except Exception:
                    pass
                
                return L, grad_x

            try:
                res = minimize(objective_beta, current_x, method='L-BFGS-B', jac=True,
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6})
                current_x = res.x
                final_pts = current_x.reshape((n, 2))
                final_score = compute_score(final_pts)
                
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_pts.copy()
                    
            except Exception:
                pass

    # Final polish with Nelder-Mead on the best candidate
    if best_points is not None and time.time() < deadline - 50:
        def nm_objective(x):
            pts = x.reshape((n, 2))
            s = compute_score(pts)
            if s == 0.0:
                return 1.0
            return -s
        
        try:
            res_nm = minimize(nm_objective, best_points.flatten(), method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            final_pts_nm = res_nm.x.reshape((n, 2))
            final_score_nm = compute_score(final_pts_nm)
            if final_score_nm > best_score:
                best_score = final_score_nm
                best_points = final_pts_nm.copy()
        except Exception:
            pass

    # Fallback
    if best_points is None or best_score < 0.01:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
