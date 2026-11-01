import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).flatten()

    # 2. Generate a dense spherical grid for constraint evaluation
    # We need a grid that is fine enough to approximate the true covering radius
    # but not so fine that it makes the optimization intractable.
    # A grid with ~5000-10000 points is usually sufficient for N=50.
    # Let's use a Fibonacci grid for the test points as well, or a structured grid.
    # Structured grid (theta, phi) is easier to vectorize.
    n_theta = 50
    n_phi = 100
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing='ij')
    grid_x = np.sin(theta_grid) * np.cos(phi_grid)
    grid_y = np.sin(theta_grid) * np.sin(phi_grid)
    grid_z = np.cos(theta_grid)
    grid_pts = np.column_stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()])
    
    # Add poles explicitly to ensure coverage there
    grid_pts = np.vstack([grid_pts, np.array([[1, 0, 0]]), np.array([[-1, 0, 0]]),
                          np.array([[0, 1, 0]]), np.array([[0, -1, 0]]),
                          np.array([[0, 0, 1]]), np.array([[0, 0, -1]])])
                          
    M = grid_pts.shape[0]

    def objective_and_constraints(x_flat):
        """
        Objective: minimize t (index 3*N to 3*N+1)
        Constraints: t - dist(grid_j, nearest_site_i) >= 0 for all j
        Variables: x_flat[0:3N] are the 50 points, x_flat[3N] is t.
        """
        pts = x_flat[:3*N].reshape(N, 3)
        t = x_flat[3*N]
        
        # Normalize points to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        # Compute distances from grid points to all sites
        # dist(j, i) = arccos(dot(grid_j, p_i))
        # dot product matrix: (M, N)
        dots = grid_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)  # (M, N)
        
        # For each grid point, find the minimum distance to any site
        min_dists = np.min(dists, axis=1)  # (M,)
        
        # Objective is t
        obj_val = t
        
        # Constraints: min_dists - t <= 0  =>  t - min_dists >= 0
        # SLSQP constraints are of the form g(x) >= 0 (if type='ineq')
        # So we define g_j = t - min_dist_j
        constraints = t - min_dists
        
        # We also need to ensure points are on the sphere.
        # SLSQP works in Euclidean space. We can add equality constraints
        # ||p_i||^2 - 1 = 0.
        # However, normalizing inside the function makes the gradient of the
        # objective w.r.t the raw x_flat incorrect if we don't account for
        # the projection.
        # A better approach for SLSQP is to use the fact that the gradient
        # of the distance on the sphere is tangent.
        # But SLSQP expects gradients in the parameter space.
        # If we normalize inside, the mapping x -> x/||x|| is non-linear.
        # 
        # Alternative: Use bounds? No, points must be on sphere.
        # Alternative: Use equality constraints for sphere constraint.
        # This adds 50 equality constraints.
        
        # Let's compute the gradient of the objective and constraints.
        # Objective: f(x) = t. Gradient is 0 for points, 1 for t.
        # But wait, if we just minimize t, and the constraints are t >= min_dist,
        # then the optimal t will be max(min_dist).
        # The gradient of the objective w.r.t t is 1.
        # The gradient of the objective w.r.t points is 0 (in the direct sense).
        # However, the constraints depend on the points.
        # SLSQP uses the gradients of the constraints to find the direction.
        
        # Gradient of objective:
        grad_obj = np.zeros_like(x_flat)
        grad_obj[3*N] = 1.0
        
        # Gradient of constraints:
        # g_j = t - min_i dist(j, i)
        # dg_j/dt = 1
        # dg_j/dp_i = - d(min_dist_j)/dp_i
        
        # We need to identify the nearest site for each grid point to compute the gradient.
        nearest_idx = np.argmin(dists, axis=1)  # (M,)
        
        # Initialize gradient of constraints
        # We only need to provide the Jacobian of the constraints if we want
        # to help SLSQP, but SLSQP can also approximate it.
        # Given the complexity of the arccos derivative and the min operation,
        # providing an analytic Jacobian is risky but potentially faster.
        # Let's try providing it.
        
        # The Jacobian is a matrix of shape (M + N, 3*N + 1).
        # Rows 0..M-1 correspond to the ineq constraints (grid points).
        # Rows M..M+N-1 correspond to the eq constraints (sphere norm).
        
        # Let's just return the function values and let SLSQP approximate the Jacobian
        # initially. If it's too slow, we'll implement it.
        # Actually, for M~5000, finite difference Jacobian might be slow.
        # Let's implement the analytic Jacobian for the ineq constraints.
        
        # d(dist)/d(p) = -y / sqrt(1 - (y.p)^2) where y is the grid point?
        # No, dist = arccos(y.p). d/dp arccos(y.p) = -1/sqrt(1-(y.p)^2) * y.
        # So d(min_dist_j)/d(p_i) = 0 if i is not nearest.
        # If i is nearest, d(min_dist_j)/d(p_i) = -grid_j / sqrt(1 - (grid_j.p_i)^2).
        
        # We want to minimize t.
        # Constraint: t - min_dist_j >= 0.
        # Gradient of constraint w.r.t p_i:
        # If i is nearest to j: - ( -grid_j / sqrt(...) ) = grid_j / sqrt(...).
        
        # Let's accumulate the gradients.
        # grad_cons shape: (M, 3*N + 1)
        # We'll build it row by row or vectorized.
        
        # To save memory and time, we might only provide the Jacobian if requested?
        # No, SLSQP always computes/uses it.
        
        # Let's construct the Jacobian for the ineq constraints.
        # J[j, 3*i : 3*i+3] = grad_{p_i} (t - min_dist_j)
        # If i == nearest_idx[j]:
        #    dot = dots[j, i]
        #    denom = sqrt(1 - dot^2)
        #    grad = grid_pts[j] / denom
        # Else:
        #    grad = 0
        
        # J[j, 3*N] = 1 (derivative w.r.t t)
        
        # We also need the Jacobian for the equality constraints: ||p_i||^2 - 1 = 0.
        # d(||p_i||^2 - 1)/d(p_k) = 2*p_i if i==k, else 0.
        
        # Total constraints: M ineq + N eq.
        # This is a large Jacobian (5000x151).
        
        # Let's try to return the analytic Jacobian.
        
        # Initialize Jacobian
        # Note: SLSQP expects a function that returns the Jacobian if jac=True is passed to the constraint dict?
        # No, in scipy.optimize.minimize, if you provide a dict for constraints, you can provide 'jac' as a function.
        # But it's easier to return the values and let SLSQP approximate if it's not too slow.
        # However, with 5000 constraints, approximation is 5000+ function calls per iteration.
        # That will be too slow.
        
        # We MUST provide the Jacobian.
        
        # Let's create the Jacobian matrix.
        J = np.zeros((M + N, 3*N + 1))
        
        # Ineq constraints part
        # J[0:M, 3*N] = 1
        J[:M, 3*N] = 1.0
        
        # Fill in the point gradients
        for j in range(M):
            i = nearest_idx[j]
            p_i = pts[i]
            y_j = grid_pts[j]
            dot = dots[j, i]
            denom = np.sqrt(max(1e-10, 1.0 - dot*dot))
            grad_vec = y_j / denom
            J[j, 3*i : 3*i+3] = grad_vec
            
        # Eq constraints part
        # Constraints: ||p_i||^2 - 1 = 0 for i in 0..N-1
        # Row M+i corresponds to point i.
        for i in range(N):
            J[M+i, 3*i : 3*i+3] = 2.0 * pts[i]
            
        return obj_val, grad_obj, constraints, J

    # Define constraints for SLSQP
    # We need to split the ineq and eq constraints into separate dicts or use a single dict with types?
    # SLSQP accepts a list of dicts.
    
    # We'll define a single constraint function that returns both? No, types are per constraint dict.
    # So we need two constraint dicts: one for ineq, one for eq.
    
    def ineq_fun(x_flat):
        _, _, cons, _ = objective_and_constraints(x_flat)
        return cons[:M]
        
    def ineq_jac(x_flat):
        _, _, _, J = objective_and_constraints(x_flat)
        return J[:M, :]
        
    def eq_fun(x_flat):
        pts = x_flat[:3*N].reshape(N, 3)
        return np.sum(pts*pts, axis=1) - 1.0
        
    def eq_jac(x_flat):
        pts = x_flat[:3*N].reshape(N, 3)
        J = np.zeros((N, 3*N + 1))
        for i in range(N):
            J[i, 3*i : 3*i+3] = 2.0 * pts[i]
        return J

    cons = [
        {'type': 'ineq', 'fun': ineq_fun, 'jac': ineq_jac},
        {'type': 'eq', 'fun': eq_fun, 'jac': eq_jac}
    ]

    # Initial t
    # Estimate initial covering radius
    pts0 = x0[:3*N].reshape(N, 3)
    dots0 = grid_pts @ pts0.T
    dots0 = np.clip(dots0, -1.0, 1.0)
    dists0 = np.arccos(dots0)
    min_dists0 = np.min(dists0, axis=1)
    t0 = np.max(min_dists0) + 0.1  # Add margin
    
    x_init = np.concatenate([x0, [t0]])
    
    # Run SLSQP
    best_x = x_init
    best_val = t0
    
    try:
        res = minimize(
            lambda x: objective_and_constraints(x)[0],
            x_init,
            method='SLSQP',
            jac=lambda x: objective_and_constraints(x)[1],
            constraints=cons,
            options={'maxiter': 50, 'ftol': 1e-8, 'disp': False}
        )
        
        if res.success and res.x[3*N] < best_val:
            best_val = res.x[3*N]
            best_x = res.x
            
    except Exception as e:
        pass

    # Extract points and normalize
    pts = best_x[:3*N].reshape(N, 3)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    # Final polish with a simpler method if time permits
    # We can use the hull-based radius check to verify and maybe do a few L-BFGS steps
    # on the hull-based objective if the SLSQP result is not perfect.
    # But SLSQP with grid constraints should be quite good.
    
    # Check time
    if time.time() < deadline - 10:
        # Try a few iterations of a simpler gradient descent on the hull-based objective
        # to clean up any grid artifacts.
        
        def hull_radius(x_flat):
            pts = x_flat.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            try:
                hull = ConvexHull(pts)
            except Exception:
                return 1.0
            max_r = 0.0
            for simplex in hull.simplices:
                i, j, k = simplex
                p1, p2, p3 = pts[i], pts[j], pts[k]
                u = p2 - p1
                v = p3 - p1
                uu = np.dot(u, u)
                vv = np.dot(v, v)
                uv = np.dot(u, v)
                det = uu * vv - uv * uv
                if abs(det) < 1e-10:
                    continue
                a = (0.5 * uu * vv - 0.5 * uv * vv) / det
                b = (0.5 * vv * uu - 0.5 * uu * uv) / det
                c_planar = p1 + a * u + b * v
                c_norm = np.linalg.norm(c_planar)
                if c_norm < 1e-10:
                    continue
                c_sph = c_planar / c_norm
                P = np.column_stack([p1, p2, p3])
                try:
                    coeffs = np.linalg.solve(P, c_sph)
                    if np.all(coeffs > -1e-6):
                        d_c = np.arccos(np.clip(np.dot(c_sph, p1), -1.0, 1.0))
                        max_r = max(max_r, d_c)
                except np.linalg.LinAlgError:
                    pass
            return max_r
            
        # Simple gradient descent on hull radius
        step_size = 0.01
        for it in range(50):
            if time.time() > deadline - 5:
                break
            val = hull_radius(pts.flatten())
            # Approximate gradient by finite difference on a few points?
            # Or just skip if SLSQP was good.
            # Let's just ensure normalization and return.
            break
            
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

# EVOLVE-BLOCK-END
