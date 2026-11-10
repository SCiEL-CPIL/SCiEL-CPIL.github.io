import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    timeout = 600
    n = 14

    def calculate_score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def project_to_sphere(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        return points / norms

    def optimize_energy(points, iterations=3000, lr=0.02, beta=12.0):
        """
        Minimize repulsive energy E = - sum 1/d_ij^beta on the sphere.
        High beta strongly penalizes close pairs, driving toward equal spacing.
        """
        points = project_to_sphere(points.copy())
        n_pts = len(points)
        
        for i in range(iterations):
            if time.time() - start_time > timeout - 10:
                break
                
            # Calculate distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)
            mask = ~np.eye(n_pts, dtype=bool)
            dist_matrix[~mask] = 1e9
            
            # Energy gradient for E = - sum_{i<j} d_ij^-beta
            # grad_p_k E = beta * sum_{j != k} d_kj^(-beta-2) * (p_k - p_j)
            grad = np.zeros_like(points)
            for k in range(n_pts):
                p_k = points[k]
                d_kj = dist_matrix[k].copy()
                d_kj[k] = 1e9
                diff_k = diff[k]
                weights = np.power(d_kj, -beta - 2)
                weights[k] = 0
                grad_k = np.sum(diff_k * weights[:, np.newaxis], axis=0)
                grad[k] = beta * grad_k
            
            # Project to tangent space
            for k in range(n_pts):
                p_k = points[k]
                g_k = grad[k]
                proj = np.dot(g_k, p_k)
                grad[k] = g_k - proj * p_k
            
            # Update: move AGAINST gradient to decrease energy
            points = points - lr * grad
            points = project_to_sphere(points)
            
            # Decay learning rate
            if i % 100 == 0:
                lr *= 0.97
                
        return points

    def polish_min_dist(points, iterations=2000, lr=0.003):
        """
        Final polish to strictly maximize min distance via gradient ascent on d_min.
        """
        points = project_to_sphere(points.copy())
        n_pts = len(points)
        
        for i in range(iterations):
            if time.time() - start_time > timeout - 5:
                break
                
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)
            mask = ~np.eye(n_pts, dtype=bool)
            dist_matrix[~mask] = np.inf
            d_min = np.min(dist_matrix)
            
            # Find all pairs at minimum distance (within tolerance)
            min_pairs = np.argwhere(np.isclose(dist_matrix, d_min, atol=1e-10))
            
            grad = np.zeros_like(points)
            for idx in min_pairs:
                a, b = idx[0], idx[1]
                if a == b:
                    continue
                vec = points[a] - points[b]
                d = np.linalg.norm(vec)
                if d > 1e-12:
                    unit = vec / d
                    grad[a] += unit
                    grad[b] -= unit
            
            # Project to tangent space
            for k in range(n_pts):
                p_k = points[k]
                g_k = grad[k]
                proj = np.dot(g_k, p_k)
                grad[k] = g_k - proj * p_k
            
            points = points + lr * grad
            points = project_to_sphere(points)
            
            if i > 300:
                lr *= 0.998
                
        return points

    def slsqp_optimize(points, max_iter=200):
        """
        Use SLSQP to maximize minimum distance with sphere constraints.
        """
        pts = project_to_sphere(points.copy())
        x0 = pts.flatten()
        
        def neg_min_dist(x):
            pts_x = x.reshape(n, 3)
            dists = pdist(pts_x)
            return -np.min(dists)
        
        def neg_min_dist_grad(x):
            eps = 1e-7
            grad = np.zeros_like(x)
            base_val = neg_min_dist(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                grad[i] = (neg_min_dist(x_plus) - base_val) / eps
            return grad
        
        def sphere_constraint(x):
            pts_x = x.reshape(n, 3)
            return np.sum(pts_x * pts_x, axis=1) - 1.0
        
        def sphere_constraint_jac(x):
            pts_x = x.reshape(n, 3)
            jac = np.zeros((n, n * 3))
            for i in range(n):
                jac[i, i*3:(i+1)*3] = 2 * pts_x[i]
            return jac
        
        try:
            result = minimize(
                neg_min_dist,
                x0,
                method='SLSQP',
                jac=neg_min_dist_grad,
                constraints=[{'type': 'eq', 'fun': sphere_constraint, 'jac': sphere_constraint_jac}],
                options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
            )
            return project_to_sphere(result.x.reshape(n, 3))
        except Exception:
            return pts

    def fibonacci_sphere(n_pts, R=1.0):
        golden = (1 + np.sqrt(5)) / 2
        i = np.arange(n_pts)
        z = 1 - (2 * i + 1) / n_pts
        r = np.sqrt(np.maximum(0, 1 - z**2))
        theta = 2 * np.pi * i / golden
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts = np.column_stack([x, y, z])
        return pts * R

    def icosahedron_based_init():
        t = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ], dtype=float)
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        np.random.seed(42)
        extra1 = np.random.randn(3)
        extra1 = extra1 / np.linalg.norm(extra1)
        extra2 = np.random.randn(3)
        extra2 = extra2 / np.linalg.norm(extra2)
        
        return np.vstack([vertices, extra1.reshape(1,3), extra2.reshape(1,3)])

    def random_init(seed):
        np.random.seed(seed)
        pts = np.random.randn(n, 3)
        return project_to_sphere(pts)

    best_score = 0.0
    best_points = None

    # Generate diverse initializations
    inits = []
    inits.append(fibonacci_sphere(n))
    inits.append(icosahedron_based_init())
    
    for seed in [1, 2, 3, 7, 13, 21, 34, 55]:
        inits.append(random_init(seed))
    
    np.random.seed(123)
    inits.append(fibonacci_sphere(n) + np.random.randn(n, 3) * 0.15)
    np.random.seed(456)
    inits.append(fibonacci_sphere(n) + np.random.randn(n, 3) * 0.3)

    for init_idx, init in enumerate(inits):
        if time.time() - start_time > timeout - 30:
            break
        
        pts = project_to_sphere(init.copy())
        
        # Phase 1: Energy minimization with high beta for strong repulsion
        if time.time() - start_time < timeout - 60:
            p1 = optimize_energy(pts, iterations=2000, lr=0.02, beta=12.0)
            score1 = calculate_score(p1)
            if score1 > best_score:
                best_score = score1
                best_points = p1.copy()
        
        # Phase 2: Polish min distance
        if time.time() - start_time < timeout - 40:
            p2 = polish_min_dist(p1, iterations=1500, lr=0.003)
            score2 = calculate_score(p2)
            if score2 > best_score:
                best_score = score2
                best_points = p2.copy()
        
        # Phase 3: SLSQP refinement on the best so far
        if time.time() - start_time < timeout - 20:
            p3 = slsqp_optimize(best_points if best_points is not None else pts, max_iter=100)
            score3 = calculate_score(p3)
            if score3 > best_score:
                best_score = score3
                best_points = p3.copy()

    if best_points is None:
        best_points = fibonacci_sphere(n)

    return best_points
