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

    def sphere_constraints(x):
        pts = x.reshape(n, 3)
        return np.sum(pts * pts, axis=1) - 1.0

    def sphere_jac(x):
        pts = x.reshape(n, 3)
        jac = np.zeros((n, n * 3))
        for i in range(n):
            jac[i, i*3:(i+1)*3] = 2 * pts[i]
        return jac

    def neg_sum_power_dist(x, p=20):
        pts = x.reshape(n, 3)
        dists = pdist(pts)
        return -np.sum(dists ** p)

    def neg_min_dist(x):
        pts = x.reshape(n, 3)
        dists = pdist(pts)
        return -np.min(dists)

    def optimize_slsqp(points, objective_func, max_iter=300):
        pts = project_to_sphere(points.copy())
        x0 = pts.flatten()
        
        try:
            result = minimize(
                objective_func,
                x0,
                method='SLSQP',
                constraints=[{'type': 'eq', 'fun': sphere_constraints, 'jac': sphere_jac}],
                options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
            )
            if result.success:
                return project_to_sphere(result.x.reshape(n, 3))
            else:
                # Sometimes SLSQP fails but gives a better point anyway
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

    inits = []
    inits.append(fibonacci_sphere(n))
    inits.append(icosahedron_based_init())
    
    for seed in [1, 2, 3, 7, 13, 21, 34, 55, 89, 144, 233, 377]:
        inits.append(random_init(seed))
        
    # Add some perturbed fibonacci
    np.random.seed(123)
    inits.append(fibonacci_sphere(n) + np.random.randn(n, 3) * 0.1)
    np.random.seed(456)
    inits.append(fibonacci_sphere(n) + np.random.randn(n, 3) * 0.2)
    np.random.seed(789)
    inits.append(fibonacci_sphere(n) + np.random.randn(n, 3) * 0.05)

    for init in inits:
        if time.time() - start_time > timeout - 30:
            break
        
        pts = project_to_sphere(init.copy())
        
        # Phase 1: Optimize smooth proxy (sum of powers)
        p1 = optimize_slsqp(pts, neg_sum_power_dist, max_iter=200)
        score1 = calculate_score(p1)
        
        # Phase 2: Polish with min distance if time allows
        if time.time() - start_time < timeout - 10:
             p2 = optimize_slsqp(p1, neg_min_dist, max_iter=100)
             score2 = calculate_score(p2)
             if score2 > score1:
                 p_best = p2
                 score_best = score2
             else:
                 p_best = p1
                 score_best = score1
        else:
             p_best = p1
             score_best = score1

        if score_best > best_score:
            best_score = score_best
            best_points = p_best.copy()

    # Final polish on the best point found
    if best_points is not None and time.time() - start_time < timeout - 5:
        p_final = optimize_slsqp(best_points, neg_min_dist, max_iter=200)
        score_final = calculate_score(p_final)
        if score_final > best_score:
            best_score = score_final
            best_points = p_final.copy()

    if best_points is None:
        best_points = fibonacci_sphere(n)

    return best_points
