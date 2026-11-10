# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _score(points):
    """Compute (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = dists.min()
    d_max = dists.max()
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Strategy:
    1. Start from multiple structured configurations (Fibonacci sphere, icosahedron variants, random).
    2. Use scipy.optimize.minimize with L-BFGS-B on the objective -log(d_min/d_max) = -log(d_min) + log(d_max)
       with soft approximations via log-sum-exp for smoothness.
    3. Multi-start with many random inits and refine.
    4. Use a two-stage approach: first coarse optimization, then fine-tuning.
    """
    n = 14
    d = 3
    n_pairs = n * (n - 1) // 2  # 91
    
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > (time_limit - safety_margin)
    
    def normalize_sphere(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return points / norms
    
    def score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = dists.min()
        d_max = dists.max()
        if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio
    
    def logsumexp(x, beta):
        """Stable log-sum-exp for a vector of values."""
        if len(x) == 0:
            return 0.0
        x = np.asarray(x)
        xmax = x.max()
        if not np.isfinite(xmax):
            return 0.0
        return xmax + np.log(np.sum(np.exp(x - xmax)))
    
    def objective_soft(x, beta=50.0):
        """
        Soft objective: minimize -soft_log(d_min) + soft_log(d_max)
        where soft_log(min) = -logsumexp(-beta * d_ij) / beta
              soft_log(max) = logsumexp(beta * d_ij) / beta
        """
        pts = x.reshape(n, d)
        # Compute all pairwise distances
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        
        # Soft min: -log(exp(-beta*d1) + ... + exp(-beta*dn)) / beta
        soft_log_min = -logsumexp(-beta * dists, beta) / beta
        # Soft max: log(exp(beta*d1) + ... + exp(beta*dn)) / beta
        soft_log_max = logsumexp(beta * dists, beta) / beta
        
        return -soft_log_min + soft_log_max
    
    def objective_soft_grad(x, beta=50.0):
        """
        Gradient of the soft objective.
        """
        pts = x.reshape(n, d)
        dists = pdist(pts)
        if len(dists) == 0:
            return np.zeros_like(x)
        
        # Soft min gradient: d/dx [-logsumexp(-beta*d)/beta]
        # = (1/beta) * sum_i exp(-beta*d_i) * (-beta) * d/dx(d_i) / sum_j exp(-beta*d_j)
        # = -sum_i exp(-beta*d_i) * d/dx(d_i) / sum_j exp(-beta*d_j)
        
        # Soft max gradient: d/dx [logsumexp(beta*d)/beta]
        # = (1/beta) * sum_i exp(beta*d_i) * beta * d/dx(d_i) / sum_j exp(beta*d_j)
        # = sum_i exp(beta*d_i) * d/dx(d_i) / sum_j exp(beta*d_j)
        
        exp_neg = np.exp(-beta * dists)
        exp_pos = np.exp(beta * dists)
        
        w_neg = exp_neg / exp_neg.sum()
        w_pos = exp_pos / exp_pos.sum()
        
        # Build pairwise index mapping
        # pdist gives distances in order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
        # We need to compute d/dx for each pair
        
        grad = np.zeros_like(x)
        pair_idx = 0
        for i in range(n):
            for j in range(i+1, n):
                w_n = w_neg[pair_idx]
                w_p = w_pos[pair_idx]
                
                diff = pts[i] - pts[j]
                dist = dists[pair_idx]
                
                if dist < 1e-15:
                    # Perturb
                    rng = np.random.default_rng(42 + pair_idx)
                    diff = rng.standard_normal(3) * 1e-8
                    dist = np.linalg.norm(diff)
                
                # d/dp_i of distance = (p_i - p_j) / d
                # d/dp_j of distance = (p_j - p_i) / d = -(p_i - p_j) / d
                unit_diff = diff / dist
                
                # Objective = -soft_log_min + soft_log_max
                # grad from soft_log_min part: -w_n * d/dp_i(dist) = -w_n * unit_diff (for p_i)
                #                         + w_n * d/dp_j(dist) = +w_n * unit_diff (for p_j)
                # grad from soft_log_max part: +w_p * d/dp_i(dist) = +w_p * unit_diff (for p_i)
                #                         -w_p * d/dp_j(dist) = -w_p * unit_diff (for p_j)
                
                grad[2*i] += (-w_n + w_p) * unit_diff
                grad[2*j] += (w_n - w_p) * unit_diff
                
                pair_idx += 1
        
        return grad
    
    def hard_objective(x, eps=1e-6):
        """
        Hard objective with smoothing: minimize -log(d_min + eps) + log(d_max + eps)
        """
        pts = x.reshape(n, d)
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = dists.min() + eps
        d_max = dists.max() + eps
        return -np.log(d_min) + np.log(d_max)
    
    def hard_objective_grad(x, eps=1e-6):
        """
        Gradient of hard objective.
        """
        pts = x.reshape(n, d)
        dists = pdist(pts)
        if len(dists) == 0:
            return np.zeros_like(x)
        
        d_min = dists.min()
        d_max = dists.max()
        
        grad = np.zeros_like(x)
        
        # Gradient of -log(d_min): for the pair achieving min
        min_idx = dists.argmin()
        # Find which pair
        pair_idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if pair_idx == min_idx:
                    diff = pts[i] - pts[j]
                    dist = d_min
                    if dist < 1e-15:
                        rng = np.random.default_rng(42)
                        diff = rng.standard_normal(3) * 1e-8
                        dist = np.linalg.norm(diff)
                    unit_diff = diff / dist
                    # d/dp_i of -log(d) = -(p_i - p_j)/d^2 = -unit_diff/d
                    # d/dp_j of -log(d) = (p_j - p_i)/d^2 = unit_diff/d
                    grad[2*i] += -unit_diff / dist
                    grad[2*j] += unit_diff / dist
                    break
                pair_idx += 1
            else:
                continue
            break
        
        # Gradient of log(d_max): for the pair achieving max
        max_idx = dists.argmax()
        pair_idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if pair_idx == max_idx:
                    diff = pts[i] - pts[j]
                    dist = d_max
                    if dist < 1e-15:
                        rng = np.random.default_rng(43)
                        diff = rng.standard_normal(3) * 1e-8
                        dist = np.linalg.norm(diff)
                    unit_diff = diff / dist
                    # d/dp_i of log(d) = (p_i - p_j)/d^2 = unit_diff/d
                    # d/dp_j of log(d) = -(p_i - p_j)/d^2 = -unit_diff/d
                    grad[2*i] += unit_diff / dist
                    grad[2*j] += -unit_diff / dist
                    break
                pair_idx += 1
            else:
                continue
            break
        
        return grad
    
    def fibonacci_sphere(n_points):
        points = []
        for i in range(n_points):
            phi = np.arccos(1 - 2 * (i + 0.5) / n_points)
            theta = np.pi * (1 + 5**0.5) * i
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])
        return np.array(points)
    
    def random_sphere(n_points, seed):
        rng = np.random.default_rng(seed)
        points = rng.standard_normal((n_points, 3))
        return normalize_sphere(points)
    
    def icosahedron_plus_antipodal():
        """Icosahedron (12 vertices) + 2 antipodal points."""
        phi = (1 + 5**0.5) / 2
        icos = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        icos = normalize_sphere(icos)
        p1 = normalize_sphere(np.array([[1.0, 1.0, 1.0]]))[0]
        p2 = normalize_sphere(np.array([[-1.0, -1.0, -1.0]]))[0]
        return np.vstack([icos, [p1, p2]])
    
    def optimize_from_init(init_pts, method='soft', maxiter=2000):
        """
        Optimize from an initial configuration using scipy.optimize.
        """
        x0 = init_pts.flatten()
        
        if method == 'soft':
            for beta in [20.0, 50.0, 100.0]:
                res = minimize(
                    objective_soft,
                    x0,
                    args=(beta,),
                    jac=lambda x, b=beta: objective_soft_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': maxiter, 'maxfun': maxiter * n * d, 'gtol': 1e-8}
                )
                if res.success or res.fun < objective_soft(x0, beta):
                    x0 = res.x
        elif method == 'hard':
            res = minimize(
                hard_objective,
                x0,
                jac=hard_objective_grad,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'maxfun': maxiter * n * d, 'gtol': 1e-8}
            )
            if res.success or res.fun < hard_objective(x0):
                x0 = res.x
        
        pts = x0.reshape(n, d)
        return normalize_sphere(pts)
    
    # Generate initial configurations
    inits = []
    inits.append(fibonacci_sphere(n))
    inits.append(icosahedron_plus_antipodal())
    
    # Many random starts
    for seed in range(30):
        inits.append(random_sphere(n, seed))
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: Soft optimization from all inits
    for idx, init in enumerate(inits):
        if check_deadline():
            break
        
        optimized = optimize_from_init(init, method='soft', maxiter=1000)
        s = score(optimized)
        if s > best_score:
            best_score = s
            best_points = optimized.copy()
    
    # Phase 2: Hard optimization from the best
    if best_points is not None and not check_deadline():
        refined = optimize_from_init(best_points, method='hard', maxiter=1000)
        s = score(refined)
        if s > best_score:
            best_score = s
            best_points = refined.copy()
    
    # Phase 3: Multi-start with more diverse inits and refinement
    if not check_deadline():
        for seed in range(30, 80):
            if check_deadline():
                break
            init = random_sphere(n, seed)
            optimized = optimize_from_init(init, method='soft', maxiter=500)
            s = score(optimized)
            if s > best_score:
                best_score = s
                best_points = optimized.copy()
                # Refine the new best
                refined = optimize_from_init(best_points, method='hard', maxiter=500)
                s2 = score(refined)
                if s2 > best_score:
                    best_score = s2
                    best_points = refined.copy()
    
    # Phase 4: Fine-tuning with more iterations
    if best_points is not None and not check_deadline():
        for beta in [100.0, 200.0]:
            if check_deadline():
                break
            res = minimize(
                objective_soft,
                best_points.flatten(),
                args=(beta,),
                jac=lambda x, b=beta: objective_soft_grad(x, b),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 1000 * n * d, 'gtol': 1e-10}
            )
            pts = normalize_sphere(res.x.reshape(n, d))
            s = score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
    
    # Fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    # Validate
    if not np.all(np.isfinite(best_points)) or best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
