# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution, basinhopping

def _score(points):
    """Compute (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = dists.min()
    d_max = dists.max()
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Uses quaternion-based optimization with Hamiltonian product operations
    to navigate configuration space, combined with multi-stage refinement.
    """
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0
    
    best_points = None
    best_score = -1.0
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > (time_limit - safety_margin)
    
    def objective(points):
        """Returns (d_min/d_max)^2, higher is better."""
        if len(points) < 2:
            return 0.0
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = dists.min()
        d_max = dists.max()
        if d_max == 0:
            return 0.0
        if not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def smooth_objective(points, beta=50.0):
        """
        Smooth surrogate: maximize soft_min(dists) / max(dists)
        """
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_max = dists.max()
        if d_max == 0 or not np.isfinite(d_max):
            return 0.0
        min_d = dists.min()
        shifted = -beta * (dists - min_d)
        max_shifted = np.max(shifted)
        log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        soft_min = min_d - log_sum / beta
        
        if not np.isfinite(soft_min) or soft_min <= 0:
            return 0.0
        
        ratio = soft_min / d_max
        if not np.isfinite(ratio):
            return 0.0
        return ratio * ratio

    def neg_smooth(x, beta=50.0):
        """Negative smooth objective for minimization."""
        points = x.reshape(n, d)
        return -smooth_objective(points, beta)

    def neg_objective(x):
        """Negative exact objective for minimization."""
        points = x.reshape(n, d)
        return -objective(points)
    
    # Quaternion utilities
    def quat_multiply(q1, q2):
        """Hamilton product of two quaternions [w, x, y, z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def quat_conj(q):
        """Conjugate of quaternion [w, x, y, z]."""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def quat_norm(q):
        """Norm of quaternion."""
        return np.sqrt(np.sum(q**2))
    
    def quat_normalize(q):
        """Normalize quaternion."""
        n = quat_norm(q)
        if n < 1e-15:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / n
    
    def quat_to_rotation_matrix(q):
        """Convert unit quaternion to 3x3 rotation matrix."""
        q = quat_normalize(q)
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
    
    def points_to_quats(points):
        """Convert 3D points to unit quaternions (treating as pure quaternion imaginary part)."""
        quats = []
        for p in points:
            q = np.array([0.0, p[0], p[1], p[2]])
            quats.append(quat_normalize(q))
        return np.array(quats)
    
    def quats_to_points(quats):
        """Convert unit quaternions back to 3D points (extract imaginary part)."""
        points = []
        for q in quats:
            points.append(q[1:4])
        return np.array(points)
    
    def quat_geodesic_distance(q1, q2):
        """Geodesic distance between two unit quaternions."""
        dot = np.dot(q1, q2)
        dot = np.clip(dot, -1.0, 1.0)
        return np.arccos(dot)
    
    def quat_pdist(quats):
        """Compute pairwise geodesic distances between quaternions."""
        m = len(quats)
        dists = np.zeros((m, m))
        for i in range(m):
            for j in range(i+1, m):
                d = quat_geodesic_distance(quats[i], quats[j])
                dists[i, j] = d
                dists[j, i] = d
        # Return condensed form like pdist
        idx = np.triu_indices(m, k=1)
        return dists[idx]
    
    def smooth_quat_objective(quats, beta=50.0):
        """Smooth objective on quaternion geodesic distances."""
        dists = quat_pdist(quats)
        if len(dists) == 0:
            return 0.0
        d_max = dists.max()
        if d_max == 0 or not np.isfinite(d_max):
            return 0.0
        min_d = dists.min()
        shifted = -beta * (dists - min_d)
        max_shifted = np.max(shifted)
        log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        soft_min = min_d - log_sum / beta
        
        if not np.isfinite(soft_min) or soft_min <= 0:
            return 0.0
        
        ratio = soft_min / d_max
        if not np.isfinite(ratio):
            return 0.0
        return ratio * ratio
    
    def neg_smooth_quat(x, beta=50.0):
        """Negative smooth quaternion objective for minimization."""
        quats = x.reshape(n, 4)
        # Normalize quaternions
        quats = np.array([quat_normalize(q) for q in quats])
        return -smooth_quat_objective(quats, beta)
    
    def neg_objective_quat(x):
        """Negative exact quaternion objective for minimization."""
        quats = x.reshape(n, 4)
        quats = np.array([quat_normalize(q) for q in quats])
        points = quats_to_points(quats)
        return -objective(points)
    
    def quaternion_local_search(x0, max_iter=1000, step_size=0.1):
        """Local search using quaternion perturbations via Hamiltonian product."""
        quats = x0.reshape(n, 4)
        quats = np.array([quat_normalize(q) for q in quats])
        
        best_score = smooth_quat_objective(quats, beta=100.0)
        best_quats = quats.copy()
        
        for it in range(max_iter):
            if check_deadline():
                break
            
            improved = False
            # Try perturbations along random quaternion directions
            for trial in range(5):
                # Generate random rotation quaternion
                angle = np.random.uniform(-step_size, step_size)
                axis = np.random.standard_normal(3)
                axis = axis / np.linalg.norm(axis)
                rot_q = np.array([np.cos(angle/2), 
                                  axis[0]*np.sin(angle/2),
                                  axis[1]*np.sin(angle/2),
                                  axis[2]*np.sin(angle/2)])
                
                # Apply rotation to a random subset of quaternions
                new_quats = quats.copy()
                idx = np.random.choice(n, size=max(1, n//4), replace=False)
                for i in idx:
                    new_quats[i] = quat_multiply(rot_q, quats[i])
                    new_quats[i] = quat_normalize(new_quats[i])
                
                score = smooth_quat_objective(new_quats, beta=100.0)
                if score > best_score + 1e-10:
                    best_score = score
                    best_quats = new_quats.copy()
                    improved = True
            
            if not improved:
                step_size *= 0.9
                if step_size < 0.001:
                    break
        
        return best_quats.flatten()
    
    # Initialization strategies
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
    
    def symmetric_start(n_points):
        phi = (1 + 5**0.5) / 2
        icosahedron = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(icosahedron, axis=1, keepdims=True)
        icosahedron /= norms
        
        p13 = np.array([1.0, 1.0, 1.0])
        p14 = np.array([-1.0, -1.0, -1.0])
        p13 /= np.linalg.norm(p13)
        p14 /= np.linalg.norm(p14)
        
        return np.vstack([icosahedron, [p13, p14]])
    
    def random_sphere(n_points, seed):
        rng = np.random.default_rng(seed)
        points = rng.standard_normal((n_points, 3))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms
    
    def static_solution():
        pts = np.array([
            [ 0.000000,  0.000000,  1.000000],
            [ 0.000000, -1.000000,  0.000000],
            [ 0.000000,  1.000000,  0.000000],
            [ 1.000000,  0.000000,  0.000000],
            [-1.000000,  0.000000,  0.000000],
            [ 0.000000,  0.000000, -1.000000],
            [ 0.577350,  0.577350,  0.577350],
            [ 0.577350, -0.577350, -0.577350],
            [-0.577350,  0.577350, -0.577350],
            [-0.577350, -0.577350,  0.577350],
            [ 0.816497,  0.408248, -0.408248],
            [-0.816497,  0.408248,  0.408248],
            [ 0.408248, -0.816497,  0.408248],
            [-0.408248, -0.816497, -0.408248],
        ])
        return pts
    
    # Phase 1: Differential evolution for global optimization (Cartesian)
    if not check_deadline():
        try:
            bounds = [(-1.0, 1.0)] * (n * d)
            de_result = differential_evolution(
                neg_smooth,
                bounds,
                args=(20.0,),
                maxiter=100,
                popsize=15,
                seed=42,
                tol=1e-6,
                polish=False
            )
            
            if de_result.success or not check_deadline():
                de_points = de_result.x.reshape(n, d)
                de_score = objective(de_points)
                
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points.copy()
                    
        except Exception:
            pass
    
    # Phase 2: Quaternion-based optimization
    if not check_deadline():
        try:
            # Initialize quaternions from best points found so far
            if best_points is not None:
                init_quats = points_to_quats(best_points)
            else:
                init_quats = points_to_quats(fibonacci_sphere(n))
            
            x0_quat = init_quats.flatten()
            
            # Basin hopping in quaternion space
            result_quat = basinhopping(
                lambda x: neg_smooth_quat(x, beta=30.0),
                x0_quat,
                minimizer_kwargs={
                    'method': 'Powell',
                    'options': {
                        'maxiter': 300,
                        'xatol': 1e-8,
                        'fatol': 1e-10
                    }
                },
                niter=15,
                stepsize=0.5,
                T=50.0,
                seed=42
            )
            
            if not check_deadline():
                # Sequential refinement with increasing beta
                x_current = result_quat.x
                
                for beta in [50.0, 100.0, 200.0, 300.0, 500.0]:
                    if check_deadline():
                        break
                    
                    res = minimize(
                        lambda x, b=beta: neg_smooth_quat(x, b),
                        x_current,
                        method='Powell',
                        options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    x_current = res.x
                
                if not check_deadline():
                    # Convert to points and evaluate
                    quats = x_current.reshape(n, 4)
                    quats = np.array([quat_normalize(q) for q in quats])
                    points_from_quat = quats_to_points(quats)
                    score_quat = objective(points_from_quat)
                    
                    if score_quat > best_score:
                        best_score = score_quat
                        best_points = points_from_quat.copy()
                        
        except Exception:
            pass
    
    # Phase 3: Multi-start basin hopping + refinement (Cartesian)
    init_strategies = []
    init_strategies.append(("fib_sphere", fibonacci_sphere(n)))
    init_strategies.append(("symmetric", symmetric_start(n)))
    init_strategies.append(("static", static_solution()))
    
    for seed in range(8):
        init_strategies.append((f"rand_sphere_{seed}", random_sphere(n, seed)))
    
    # Add best result if found
    if best_points is not None:
        init_strategies.append(("best_so_far", best_points.copy()))
    
    for name, init_points in init_strategies:
        if check_deadline():
            break
        
        x0 = init_points.flatten()
        
        try:
            # Basin hopping with smooth surrogate
            result = basinhopping(
                lambda x: neg_smooth(x, beta=30.0),
                x0,
                minimizer_kwargs={
                    'method': 'Powell',
                    'options': {
                        'maxiter': 400,
                        'xatol': 1e-8,
                        'fatol': 1e-10
                    }
                },
                niter=15,
                stepsize=0.8,
                T=100.0,
                seed=42
            )
            
            if check_deadline():
                break
            
            # Sequential refinement with increasing beta
            x_current = result.x
            
            for beta in [50.0, 100.0, 200.0, 300.0, 500.0]:
                if check_deadline():
                    break
                
                res = minimize(
                    lambda x, b=beta: neg_smooth(x, b),
                    x_current,
                    method='Powell',
                    options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                x_current = res.x
            
            if check_deadline():
                break
            
            # SLSQP polish
            res_slsqp = minimize(
                lambda x: neg_smooth(x, beta=500.0),
                x_current,
                method='SLSQP',
                options={'maxiter': 1500, 'ftol': 1e-14}
            )
            
            if check_deadline():
                break
            
            # Final exact-objective Nelder-Mead polish
            res_final = minimize(
                neg_objective,
                res_slsqp.x,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            
            if check_deadline():
                break
            
            optimized_points = res_final.x.reshape(n, d)
            score = objective(optimized_points)
            
            if score > best_score:
                best_score = score
                best_points = optimized_points.copy()
                
        except Exception:
            continue
    
    # Phase 4: Quaternion local search from best point
    if not check_deadline() and best_points is not None:
        try:
            quats = points_to_quats(best_points)
            x0_quat = quats.flatten()
            
            # Run quaternion local search
            x_quat_opt = quaternion_local_search(x0_quat, max_iter=500, step_size=0.2)
            
            quats_opt = x_quat_opt.reshape(n, 4)
            quats_opt = np.array([quat_normalize(q) for q in quats_opt])
            points_quat_opt = quats_to_points(quats_opt)
            
            # Polish in Cartesian space
            res_polish = minimize(
                neg_objective,
                points_quat_opt.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            
            points_final = res_polish.x.reshape(n, d)
            score_final = objective(points_final)
            
            if score_final > best_score:
                best_score = score_final
                best_points = points_final.copy()
                
        except Exception:
            pass
    
    # If no valid result, use Fibonacci sphere as fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    # Verify the result is valid
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)
    
    # Ensure shape is correct
    if best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    # Final validation
    if not np.isfinite(best_score) or best_score < 0:
        best_points = fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
