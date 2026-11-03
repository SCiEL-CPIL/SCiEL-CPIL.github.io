"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use random sampling
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 5000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1)))
    
    max_r = 0.0
    for simplex in hull.simplices:
        pts = points[simplex]
        if len(pts) != 3:
            continue
        a, b, c = pts[0], pts[1], pts[2]
        v = np.cross(b - a, c - a)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v = v / v_norm
        dot_a = np.dot(v, a)
        if dot_a < 0:
            v = -v
            dot_a = -dot_a
        cos_r = np.clip(dot_a, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
    
    return max_r


def _fibonacci_points(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, seed=42):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _icosahedron_based_points(n):
    """Generate points based on icosahedron symmetry with perturbations."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    
    pts = base.copy()
    rng = np.random.RandomState(123)
    while len(pts) < n:
        i, j = rng.randint(0, len(pts), 2)
        if i == j:
            continue
        mid = (pts[i] + pts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            mid /= mid_norm
            dists = 1.0 - np.sum(mid * pts, axis=1)
            if np.min(dists) > 0.01:
                pts = np.vstack([pts, mid])
    
    return pts[:n]


def _spherical_lloyd_iteration(points, max_iters=500, tol=1e-10, time_limit=None, start_time=None):
    """Spherical Lloyd's algorithm."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        new_points = np.zeros_like(points)
        point_to_facets = [[] for _ in range(len(points))]
        for fi, simplex in enumerate(hull.simplices):
            for idx in simplex:
                point_to_facets[idx].append(fi)
        
        for i in range(len(points)):
            if not point_to_facets[i]:
                new_points[i] = points[i]
                continue
            
            circumcenters = []
            for fi in point_to_facets[i]:
                simplex = hull.simplices[fi]
                pts = points[simplex]
                a, b, c = pts[0], pts[1], pts[2]
                v = np.cross(b - a, c - a)
                v_norm = np.linalg.norm(v)
                if v_norm < 1e-12:
                    continue
                v = v / v_norm
                dot_a = np.dot(v, a)
                if dot_a < 0:
                    v = -v
                circumcenters.append(v)
            
            if not circumcenters:
                new_points[i] = points[i]
                continue
            
            centroid = np.mean(circumcenters, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-12:
                new_points[i] = points[i]
            else:
                new_points[i] = centroid / centroid_norm
        
        diff = np.max(np.linalg.norm(new_points - points, axis=1))
        if diff < tol:
            points = new_points
            break
        
        points = new_points
        cr = _covering_radius(points)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    return best_points, best_cr


def _gram_matrix_optimization(points, time_limit=None, start_time=None):
    """
    Optimize the Gram matrix to maximize minimum off-diagonal entry (minimize max angular distance).
    Uses projected gradient ascent on a smooth surrogate.
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    def _soft_max_min_dot(G, beta=30.0):
        """
        Smooth approximation of the minimum off-diagonal entry of G.
        We want to maximize this, so we minimize -soft_max_min.
        """
        n = G.shape[0]
        # Extract off-diagonal entries
        mask = ~np.eye(n, dtype=bool)
        off_diag = G[mask]
        
        # Soft min: -log-sum-exp(-x*beta)/beta approximates min(x)
        scaled = -off_diag * beta
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -log_sum_exp / beta
        
        return soft_min
    
    def _objective_and_grad(x_flat):
        """
        Objective: -soft_min(off_diag(G)) where G = P P^T
        Gradient computed via chain rule.
        """
        P = x_flat.reshape(N, 3)
        # Project to sphere
        P_norms = np.linalg.norm(P, axis=1, keepdims=True)
        P_norms[P_norms < 1e-15] = 1.0
        P = P / P_norms
        
        # Gram matrix
        G = P @ P.T
        
        # Soft min of off-diagonal entries
        n = G.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off_diag = G[mask]
        
        beta = 30.0
        scaled = -off_diag * beta
        max_scaled = np.max(scaled)
        exp_scaled = np.exp(scaled - max_scaled)
        sum_exp = np.sum(exp_scaled)
        log_sum_exp = max_scaled + np.log(sum_exp)
        soft_min = -log_sum_exp / beta
        
        # We want to maximize soft_min, so minimize -soft_min
        obj = -soft_min
        
        # Gradient of soft_min w.r.t. off_diag entries
        weights = exp_scaled / sum_exp  # These are the softmax weights for the min
        
        # Build gradient of soft_min w.r.t. G
        grad_G = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    # d(soft_min)/d(G_ij) = -weights[idx] (since soft_min = -lse(-G*beta)/beta)
                    # Actually: soft_min = -1/beta * lse(-beta * off_diag)
                    # d(soft_min)/d(off_diag_k) = -1/beta * (-beta * exp(-beta*off_diag_k) / sum(exp(-beta*off_diag)))
                    # = exp(-beta*off_diag_k) / sum(exp(-beta*off_diag)) = weights[idx]
                    # So d(soft_min)/d(G_ij) = weights[idx] for i!=j
                    grad_G[i, j] = weights[idx]
                    idx += 1
        
        # Gradient of G w.r.t. P: dG/dP_k = P_k * 2 * I_k (where I_k is identity with 1 at k)
        # More precisely: G = P P^T, dG_ij = sum_k (dP_ik P_jk + P_ik dP_jk)
        # d(obj)/dP = -d(soft_min)/dP = -sum_ij d(soft_min)/dG_ij * dG_ij/dP
        
        grad_P = np.zeros_like(P)
        for k in range(n):
            # dG/dP_k contributes to all G_ij where i=k or j=k
            # dG_ik/dP_k = P_k (row vector), dG_kj/dP_k = P_k (column vector)
            # Actually G_ik = sum_m P_im P_km, so dG_ik/dP_km = P_km
            # G_kj = sum_m P_km P_jm, so dG_kj/dP_km = P_jm
            # So d(soft_min)/dP_k = sum_i grad_G[i,k] * P_k + sum_j grad_G[k,j] * P_j
            # = P_k * sum_i grad_G[i,k] + sum_j grad_G[k,j] * P_j
            
            row_sum = np.sum(grad_G[:, k])
            col_vec = np.sum(grad_G[k, :], axis=0)  # This is just grad_G[k, :]
            
            # Wait, let me redo this more carefully
            # G = P @ P.T
            # dG[i,j] = sum_m (dP[i,m]*P[j,m] + P[i,m]*dP[j,m])
            # d(obj)/dP[i,m] = sum_j d(obj)/dG[i,j] * dG[i,j]/dP[i,m]
            #                = sum_j d(obj)/dG[i,j] * P[j,m]
            #                = (d(obj)/dG[i,:]) @ P[:,m]
            # So d(obj)/dP = (d(obj)/dG) @ P.T
            
            pass
        
        # Cleaner: d(obj)/dP = (d(obj)/dG) @ P.T
        # where d(obj)/dG = -grad_G (since obj = -soft_min)
        grad_P = (-grad_G) @ P.T
        
        return obj, grad_P.flatten()
    
    # Run optimization
    x0 = points.flatten()
    
    try:
        result = minimize(
            _objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 150, 'gtol': 1e-8, 'ftol': 1e-10, 'maxfun': 300}
        )
        
        if result.x is not None:
            new_points = result.x.reshape(N, 3)
            new_norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            new_norms[new_norms < 1e-15] = 1.0
            new_points = new_points / new_norms
            new_cr = _covering_radius(new_points)
            if new_cr < best_cr:
                best_cr = new_cr
                best_points = new_points.copy()
    except Exception:
        pass
    
    return best_points, best_cr


def _gradient_descent(points, max_iters=2000, lr_init=0.005, lr_min=1e-8, time_limit=None, start_time=None):
    """Gradient descent to minimize covering radius."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    lr = lr_init
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        facet_data = []
        for simplex in hull.simplices:
            pts = points[simplex]
            if len(pts) != 3:
                continue
            a, b, c = pts[0], pts[1], pts[2]
            v = np.cross(b - a, c - a)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12:
                continue
            v = v / v_norm
            dot_a = np.dot(v, a)
            if dot_a < 0:
                v = -v
                dot_a = -dot_a
            cos_r = np.clip(dot_a, -1.0, 1.0)
            r = np.arccos(cos_r)
            facet_data.append((r, simplex, v))
        
        if not facet_data:
            break
        
        facet_data.sort(key=lambda x: -x[0])
        top_k = min(5, len(facet_data))
        
        step = np.zeros_like(points)
        for r, simplex, center in facet_data[:top_k]:
            weight = max(0.0, r - best_cr + 0.01)
            for idx in simplex:
                p = points[idx]
                d = center - p
                d_tangent = d - np.dot(d, p) * p
                step[idx] += weight * d_tangent
        
        step_norm = np.linalg.norm(step)
        if step_norm < 1e-15:
            break
        step /= step_norm
        
        new_points = points + lr * step
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_cr = _covering_radius(new_points)
        if new_cr < best_cr - 1e-13:
            best_cr = new_cr
            best_points = new_points.copy()
            points = new_points
            lr = max(lr * 0.999, lr_min)
        else:
            lr = max(lr * 0.92, lr_min)
            if lr < lr_min * 1.01:
                break
    
    return best_points, best_cr


def _simulated_annealing(points, time_limit=None, start_time=None, seed=42):
    """Simulated annealing to minimize covering radius."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    rng = np.random.RandomState(seed)
    T = 0.05
    cooling = 0.995
    T_min = 1e-6
    
    current_cr = best_cr
    current_points = points.copy()
    
    while T > T_min:
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        n_pert = rng.randint(1, 4)
        indices = rng.choice(N, size=n_pert, replace=False)
        perturbed = current_points.copy()
        for idx in indices:
            noise = rng.randn(3) * T
            perturbed[idx] = current_points[idx] + noise
            perturbed[idx] /= np.linalg.norm(perturbed[idx])
        
        new_cr = _covering_radius(perturbed)
        delta = new_cr - current_cr
        
        if delta < 0 or rng.rand() < np.exp(-delta / T):
            current_points = perturbed
            current_cr = new_cr
            if new_cr < best_cr:
                best_cr = new_cr
                best_points = perturbed.copy()
        
        T *= cooling
    
    return best_points, best_cr


def _nelder_mead_polish(points, time_limit=None, start_time=None):
    """High-precision polish using Nelder-Mead on individual coordinates."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    # Coordinate-wise Nelder-Mead for high precision
    from scipy.optimize import minimize
    
    def _objective(x_flat):
        P = x_flat.reshape(N, 3)
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        return _covering_radius(P)
    
    # Try a few random perturbations and polish the best
    rng = np.random.RandomState(999)
    for trial in range(3):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Small perturbation
        perturbed = points.copy()
        mask = rng.rand(N) < 0.3
        perturbed[mask] += rng.randn(np.sum(mask), 3) * 0.01
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
        
        try:
            result = minimize(
                _objective,
                perturbed.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-14}
            )
            
            new_points = result.x.reshape(N, 3)
            new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
            new_cr = _covering_radius(new_points)
            
            if new_cr < best_cr:
                best_cr = new_cr
                best_points = new_points.copy()
        except Exception:
            continue
    
    return best_points, best_cr


def _refine(points, time_limit=None, start_time=None):
    """Multi-stage refinement: Lloyd + Gram matrix + gradient descent + NM polish."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    # Stage 1: Lloyd's algorithm
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            return points, _covering_radius(points)
        lloyd_time = min(remaining * 0.2, 6)
    else:
        lloyd_time = 6
    
    points, cr1 = _spherical_lloyd_iteration(points, max_iters=200, time_limit=lloyd_time, start_time=start_time)
    
    # Stage 2: Gram matrix optimization
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return points, cr1
        gram_time = min(remaining * 0.3, 12)
    else:
        gram_time = 12
    
    points, cr2 = _gram_matrix_optimization(points, time_limit=gram_time, start_time=start_time)
    
    # Stage 3: Gradient descent polish
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            return points, cr2
        gd_time = min(remaining * 0.25, 8)
    else:
        gd_time = 8
    
    points, cr3 = _gradient_descent(points, max_iters=300, lr_init=0.003, time_limit=gd_time, start_time=start_time)
    
    # Stage 4: Nelder-Mead high-precision polish
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return points, cr3
        nm_time = min(remaining * 0.25, 10)
    else:
        nm_time = 10
    
    points, cr4 = _nelder_mead_polish(points, time_limit=nm_time, start_time=start_time)
    
    return points, cr4


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate multiple diverse initial configurations
    initial_configs = [
        _fibonacci_points(N),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
        _icosahedron_based_points(N),
    ]
    
    best_points = initial_configs[0].copy()
    best_cr = _covering_radius(best_points)
    
    # Phase 1: Refine each initial configuration
    for i, init_pts in enumerate(initial_configs):
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        time_per_config = min(remaining / (len(initial_configs) - i), 25)
        refined, cr = _refine(init_pts, time_limit=time_per_config, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = refined.copy()
    
    # Phase 2: Simulated annealing from best configuration
    remaining = deadline - time.time()
    if remaining > 15:
        sa_time = min(remaining * 0.3, 20)
        sa_points, sa_cr = _simulated_annealing(best_points, time_limit=sa_time, start_time=start_time, seed=123)
        if sa_cr < best_cr:
            best_cr = sa_cr
            best_points = sa_points.copy()
    
    # Phase 3: Final refinement with full pipeline
    remaining = deadline - time.time()
    if remaining > 10:
        final_time = min(remaining * 0.7, 20)
        final_points, final_cr = _refine(best_points, time_limit=final_time, start_time=start_time)
        if final_cr < best_cr:
            best_cr = final_cr
            best_points = final_points.copy()
    
    # Normalize final result
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
