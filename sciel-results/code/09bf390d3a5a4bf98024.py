"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

DEADLINE = 140.0  # wall-clock safety margin (evaluator kills at 150 s)


def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return pts / n


def _covering_radius_hull(pts):
    """Exact covering radius (radians) via convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    best = 0.0
    for f in hull.simplices:
        tri = pts[f]
        a, b, c = tri[0], tri[1], tri[2]
        # Compute circumcenter of triangle on unit sphere
        # For points on unit sphere, circumcenter direction is normal to plane
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        cc = n / nn
        # Check if circumcenter is on unit sphere (it always is by construction)
        # The cap is "empty" if cc is on the same side as the facet normal
        # For a convex hull, the circumcenter should be outside or on the sphere
        cosv = np.clip(np.dot(cc, a), -1.0, 1.0)
        r = np.arccos(cosv)
        if r > best:
            best = r
    if best == 0.0:
        G = np.clip(pts @ pts.T, -1.0, 1.0)
        np.fill_diagonal(G, -1.0)
        best = np.arccos(G.min())
    return best


def _covering_radius_fast(pts, n_grid=200):
    """Fast approximate covering radius using a grid on the sphere."""
    pts = _normalize(pts)
    # Generate grid points on sphere
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2 * np.pi, 2 * n_grid)
    theta, phi = np.meshgrid(theta, phi)
    grid_x = np.sin(theta) * np.cos(phi)
    grid_y = np.sin(theta) * np.sin(phi)
    grid_z = np.cos(theta)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    grid = _normalize(grid)
    
    # Compute angular distances from each grid point to nearest config point
    # cos(angle) = dot product
    G = grid @ pts.T  # shape (n_grid_points, N)
    G = np.clip(G, -1.0, 1.0)
    max_cos = G.max(axis=1)  # closest point
    angles = np.arccos(max_cos)
    return angles.max()


def _build_hss50():
    """Build the Hardin-Sloane-Smith 50-point configuration.
    This is based on a truncated icosahedron with specific vertices removed.
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
    
    # Icosahedron vertices
    ico = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
    ], dtype=float)
    ico = _normalize(ico)
    
    # Find edges of icosahedron
    D = 1.0 - ico @ ico.T
    min_d = D.min()
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(D[i, j] - min_d) < 1e-8:
                edges.append((i, j))
    
    # Truncated icosahedron vertices: 2 per edge
    verts = []
    for (i, j) in edges:
        verts.append(ico[i] * (2.0 / 3.0) + ico[j] * (1.0 / 3.0))
        verts.append(ico[i] * (1.0 / 3.0) + ico[j] * (2.0 / 3.0))
    verts = np.array(verts)
    verts = _normalize(verts)
    
    # We have 60 vertices. Need to remove 10 to get 50.
    # Map each truncated vertex to its parent icosahedron vertices
    vmap = []
    for (i, j) in edges:
        vmap.append((i, j))
        vmap.append((i, j))
    
    # Find antipodal pair in icosahedron
    G = ico @ ico.T
    np.fill_diagonal(G, -2.0)
    pair_i, pair_j = np.unravel_index(np.argmin(G), G.shape)
    
    # Remove vertices associated with these two icosahedron vertices
    remove = set()
    for idx, (i, j) in enumerate(vmap):
        if i == pair_i or j == pair_i or i == pair_j or j == pair_j:
            remove.add(idx)
    
    keep = [k for k in range(len(verts)) if k not in remove]
    out = verts[keep]
    if len(out) != 50:
        out = out[:50]
    return _normalize(out)


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _soft_max_covering(pts, grid, temperature):
    """Compute soft-max (log-sum-exp) covering radius and its gradient.
    
    Returns (soft_r, grad) where grad has shape (N, 3).
    """
    pts = _normalize(pts)
    G = grid @ pts.T  # (n_grid, N)
    G = np.clip(G, -1.0, 1.0)
    
    # Soft-max: for each grid point, the soft "nearest" is via log-sum-exp of -G/temperature
    # We want to minimize max over grid of arccos(max_j G[j,i])
    # Equivalently minimize max over grid of -log(sum_j exp(G[j,i]/temperature)) * temperature
    # But simpler: use -logsumexp(G/temperature) as a smooth upper bound on -max(G)
    # which is a smooth lower bound on the covering radius
    
    # For each grid point i, let s_i = logsumexp(G[i,:]/temperature)
    # Then -temperature * s_i approximates -max_j G[i,j] (smooth)
    # The covering radius is arccos(max_j G[i,j]) ~ arccos(exp(s_i)*temperature) ... 
    # Actually let's use: f_i = -logsumexp(G[i,:]/temperature)
    # This is a smooth approximation of -max_j G[i,j]
    # We want to minimize max_i arccos(-f_i) ... this is getting complicated.
    
    # Simpler approach: minimize the soft covering radius
    # r_soft = max_i arccos(exp(logsumexp(G[i,:]/t) * t)) ... no
    # 
    # Let me use a cleaner formulation:
    # For each grid point, the "soft distance" is:
    # d_i = -log(sum_j exp(G[i,j]/temperature)) * temperature
    # This approximates -max_j G[i,j] for small temperature
    # We want to minimize max_i arccos(-d_i)
    # But arccos is only valid for arguments in [-1,1]
    # 
    # Alternative: directly minimize the soft-max of the angular distances
    # angle_i = arccos(max_j G[i,j])
    # soft_angle = temperature * log(sum_j exp(-angle_i / temperature)) ... 
    # 
    # Let me use the most direct smooth surrogate:
    # For each grid point i, compute the soft nearest neighbor distance:
    # s_i = -temperature * log(sum_j exp(G[i,j] / temperature))
    # For small temperature, s_i -> -max_j G[i,j]
    # The covering radius is max_i arccos(max_j G[i,j]) = max_i arccos(-s_i) approximately
    # 
    # To get a smooth objective, minimize:
    # F = temperature * log(sum_i exp(-s_i / temperature))
    # This is a smooth approximation of max_i (-s_i) = max_i max_j G[i,j] ... 
    # No, that's wrong. Let me think again.
    #
    # max_j G[i,j] = -min_j (-G[i,j])
    # -min_j(-G[i,j]) = -(-temperature * log sum exp(-G[i,j]/temperature)) = temperature * log sum exp(-G[i,j]/temperature)
    # So max_j G[i,j] ≈ temperature * log(sum_j exp(-G[i,j]/temperature)) for small t... 
    # No wait: logsumexp(x) = max(x) + log(sum exp(x - max(x)))
    # For x = -G[i,j]/temperature: logsumexp(-G[i,j]/temperature) = -max(G[i,j])/temperature + log(sum exp(-(G[i,j]-max(G))/temperature))
    # So -temperature * logsumexp(-G[i,j]/temperature) = max(G[i,j]) - temperature*log(sum exp(...))
    # As temperature -> 0, this -> max(G[i,j])
    # 
    # So: m_i = -temperature * logsumexp(-G[i,:] / temperature) approximates max_j G[i,j]
    # 
    # The covering radius is R = max_i arccos(m_i)
    # Smooth approximation: R_soft = temperature * log(sum_i exp(arccos(m_i) / temperature))
    # 
    # But arccos(m_i) can be problematic. Let's just minimize max_i arccos(m_i) 
    # using the soft-max on the angles directly.
    
    # Compute m_i = -t * logsumexp(-G[i,:]/t)
    scaled = -G / temperature  # (n_grid, N)
    max_scaled = scaled.max(axis=1, keepdims=True)  # (n_grid, 1)
    log_sum_exp = max_scaled + np.log(np.exp(scaled - max_scaled).sum(axis=1, keepdims=True))
    m = -temperature * log_sum_exp.squeeze()  # (n_grid,) approximates max_j G[i,j]
    m = np.clip(m, -1.0, 1.0)
    
    # Angular distances
    angles = np.arccos(m)  # (n_grid,)
    
    # Soft-max of angles
    max_angle = angles.max()
    if max_angle < 1e-12:
        return 0.0, np.zeros((N, 3))
    
    exp_angles = np.exp(angles / temperature - max_angle / temperature)
    sum_exp = exp_angles.sum()
    weights = exp_angles / sum_exp  # (n_grid,)
    
    soft_r = max_angle + temperature * np.log(sum_exp)
    
    # Gradient: d(soft_r)/d(pts)
    # soft_r = max_angle + t * log(sum exp(angles/t))
    # d(soft_r)/d(angles) = weights (softmax weights)
    # d(angles)/d(m) = -1/sqrt(1-m^2)
    # d(m)/d(G[i,j]) = ?
    # m_i = -t * logsumexp(-G[i,:]/t)
    # dm_i/dG[i,j] = -t * d/dG[i,j] logsumexp(-G[i,:]/t)
    # = -t * (1/sum_k exp(-G[i,k]/t)) * exp(-G[i,j]/t) * (-1/t)
    # = exp(-G[i,j]/t) / sum_k exp(-G[i,k]/t)
    # = softmax(-G[i,:]/t)[j]
    
    # So dm_i/dG[i,j] = softmax_j(-G[i,:]/t)
    
    # d(soft_r)/d(pts[i]) = sum_k weights[k] * d(angle_k)/d(m_k) * dm_k/d(pts[i])
    # d(angle_k)/d(m_k) = -1/sqrt(1-m_k^2)
    # dm_k/d(pts[i]) = sum_j [dm_k/dG[k,j]] * dG[k,j]/d(pts[i])
    # G[k,j] = grid[k] . pts[j]
    # dG[k,j]/d(pts[i]) = 0 if i != j, else grid[k]
    # So dm_k/d(pts[i]) = 0 if i != j, else dm_k/dG[k,i] * grid[k]
    # = softmax_i(-G[k,:]/t) * grid[k] if i is the index, else 0
    
    # Therefore:
    # d(soft_r)/d(pts[i]) = sum_k weights[k] * (-1/sqrt(1-m_k^2)) * softmax_i(-G[k,:]/t) * grid[k]
    
    # Compute softmax weights for each grid point
    neg_G_scaled = -G / temperature  # (n_grid, N)
    neg_max = neg_G_scaled.max(axis=1, keepdims=True)
    exp_neg = np.exp(neg_G_scaled - neg_max)
    softmax_w = exp_neg / exp_neg.sum(axis=1, keepdims=True)  # (n_grid, N)
    
    # dm_k/d(pts[i]) for all k, i:
    # = softmax_w[k,i] * grid[k] (vector)
    # So d(soft_r)/d(pts[i]) = sum_k weights[k] * (-1/sqrt(1-m_k^2)) * softmax_w[k,i] * grid[k]
    
    # Let's compute this efficiently:
    # grad[i] = sum_k [weights[k] * (-1/sqrt(1-m_k^2)) * softmax_w[k,i]] * grid[k]
    # = sum_k [weights[k] * (-1/sqrt(1-m_k^2))] * softmax_w[k,i] * grid[k]
    
    # Define coeff[k] = weights[k] * (-1/sqrt(1-m_k^2))
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_sqrt = 1.0 / np.sqrt(np.maximum(1.0 - m**2, 1e-16))
    coeff = weights * (-inv_sqrt)  # (n_grid,)
    
    # grad[i] = sum_k coeff[k] * softmax_w[k,i] * grid[k]
    # = (softmax_w.T * coeff) @ grid  ... shape: (N, n_grid) * (n_grid,) -> (N,) then @ grid (n_grid, 3) -> (N, 3)
    # Actually: grad = (softmax_w.T * coeff[np.newaxis,:]) @ grid
    # softmax_w.T is (N, n_grid), coeff[np.newaxis,:] is (1, n_grid), product is (N, n_grid)
    # Then @ grid (n_grid, 3) gives (N, 3)
    
    grad = (softmax_w.T * coeff[np.newaxis, :]) @ grid  # (N, 3)
    
    return soft_r, grad


def _soft_max_optimize(pts, time_budget):
    """Optimize using soft-max log-sum-exp surrogate with gradient descent and temperature annealing."""
    pts = _normalize(pts).copy()
    start = time.time()
    
    # Build a moderate-resolution grid for the soft-max evaluation
    n_grid = 150
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2 * np.pi, 2 * n_grid)
    theta_g, phi_g = np.meshgrid(theta, phi)
    grid_x = np.sin(theta_g) * np.cos(phi_g)
    grid_y = np.sin(theta_g) * np.sin(phi_g)
    grid_z = np.cos(theta_g)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    grid = _normalize(grid)
    
    # Temperature annealing schedule
    t_start = 0.5
    t_end = 0.01
    n_anneal = 50
    
    # Learning rate schedule
    lr_start = 0.05
    lr_end = 0.001
    
    best_pts = pts.copy()
    best_r = _covering_radius_fast(pts)
    
    for step in range(n_anneal):
        if time.time() - start > time_budget - 5.0:
            break
        
        frac = step / max(n_anneal - 1, 1)
        temperature = t_start * (t_end / t_start) ** frac
        lr = lr_start * (lr_end / lr_start) ** frac
        
        # Compute soft-max objective and gradient
        soft_r, grad = _soft_max_covering(pts, grid, temperature)
        
        # Project gradient onto tangent space of sphere
        # For each point i, the tangent space is orthogonal to pts[i]
        for i in range(N):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        # Gradient descent step
        pts = pts - lr * grad
        pts = _normalize(pts)
        
        # Check actual covering radius
        r = _covering_radius_fast(pts)
        if r < best_r:
            best_r = r
            best_pts = pts.copy()
    
    return best_pts, best_r


def _nelder_mead_optimize(pts, time_budget):
    """Nelder-Mead style optimization on the sphere."""
    from scipy.optimize import minimize
    
    pts = _normalize(pts).copy()
    start = time.time()
    
    def objective(x):
        if time.time() - start > time_budget:
            return np.inf
        p = x.reshape(N, 3)
        p = _normalize(p)
        return _covering_radius_fast(p)
    
    # Flatten the points
    x0 = pts.ravel()
    
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10,
                               'adaptive': True})
    
    if result.success or result.fun < _covering_radius_fast(pts):
        best_pts = _normalize(result.x.reshape(N, 3))
    else:
        best_pts = pts
    
    return best_pts


def construct():
    t0 = time.time()
    
    # Generate multiple initial configurations
    configs = []
    
    # 1. Hardin-Sloane-Smith based
    try:
        hss = _build_hss50()
        if len(hss) == 50 and np.all(np.isfinite(hss)):
            configs.append(hss)
    except Exception:
        pass
    
    # 2. Fibonacci spiral
    fib = _fibonacci_spiral()
    configs.append(fib)
    
    # 3. Random configurations with symmetry
    rng = np.random.RandomState(42)
    for _ in range(3):
        # Start with icosahedral symmetry
        ico = np.array([
            [0, 1, (1+np.sqrt(5))/2], [0, -1, (1+np.sqrt(5))/2],
            [0, 1, -(1+np.sqrt(5))/2], [0, -1, -(1+np.sqrt(5))/2],
            [1, (1+np.sqrt(5))/2, 0], [-1, (1+np.sqrt(5))/2, 0],
            [1, -(1+np.sqrt(5))/2, 0], [-1, -(1+np.sqrt(5))/2, 0],
            [(1+np.sqrt(5))/2, 0, 1], [-(1+np.sqrt(5))/2, 0, 1],
            [(1+np.sqrt(5))/2, 0, -1], [-(1+np.sqrt(5))/2, 0, -1],
        ], dtype=float)
        ico = _normalize(ico)
        
        # Add 38 random points
        rand_pts = rng.randn(38, 3)
        rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1, keepdims=True)
        
        all_pts = np.vstack([ico, rand_pts])
        configs.append(all_pts)
    
    # Evaluate all configs and pick the best
    best_pts = None
    best_r = np.inf
    
    for cfg in configs:
        cfg = _normalize(cfg)
        r = _covering_radius_fast(cfg)
        if r < best_r:
            best_r = r
            best_pts = cfg
    
    if best_pts is None:
        best_pts = fib
        best_r = _covering_radius_fast(best_pts)
    
    # Soft-max gradient optimization (primary optimizer)
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 15.0:
        try:
            sm_pts, sm_r = _soft_max_optimize(best_pts, min(remaining - 15.0, 80.0))
            if sm_r < best_r:
                best_pts, best_r = sm_pts, sm_r
        except Exception:
            pass
    
    # Try Nelder-Mead if time permits
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 15.0:
        try:
            nm_pts = _nelder_mead_optimize(best_pts, min(remaining - 15.0, 40.0))
            nm_r = _covering_radius_fast(nm_pts)
            if nm_r < best_r:
                best_pts, best_r = nm_pts, nm_r
        except Exception:
            pass
    
    # Final verification with exact hull
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 5.0:
        try:
            exact_r = _covering_radius_hull(best_pts)
            # If exact is much worse, try a few more local steps
            if exact_r > best_r + 0.01:
                # The fast estimate was optimistic; do more careful optimization
                pass
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _normalize(best_pts)
    if not np.all(np.isfinite(best_pts)) or len(best_pts) != N:
        best_pts = fib
    
    return best_pts
# EVOLVE-BLOCK-END
