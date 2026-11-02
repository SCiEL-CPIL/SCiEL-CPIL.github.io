"""Heilbronn triangles: place 11 points to maximize the minimum triangle area."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0
DEADLINE = 145  # safety margin from 150s limit

def _min_area_normalized(points):
    """Compute minimum triangle area normalized by the equilateral triangle area."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    
    cross = dx[:, :, None] * dy[:, None, :] - dy[:, :, None] * dx[:, None, :]
    areas = 0.5 * np.abs(cross)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return 0.0
    
    min_area = np.min(valid_areas)
    return min_area / TRI_AREA

def _cartesian_to_barycentric(points):
    """Convert Cartesian to barycentric coordinates."""
    x = points[:, 0]
    y = points[:, 1]
    w = y / S3H
    u = 1.0 - x - 0.5 * w
    v = x - 0.5 * w
    return np.column_stack([u, v, w])

def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates to Cartesian."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = S3H * bary[:, 2]
    return np.column_stack([x, y])

def _soft_min_objective_flat(flat_params, alpha=100.0):
    """
    Objective in barycentric coordinates using soft-min.
    flat_params: 2*N array of (u_i, v_i) for i=0..N-1, where w_i = 1 - u_i - v_i.
    """
    n = len(flat_params) // 2
    uv = flat_params.reshape(n, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    # Penalty for negative barycentric coords
    neg_u = np.maximum(0.0, -u)
    neg_v = np.maximum(0.0, -v)
    neg_w = np.maximum(0.0, -w)
    penalty = 1000.0 * (np.sum(neg_u**2) + np.sum(neg_v**2) + np.sum(neg_w**2))
    
    # Convert to Cartesian
    x = v + 0.5 * w
    y = S3H * w
    
    # Compute all triangle areas
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    cross = dx[:, :, None] * dy[:, None, :] - dy[:, :, None] * dx[:, None, :]
    areas = 0.5 * np.abs(cross)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return penalty
    
    # Soft minimum
    min_val = -np.log(np.sum(np.exp(-alpha * valid_areas))) / alpha
    return -min_val + penalty

def _min_area_objective_flat(flat_params):
    """Non-smooth objective: negative min area plus penalty."""
    n = len(flat_params) // 2
    uv = flat_params.reshape(n, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    neg_u = np.maximum(0.0, -u)
    neg_v = np.maximum(0.0, -v)
    neg_w = np.maximum(0.0, -w)
    penalty = 1000.0 * (np.sum(neg_u**2) + np.sum(neg_v**2) + np.sum(neg_w**2))
    
    x = v + 0.5 * w
    y = S3H * w
    
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    cross = dx[:, :, None] * dy[:, None, :] - dy[:, :, None] * dx[:, None, :]
    areas = 0.5 * np.abs(cross)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return penalty
    
    min_val = np.min(valid_areas) / TRI_AREA
    return -min_val + penalty

def _is_valid(points):
    """Check if all points are inside the triangle."""
    bary = _cartesian_to_barycentric(points)
    return np.all(bary >= -1e-9)

def _farthest_point_sampling(rng, n_points):
    """
    Generate n_points in the triangle using Farthest Point Sampling (FPS).
    """
    points = []
    while True:
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            break
    points.append(np.array([u + 0.5 * v, S3H * v]))
    
    for _ in range(n_points - 1):
        best_dist = -1
        best_pt = None
        
        for _ in range(200):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 1.0:
                    break
            cand = np.array([u + 0.5 * v, S3H * v])
            
            dists = np.linalg.norm(points - cand, axis=1)
            min_d = np.min(dists)
            
            if min_d > best_dist:
                best_dist = min_d
                best_pt = cand
        
        points.append(best_pt)
        
    return np.array(points)

def _hexagonal_initialization(n_points, rng):
    """
    Generate a hexagonal grid of points inside the triangle.
    """
    points = []
    best_pts = None
    best_score = -1
    
    for scale in [0.1, 0.15, 0.2, 0.25, 0.3]:
        pts = []
        k = 0
        while True:
            y = k * np.sqrt(3)/2 * scale
            if y > S3H:
                break
            j = -10
            while j < 10:
                x = (j + (k % 2) * 0.5) * scale
                w = y / S3H
                u = 1.0 - x - 0.5 * w
                v = x - 0.5 * w
                if u >= -1e-9 and v >= -1e-9 and w >= -1e-9:
                    pts.append([x, y])
                j += 1
            k += 1
        
        pts = np.array(pts)
        if len(pts) == 0:
            continue
        
        if len(pts) >= n_points:
            indices = rng.choice(len(pts), n_points, replace=False)
            selected = pts[indices]
        else:
            selected = pts.copy()
            while len(selected) < n_points:
                u, v = rng.random(), rng.random()
                if u + v < 1.0:
                    w = 1 - u - v
                    x = v + 0.5 * w
                    y = S3H * w
                    selected = np.vstack([selected, [[x, y]]])
        
        score = _min_area_normalized(selected)
        if score > best_score:
            best_score = score
            best_pts = selected
    
    if best_pts is None:
        return _farthest_point_sampling(rng, n_points)
    
    return best_pts

def _simulated_annealing(points, n_iter=20000, T_start=0.05, T_end=0.001, rng=None):
    """
    Run Simulated Annealing to maximize the minimum triangle area.
    points: (N, 2) array of Cartesian coordinates.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    current_points = points.copy()
    current_score = _min_area_normalized(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    n = len(points)
    
    for i in range(n_iter):
        T = T_start * (T_end / T_start) ** (i / n_iter)
        
        idx = rng.randint(n)
        cand_points = current_points.copy()
        
        step = 0.05 * (1.0 + 0.5 * (T / T_start))
        dx = rng.normal(0, step)
        dy = rng.normal(0, step)
        
        cand_points[idx, 0] += dx
        cand_points[idx, 1] += dy
        
        bary = _cartesian_to_barycentric(cand_points)
        bary = np.maximum(bary, 0)
        s = np.sum(bary, axis=1, keepdims=True)
        s[s == 0] = 1
        bary = bary / s
        cand_points = _barycentric_to_cartesian(bary)
        
        cand_score = _min_area_normalized(cand_points)
        
        if cand_score >= current_score:
            current_points = cand_points
            current_score = cand_score
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand_points.copy()
        else:
            delta = current_score - cand_score
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_points = cand_points
                    current_score = cand_score
    
    return best_points, best_score

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    best_points = None
    best_score = -float('inf')
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Simulated Annealing from diverse starts
    if elapsed() < 60:
        starts = []
        
        # FPS start
        fps_pts = _farthest_point_sampling(rng, N)
        starts.append(fps_pts)
        
        # Hexagonal start
        hex_pts = _hexagonal_initialization(N, rng)
        starts.append(hex_pts)
        
        # Random starts
        for _ in range(3):
            u = rng.random(N)
            v = rng.random(N)
            s = u + v
            s[s == 0] = 1.0
            scale = np.minimum(1.0, 0.99 / s)
            u = u * scale
            v = v * scale
            w = 1 - u - v
            x = v + 0.5 * w
            y = S3H * w
            pts = np.column_stack([x, y])
            starts.append(pts)
        
        for start_pts in starts:
            if elapsed() > DEADLINE - 30:
                break
            
            try:
                sa_pts, sa_score = _simulated_annealing(start_pts, n_iter=5000, T_start=0.05, T_end=0.001, rng=rng)
                if sa_score > best_score:
                    best_score = sa_score
                    best_points = sa_pts.copy()
            except Exception:
                pass
    
    # Phase 2: Generate diverse initial guesses
    initial_guesses = []
    
    # FPS-based start
    fps_pts = _farthest_point_sampling(rng, N)
    bary = _cartesian_to_barycentric(fps_pts)
    initial_guesses.append(np.column_stack([bary[:, 0], bary[:, 1]]).flatten())
    
    # Random starts
    for _ in range(5):
        u = rng.random(N)
        v = rng.random(N)
        s = u + v
        s[s == 0] = 1.0
        scale = np.minimum(1.0, 0.99 / s)
        u = u * scale
        v = v * scale
        initial_guesses.append(np.column_stack([u, v]).flatten())
    
    # Add DE result if found
    if best_points is not None:
        bary = _cartesian_to_barycentric(best_points)
        initial_guesses.append(np.column_stack([bary[:, 0], bary[:, 1]]).flatten())
    
    # Phase 3: Powell optimization from each initial guess
    for i, x0 in enumerate(initial_guesses):
        if elapsed() > DEADLINE - 25:
            break
        
        def obj_powell(x):
            return _min_area_objective_flat(x)
        
        try:
            res = minimize(
                obj_powell,
                x0,
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
            )
        except Exception:
            continue
        
        if elapsed() > DEADLINE - 20:
            break
        
        uv = res.x.reshape(N, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1.0 - u - v
        
        if np.min(u) < -1e-6 or np.min(v) < -1e-6 or np.min(w) < -1e-6:
            u = np.maximum(u, 0)
            v = np.maximum(v, 0)
            w = 1.0 - u - v
            if np.min(w) < -1e-6:
                continue
        
        x_cart = v + 0.5 * w
        y_cart = S3H * w
        points = np.column_stack([x_cart, y_cart])
        
        if not _is_valid(points):
            continue
        
        score = _min_area_normalized(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Phase 4: Polish with L-BFGS-B using soft-min objective
    if best_points is not None and elapsed() < DEADLINE - 10:
        bary = _cartesian_to_barycentric(best_points)
        x0 = np.column_stack([bary[:, 0], bary[:, 1]]).flatten()
        
        def obj_refine(x):
            return _soft_min_objective_flat(x, alpha=200.0)
        
        try:
            res = minimize(
                obj_refine,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            uv = res.x.reshape(N, 2)
            u = uv[:, 0]
            v = uv[:, 1]
            w = 1.0 - u - v
            
            if np.min(u) >= -1e-6 and np.min(v) >= -1e-6 and np.min(w) >= -1e-6:
                x_cart = v + 0.5 * w
                y_cart = S3H * w
                points = np.column_stack([x_cart, y_cart])
                
                if _is_valid(points):
                    score = _min_area_normalized(points)
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
        except Exception:
            pass
    
    # Phase 5: Nelder-Mead polish in Cartesian space
    if best_points is not None and elapsed() < DEADLINE - 3:
        def cart_objective(flat):
            pts = flat.reshape(N, 2)
            bary = _cartesian_to_barycentric(pts)
            penalty = 1000.0 * (np.sum(np.maximum(0, -bary[:, 0])**2) +
                               np.sum(np.maximum(0, -bary[:, 1])**2) +
                               np.sum(np.maximum(0, -bary[:, 2])**2))
            min_a = _min_area_normalized(pts)
            return -min_a + penalty
        
        try:
            res = minimize(
                cart_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            polished = res.x.reshape(N, 2)
            if _is_valid(polished):
                score = _min_area_normalized(polished)
                if score > best_score:
                    best_score = score
                    best_points = polished.copy()
        except Exception:
            pass
    
    # Phase 6: Perturb and re-optimize
    if best_points is not None and elapsed() < DEADLINE - 5:
        for s in range(5):
            if elapsed() > DEADLINE - 5:
                break
            perturbed = best_points + rng.normal(0, 0.02, size=(N, 2))
            
            # Project to triangle
            bary = _cartesian_to_barycentric(perturbed)
            bary = np.maximum(bary, 0)
            s_sum = np.sum(bary, axis=1, keepdims=True)
            s_sum[s_sum == 0] = 1
            bary = bary / s_sum
            
            points = _barycentric_to_cartesian(bary)
            
            bary2 = _cartesian_to_barycentric(points)
            x0 = np.column_stack([bary2[:, 0], bary2[:, 1]]).flatten()
            
            try:
                res = minimize(
                    _min_area_objective_flat,
                    x0,
                    method='Powell',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                uv = res.x.reshape(N, 2)
                u = uv[:, 0]
                v = uv[:, 1]
                w = 1.0 - u - v
                
                if np.min(u) >= -1e-6 and np.min(v) >= -1e-6 and np.min(w) >= -1e-6:
                    x_cart = v + 0.5 * w
                    y_cart = S3H * w
                    pts = np.column_stack([x_cart, y_cart])
                    
                    if _is_valid(pts):
                        score = _min_area_normalized(pts)
                        if score > best_score:
                            best_score = score
                            best_points = pts.copy()
            except Exception:
                continue
    
    # Fallback
    if best_points is None:
        best_points = _farthest_point_sampling(rng, N)
    
    return best_points
