# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _triangle_areas(pts, tri_idx):
    """Compute all triangle areas vectorized."""
    a = pts[tri_idx[:, 0]]
    b = pts[tri_idx[:, 1]]
    c = pts[tri_idx[:, 2]]
    area = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    return area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _get_tri_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=int)

def _compute_score(pts, tri_idx):
    """Compute score given points and precomputed triangle indices."""
    if not np.all(np.isfinite(pts)):
        return 0.0
    areas = _triangle_areas(pts, tri_idx)
    min_area = areas.min()
    hull_area = _hull_area(pts)
    if hull_area <= 1e-9:
        return 0.0
    return min_area / hull_area

def heilbronn_convex13() -> np.ndarray:
    n = 13
    tri_idx = _get_tri_indices(n)
    
    start_time = time.time()
    deadline = start_time + 570  # Safety margin for 630s limit

    # --- Structural Representation: Radial/Angular Perturbation of Regular Polygon ---
    # Instead of free 2D coordinates, we represent points as:
    # p_i = (r_i * cos(theta_i), r_i * sin(theta_i))
    # where theta_i = base_angle_i + delta_theta_i
    # and r_i = base_radius + delta_r_i
    # This reduces the effective search space by leveraging the symmetry of the optimal
    # configuration (which is often close to a regular polygon with slight perturbations).
    
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    base_radius = 1.0

    def _params_to_points(params):
        """Convert (delta_r, delta_theta) to 2D points."""
        dr = params[:n]
        dth = params[n:2*n]
        r = base_radius + dr
        th = base_angles + dth
        # Ensure r > 0 to avoid degeneracy near origin
        r = np.maximum(r, 0.1)
        x = r * np.cos(th)
        y = r * np.sin(th)
        return np.column_stack([x, y])

    def _points_to_params(pts):
        """Convert 2D points back to (delta_r, delta_theta) parameters."""
        r = np.linalg.norm(pts, axis=1)
        th = np.arctan2(pts[:, 1], pts[:, 0])
        dr = r - base_radius
        dth = th - base_angles
        # Unwrap angles to handle discontinuities
        dth = np.unwrap(dth)
        return np.concatenate([dr, dth])

    def _objective(params, beta):
        """
        Objective: Minimize -score.
        We use a smooth approximation of the min area to make it differentiable.
        """
        pts = _params_to_points(params)
        if not np.all(np.isfinite(pts)):
            return 1e6
        
        areas = _triangle_areas(pts, tri_idx)
        if np.any(areas < 1e-12):
            return 1e6
        
        hull_area = _hull_area(pts)
        if hull_area < 1e-9:
            return 1e6
        
        # Smooth min using log-sum-exp trick
        # We want to maximize min(areas) / hull_area
        # This is equivalent to minimizing -min(areas)/hull_area
        # Using soft min: soft_min(areas) approx min(areas) for large beta
        min_a = np.min(areas)
        shifted = beta * (areas - min_a)
        # If shifted is very negative, exp is 0. If very positive, exp is large.
        # Max shifted is 0 (for min_a).
        exp_terms = np.exp(-shifted)
        sum_exp = np.sum(exp_terms)
        
        soft_min_val = min_a - (1.0 / beta) * np.log(sum_exp)
        
        if soft_min_val <= 0:
            return 1e6
            
        score_approx = soft_min_val / hull_area
        return -score_approx

    # --- Initial Candidates ---
    # 1. Regular Polygon (zero perturbation)
    init_params = np.zeros(2 * n)
    
    # 2. Small random perturbations
    rng = np.random.default_rng(42)
    candidates = [init_params]
    for _ in range(15):
        dr = rng.normal(0, 0.05, n)
        dth = rng.normal(0, 0.05, n)
        candidates.append(np.concatenate([dr, dth]))
        
    # 3. Specific structural guesses: e.g., alternating radii
    for scale in [0.1, 0.2, 0.3]:
        dr = np.zeros(n)
        dr[::2] = scale
        dr[1::2] = -scale
        dth = np.zeros(n)
        candidates.append(np.concatenate([dr, dth]))
        
    # 4. Lattice-inspired perturbation
    # Generate a triangular lattice subset, project to polar, use as init
    lattice_pts = []
    s = 0.5
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i * s + (j % 2) * s * 0.5
            y = j * s * np.sqrt(3) / 2
            if x*x + y*y <= 1.0:
                lattice_pts.append([x, y])
    if len(lattice_pts) >= n:
        lattice_pts = np.array(lattice_pts[:n])
        # Normalize to unit radius roughly
        max_norm = np.linalg.norm(lattice_pts, axis=1).max()
        if max_norm > 0:
            lattice_pts /= max_norm
        lat_params = _points_to_params(lattice_pts)
        candidates.append(lat_params)

    # 5. Halton sequence mapped to unit disk
    def _halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    halton_pts = np.zeros((n, 2))
    for i in range(n):
        halton_pts[i, 0] = _halton(i + 1, 2)
        halton_pts[i, 1] = _halton(i + 1, 3)
    max_norm = np.linalg.norm(halton_pts, axis=1).max()
    halton_pts /= max_norm
    halton_pts *= 0.95
    hal_params = _points_to_params(halton_pts)
    candidates.append(hal_params)

    # 6. Squashed ellipses
    for sx, sy in [(1.2, 0.8), (0.8, 1.2), (1.1, 0.9), (0.9, 1.1)]:
        pts_sq = _params_to_points(np.zeros(2*n))
        pts_sq[:, 0] *= sx
        pts_sq[:, 1] *= sy
        sq_params = _points_to_params(pts_sq)
        candidates.append(sq_params)

    # --- Phase 1: Local Optimization in Parameter Space ---
    best_score = -1.0
    best_params = None
    best_pts = None

    # We try different betas. Small beta is smoother (global search), large beta is sharper (local refine).
    betas = [20, 50, 100]
    
    for cand in candidates:
        if time.time() > deadline - 150:
            break
        for beta in betas:
            if time.time() > deadline - 60:
                break
            try:
                res = minimize(
                    _objective, 
                    cand, 
                    args=(beta,),
                    method='L-BFGS-B', 
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                res_pts = _params_to_points(res.x)
                res_score = _compute_score(res_pts, tri_idx)
                if res_score > best_score:
                    best_score = res_score
                    best_params = res.x.copy()
                    best_pts = res_pts.copy()
            except Exception:
                pass

    # --- Phase 2: Subgradient Ascent on Active Triangles (in 2D space) ---
    # This handles the non-differentiable min area directly and is robust.
    # We switch to 2D coordinates here for the subgradient method as it's easier to define.
    
    if best_pts is not None and time.time() < deadline - 60:
        def _signed_areas(points):
            p1 = points[tri_idx[:, 0]]
            p2 = points[tri_idx[:, 1]]
            p3 = points[tri_idx[:, 2]]
            return (
                p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
                p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
                p3[:, 0] * (p1[:, 1] - p2[:, 1])
            )
        
        def _subgrad_ascent(points, max_iters=2000, base_lr=0.02):
            pts = points.copy()
            best_pts_local = pts.copy()
            best_score_local = _compute_score(pts, tri_idx)
            lr = base_lr
            patience = 0
            max_patience = 50
            
            for it in range(max_iters):
                if time.time() > deadline - 30:
                    break
                
                sa = _signed_areas(pts)
                areas = 0.5 * np.abs(sa)
                min_val = np.min(areas)
                
                # Include triangles near the minimum
                threshold = min_val * 1.02
                mask = areas <= threshold
                count = np.sum(mask)
                if count == 0:
                    break
                
                # Compute subgradient: average gradient of active minimum triangles
                grad = np.zeros((n, 2))
                for idx in np.where(mask)[0]:
                    i, j, k = tri_idx[idx]
                    S = sa[idx]
                    s = 1.0 if S >= 0 else -1.0
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    grad[i] += s * np.array([p2[1] - p3[1], p3[0] - p2[0]])
                    grad[j] += s * np.array([p3[1] - p1[1], p1[0] - p3[0]])
                    grad[k] += s * np.array([p1[1] - p2[1], p2[0] - p1[0]])
                grad /= count
                
                gn = np.linalg.norm(grad)
                if gn < 1e-15:
                    break
                grad /= gn
                
                # Try the step
                new_pts = pts + lr * grad
                ns = _compute_score(new_pts, tri_idx)
                if ns > best_score_local:
                    best_score_local = ns
                    best_pts_local = new_pts.copy()
                    pts = new_pts
                    patience = 0
                    lr = min(lr * 1.03, 0.05)
                else:
                    # Backtracking
                    ok = False
                    for f in [0.5, 0.25, 0.1]:
                        np_ = pts + lr * f * grad
                        ns2 = _compute_score(np_, tri_idx)
                        if ns2 > best_score_local:
                            best_score_local = ns2
                            best_pts_local = np_.copy()
                            pts = np_
                            ok = True
                            break
                    if not ok:
                        patience += 1
                        lr *= 0.85
                        if lr < 1e-9 or patience > max_patience:
                            break
            return best_pts_local, best_score_local

        refined_pts, refined_score = _subgrad_ascent(best_pts, max_iters=2000)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts

    # Fallback if optimization failed completely
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_pts

# EVOLVE-BLOCK-END
