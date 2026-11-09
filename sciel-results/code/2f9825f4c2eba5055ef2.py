# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
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

    # --- Generate diverse initial points ---
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 2. Halton sequence mapped to unit disk
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
    # Scale to fit in unit disk
    max_norm = np.linalg.norm(halton_pts, axis=1).max()
    halton_pts /= max_norm
    halton_pts *= 0.95
    
    # 3. Random points in unit disk
    rng = np.random.default_rng(42)
    random_pts = np.zeros((n, 2))
    for i in range(n):
        r = np.sqrt(rng.random()) * 0.95
        theta = rng.random() * 2 * np.pi
        random_pts[i] = [r * np.cos(theta), r * np.sin(theta)]
    
    # 4. Lattice-inspired: triangular lattice subset
    lattice_pts = []
    s = 0.5
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i * s + (j % 2) * s * 0.5
            y = j * s * np.sqrt(3) / 2
            if x*x + y*y <= 1.0:
                lattice_pts.append([x, y])
    lattice_pts = np.array(lattice_pts[:n]) if len(lattice_pts) >= n else np.array(lattice_pts)
    if len(lattice_pts) < n:
        lattice_pts = init_pts.copy()
    
    initial_candidates = [init_pts, halton_pts, random_pts, lattice_pts]
    
    # --- Local refinement using Nelder-Mead on full 2D coordinates ---
    def _nm_objective(x):
        pts = x.reshape(n, 2)
        if np.any(~np.isfinite(pts)):
            return 1e10
        try:
            areas = _triangle_areas(pts, tri_idx)
            a_min = np.min(areas)
            if a_min < 1e-15:
                return 1e10
            hull = ConvexHull(pts)
            a_hull = hull.volume
            if a_hull < 1e-15:
                return 1e10
            return -(a_min / a_hull)
        except Exception:
            return 1e10
    
    best_score = -1.0
    best_pts = None
    
    # Try each initial candidate with Nelder-Mead
    for init in initial_candidates:
        if time.time() > deadline - 120:
            break
        try:
            res = minimize(_nm_objective, init.flatten(), method='nelder-mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 
                                   'adaptive': True, 'initial_simplex': None})
            res_pts = res.x.reshape(n, 2)
            res_score = _compute_score(res_pts, tri_idx)
            if res_score > best_score:
                best_score = res_score
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    # --- Subgradient ascent on active triangles ---
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
        
        def _subgrad_ascent(points, max_iters=2000, base_lr=0.01):
            pts = points.copy()
            best_pts_local = pts.copy()
            best_score_local = _compute_score(pts, tri_idx)
            lr = base_lr
            patience = 0
            max_patience = 80
            
            for it in range(max_iters):
                if time.time() > deadline - 30:
                    break
                
                sa = _signed_areas(pts)
                areas = 0.5 * np.abs(sa)
                min_val = np.min(areas)
                
                # Include triangles near the minimum
                threshold = min_val * 1.05
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
    
    # --- Final polish with Nelder-Mead ---
    if best_pts is not None and time.time() < deadline - 10:
        try:
            res = minimize(_nm_objective, best_pts.flatten(), method='nelder-mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 
                                   'adaptive': True})
            res_pts = res.x.reshape(n, 2)
            res_score = _compute_score(res_pts, tri_idx)
            if res_score > best_score:
                best_score = res_score
                best_pts = res_pts
        except Exception:
            pass

    # Fallback if optimization failed completely
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_pts

# EVOLVE-BLOCK-END
