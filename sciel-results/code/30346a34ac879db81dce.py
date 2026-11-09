"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize, dual_annealing, linprog
from scipy.spatial import Delaunay

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)
_n_tris = len(_idx_i)

# Precompute point masks: which triangles involve each point
_point_masks = []
for p in range(N):
    mask = ((_idx_i == p) | (_idx_j == p) | (_idx_k == p))
    _point_masks.append(mask)

# Precompute edge pairs for each triangle (for area computation)
# For triangle (i,j,k): edges are (i,j), (i,k), (j,k)
# Area = 0.5 * |cross(pj-pi, pk-pi)|


def _min_area_fast(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    cross = np.abs(dx1 * dy2 - dy1 * dx2)
    return 0.5 * np.min(cross)


def _min_area_and_idx(points):
    """Return (min_area, index_of_worst_triangle)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    cross = np.abs(dx1 * dy2 - dy1 * dx2)
    areas = 0.5 * cross
    idx = int(np.argmin(areas))
    return areas[idx], idx


def _all_areas(points):
    """Compute all triangle areas."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    cross = np.abs(dx1 * dy2 - dy1 * dx2)
    return 0.5 * cross


def _inside_penalty(points, margin=0.002):
    """Smooth penalty for points outside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    viol1 = np.maximum(0.0, -y + margin)
    viol2 = np.maximum(0.0, y - np.sqrt(3) * x + margin)
    viol3 = np.maximum(0.0, y - np.sqrt(3) * (1.0 - x) + margin)
    return np.sum(viol1**2) + np.sum(viol2**2) + np.sum(viol3**2)


def _is_inside(points, tol=1e-8):
    """Check if all points are inside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    return (np.all(y >= -tol) and
            np.all(y <= np.sqrt(3) * x + tol) and
            np.all(y <= np.sqrt(3) * (1.0 - x) + tol))


def _objective(params):
    """Objective: negative min area + boundary penalty."""
    points = params.reshape(N, 2)
    pen = _inside_penalty(points)
    if pen > 1e-6:
        return -_min_area_fast(points) + 10000.0 * pen
    return -_min_area_fast(points)


def _objective_constrained(params):
    """Objective with hard constraint checking (for dual annealing)."""
    points = params.reshape(N, 2)
    if not _is_inside(points, tol=1e-6):
        return 1000.0
    return -_min_area_fast(points)


def _init_symmetric():
    """Generate a good symmetric starting configuration based on known near-optimal patterns."""
    points = np.zeros((N, 2))
    # Use a pattern inspired by the best known configurations
    # Row structure: 4-3-3-1 or similar
    row_ys = [0.07 * H, 0.28 * H, 0.52 * H, 0.75 * H]
    row_ns = [4, 3, 3, 1]
    
    pi = 0
    for y, n_pts in zip(row_ys, row_ns):
        x_left = 0.5 * y / H
        x_right = 1.0 - 0.5 * y / H
        width = x_right - x_left
        
        if n_pts == 1:
            points[pi, 0] = 0.5
            points[pi, 1] = y
            pi += 1
        else:
            # Distribute points with some margin from edges
            margin_frac = 0.12
            usable = width * (1 - 2 * margin_frac)
            start = x_left + width * margin_frac
            if n_pts == 2:
                xs = [start, start + usable]
            else:
                xs = start + np.linspace(0, usable, n_pts)
            for x in xs:
                points[pi, 0] = x
                points[pi, 1] = y
                pi += 1
    
    return points


def _init_symmetric_v2():
    """Alternative symmetric initialization with different row structure."""
    points = np.zeros((N, 2))
    # Try 3-4-3-1 pattern
    row_ys = [0.10 * H, 0.32 * H, 0.55 * H, 0.78 * H]
    row_ns = [3, 4, 3, 1]
    
    pi = 0
    for y, n_pts in zip(row_ys, row_ns):
        x_left = 0.5 * y / H
        x_right = 1.0 - 0.5 * y / H
        width = x_right - x_left
        
        if n_pts == 1:
            points[pi, 0] = 0.5
            points[pi, 1] = y
            pi += 1
        else:
            margin_frac = 0.10
            usable = width * (1 - 2 * margin_frac)
            start = x_left + width * margin_frac
            xs = start + np.linspace(0, usable, n_pts)
            for x in xs:
                points[pi, 0] = x
                points[pi, 1] = y
                pi += 1
    
    return points


def _init_random(rng):
    """Generate random points inside the triangle."""
    points = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1:
                points[i, 0] = v + 0.5 * (1 - u - v)
                points[i, 1] = H * (1 - u - v)
                break
    return points


def _init_grid(rng):
    """Generate points on a barycentric grid with random perturbation."""
    candidates = []
    n_div = 7
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            u = i / n_div
            v = j / n_div
            w = 1.0 - u - v
            if w >= -1e-9:
                w = max(0.0, w)
                x = v * 1.0 + w * 0.5
                y = w * H
                if x >= -1e-9 and x <= 1 + 1e-9 and y >= -1e-9 and y <= H + 1e-9:
                    candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    # Select N points that maximize minimum pairwise distance (greedy farthest-point)
    if len(candidates) > N:
        # Start with a random point
        start_idx = rng.integers(0, len(candidates))
        selected = [candidates[start_idx]]
        for _ in range(N - 1):
            dists = np.array([np.min(np.linalg.norm(candidates - s, axis=1)) for s in selected])
            # Actually compute min distance to any selected point
            min_dists = np.full(len(candidates), np.inf)
            for s in selected:
                d = np.linalg.norm(candidates - s, axis=1)
                min_dists = np.minimum(min_dists, d)
            next_idx = np.argmax(min_dists)
            selected.append(candidates[next_idx])
        pts = np.array(selected)
    else:
        pts = candidates.copy()
        while len(pts) < N:
            u = rng.random()
            v = rng.random()
            if u + v <= 1:
                px = v + 0.5 * (1 - u - v)
                py = H * (1 - u - v)
                pts = np.vstack([pts, [[px, py]]])
        pts = pts[:N]
    
    pts += rng.normal(0, 0.015, size=pts.shape)
    return _clamp_to_triangle(pts)


def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle."""
    points = points.copy()
    for i in range(N):
        x, y = points[i]
        y = max(0.0, min(H, y))
        x_min = y / np.sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        if x_min > x_max:
            x, y = 0.5, 0.0
        else:
            x = max(x_min, min(x_max, x))
        points[i] = [x, y]
    return points


def _local_refine(points, time_limit):
    """
    Iterative local refinement: repeatedly optimize the worst triangle's vertices,
    then try pair moves and single-point moves for diversity.
    """
    start = time.time()
    best_points = points.copy()
    best_score = _min_area_fast(best_points)
    no_improve_count = 0
    max_no_improve = 15
    
    for iteration in range(2000):
        if time.time() > time_limit:
            break
        if no_improve_count >= max_no_improve:
            break
        
        min_area, worst_idx = _min_area_and_idx(best_points)
        i, j, k = int(_idx_i[worst_idx]), int(_idx_j[worst_idx]), int(_idx_k[worst_idx])
        
        # Strategy 1: Optimize the 3 vertices of the worst triangle
        combined_mask = _point_masks[i] | _point_masks[j] | _point_masks[k]
        tri_points = best_points[[i, j, k]].copy()
        
        def _tri_obj(tri_params):
            tp = tri_params.reshape(3, 2)
            x, y = tp[:, 0], tp[:, 1]
            if np.any(y < -1e-7) or np.any(y > np.sqrt(3)*x + 1e-7) or np.any(y > np.sqrt(3)*(1-x) + 1e-7):
                return 1e6
            full = best_points.copy()
            full[i], full[j], full[k] = tp[0], tp[1], tp[2]
            areas = _all_areas(full)
            return -np.min(areas[combined_mask])
        
        improved = False
        try:
            res = minimize(_tri_obj, tri_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 2000})
            new_tri = res.x.reshape(3, 2)
            if _is_inside(new_tri, tol=1e-10):
                test_points = best_points.copy()
                test_points[i], test_points[j], test_points[k] = new_tri[0], new_tri[1], new_tri[2]
                test_score = _min_area_fast(test_points)
                if test_score > best_score + 1e-14:
                    best_points = test_points
                    best_score = test_score
                    improved = True
        except Exception:
            pass
        
        # Strategy 2: Try optimizing pairs of points (random pairs)
        if not improved and iteration % 2 == 0:
            a, b = int(rng_pair := np.random.randint(0, N)), int(np.random.randint(0, N))
            if a != b:
                pair_mask = _point_masks[a] | _point_masks[b]
                pair_points = best_points[[a, b]].copy()
                
                def _pair_obj(pair_params):
                    pp = pair_params.reshape(2, 2)
                    x, y = pp[:, 0], pp[:, 1]
                    if np.any(y < -1e-7) or np.any(y > np.sqrt(3)*x + 1e-7) or np.any(y > np.sqrt(3)*(1-x) + 1e-7):
                        return 1e6
                    full = best_points.copy()
                    full[a], full[b] = pp[0], pp[1]
                    areas = _all_areas(full)
                    return -np.min(areas[pair_mask])
                
                try:
                    res = minimize(_pair_obj, pair_points.flatten(), method='Nelder-Mead',
                                  options={'maxiter': 400, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 1000})
                    new_pair = res.x.reshape(2, 2)
                    if _is_inside(new_pair, tol=1e-10):
                        test_points = best_points.copy()
                        test_points[a], test_points[b] = new_pair[0], new_pair[1]
                        test_score = _min_area_fast(test_points)
                        if test_score > best_score + 1e-14:
                            best_points = test_points
                            best_score = test_score
                            improved = True
                except Exception:
                    pass
        
        # Strategy 3: Try moving a single point
        if not improved and iteration % 3 == 0:
            a = int(np.random.randint(0, N))
            single_mask = _point_masks[a]
            single_point = best_points[a].copy()
            
            def _single_obj(sp):
                p = sp.reshape(1, 2)
                x, y = p[0, 0], p[0, 1]
                if y < -1e-7 or y > np.sqrt(3)*x + 1e-7 or y > np.sqrt(3)*(1-x) + 1e-7:
                    return 1e6
                full = best_points.copy()
                full[a] = p[0]
                areas = _all_areas(full)
                return -np.min(areas[single_mask])
            
            try:
                res = minimize(_single_obj, single_point, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 800})
                new_point = res.x.reshape(1, 2)
                if _is_inside(new_point, tol=1e-10):
                    test_points = best_points.copy()
                    test_points[a] = new_point[0]
                    test_score = _min_area_fast(test_points)
                    if test_score > best_score + 1e-14:
                        best_points = test_points
                        best_score = test_score
                        improved = True
            except Exception:
                pass
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    return best_points, best_score


def _nelder_mead_full(points, time_limit):
    """Run Nelder-Mead on all 22 coordinates for global polish."""
    start = time.time()
    try:
        res = minimize(_objective, points.flatten(), method='Nelder-Mead',
                      options={'maxiter': 15000, 'xatol': 1e-13, 'fatol': 1e-15,
                               'maxfev': 25000})
        new_points = res.x.reshape(N, 2)
        if _is_inside(new_points, tol=1e-10):
            return new_points
    except Exception:
        pass
    return points


def _differential_evolution_search(time_limit, seed=42):
    """Run differential evolution for global search."""
    start = time.time()
    bounds = [(0, 1), (0, H)] * N
    remaining = time_limit - time.time()
    if remaining < 5:
        return None
    try:
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=50,
            popsize=30,
            tol=1e-10,
            seed=seed,
            polish=True,
            init='latinhypercube',
            mutation=(0.5, 1.0),
            recombination=0.7,
            updating='immediate'
        )
        de_points = result.x.reshape(N, 2)
        de_points = _clamp_to_triangle(de_points)
        if _is_inside(de_points, tol=1e-8):
            return de_points
    except Exception:
        pass
    return None


def _dual_annealing_search(time_limit, seed=42):
    """Run dual annealing for global search."""
    start = time.time()
    bounds = [(0, 1), (0, H)] * N
    remaining = time_limit - time.time()
    if remaining < 5:
        return None
    try:
        da_result = dual_annealing(
            _objective_constrained,
            bounds=bounds,
            seed=seed,
            maxiter=100,
            restarts=5,
            initial_temp=5000.0,
            visit=[1.2, 1.4],
            clip=[-1, 1]
        )
        da_points = da_result.x.reshape(N, 2)
        da_points = _clamp_to_triangle(da_points)
        if _is_inside(da_points, tol=1e-8):
            return da_points
    except Exception:
        pass
    return None


def _refine_worst_triangles(points, time_limit, n_worst=5):
    """Refine multiple worst triangles simultaneously for faster convergence."""
    start = time.time()
    best_points = points.copy()
    best_score = _min_area_fast(best_points)
    
    for iteration in range(500):
        if time.time() > time_limit:
            break
        
        areas = _all_areas(best_points)
        # Get indices of worst triangles
        worst_indices = np.argsort(areas)[:n_worst]
        
        # Collect unique points involved in worst triangles
        involved = set()
        for wi in worst_indices:
            involved.add(int(_idx_i[wi]))
            involved.add(int(_idx_j[wi]))
            involved.add(int(_idx_k[wi]))
        
        if len(involved) > 8:
            involved = list(involved)[:8]
        else:
            involved = list(involved)
        
        if not involved:
            break
        
        involved_arr = np.array(involved)
        # Compute combined mask
        combined_mask = np.zeros(_n_tris, dtype=bool)
        for p in involved:
            combined_mask |= _point_masks[p]
        
        sub_points = best_points[involved_arr].copy()
        
        def _sub_obj(sub_params):
            sp = sub_params.reshape(len(involved), 2)
            x, y = sp[:, 0], sp[:, 1]
            if np.any(y < -1e-7) or np.any(y > np.sqrt(3)*x + 1e-7) or np.any(y > np.sqrt(3)*(1-x) + 1e-7):
                return 1e6
            full = best_points.copy()
            for idx, p_idx in enumerate(involved_arr):
                full[p_idx] = sp[idx]
            areas = _all_areas(full)
            return -np.min(areas[combined_mask])
        
        improved = False
        try:
            res = minimize(_sub_obj, sub_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 3000})
            new_sub = res.x.reshape(len(involved), 2)
            if _is_inside(new_sub, tol=1e-10):
                test_points = best_points.copy()
                for idx, p_idx in enumerate(involved_arr):
                    test_points[p_idx] = new_sub[idx]
                test_score = _min_area_fast(test_points)
                if test_score > best_score + 1e-14:
                    best_points = test_points
                    best_score = test_score
                    improved = True
        except Exception:
            pass
        
        if not improved:
            # Try individual worst triangle
            wi = worst_indices[0]
            i, j, k = int(_idx_i[wi]), int(_idx_j[wi]), int(_idx_k[wi])
            tri_mask = _point_masks[i] | _point_masks[j] | _point_masks[k]
            tri_points = best_points[[i, j, k]].copy()
            
            def _tri_obj(tri_params):
                tp = tri_params.reshape(3, 2)
                x, y = tp[:, 0], tp[:, 1]
                if np.any(y < -1e-7) or np.any(y > np.sqrt(3)*x + 1e-7) or np.any(y > np.sqrt(3)*(1-x) + 1e-7):
                    return 1e6
                full = best_points.copy()
                full[i], full[j], full[k] = tp[0], tp[1], tp[2]
                areas = _all_areas(full)
                return -np.min(areas[tri_mask])
            
            try:
                res = minimize(_tri_obj, tri_points.flatten(), method='Nelder-Mead',
                              options={'maxiter': 600, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 1500})
                new_tri = res.x.reshape(3, 2)
                if _is_inside(new_tri, tol=1e-10):
                    test_points = best_points.copy()
                    test_points[i], test_points[j], test_points[k] = new_tri[0], new_tri[1], new_tri[2]
                    test_score = _min_area_fast(test_points)
                    if test_score > best_score + 1e-14:
                        best_points = test_points
                        best_score = test_score
            except Exception:
                pass
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Multiple diverse initializations
    inits = [
        _init_symmetric(),
        _init_symmetric_v2(),
        _init_random(rng),
        _init_grid(rng),
    ]
    
    for init in inits:
        init = _clamp_to_triangle(init)
        score = _min_area_fast(init)
        if score > best_score:
            best_score = score
            best_points = init.copy()
    
    # Phase 2: Quick local refinement from best init
    if time.time() < deadline - 120:
        t_limit = time.time() + 25
        refined_pts, refined_score = _local_refine(best_points, t_limit)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts
    
    # Phase 3: Global search with differential evolution
    if time.time() < deadline - 80:
        t_limit = time.time() + 35
        de_points = _differential_evolution_search(t_limit, seed=42)
        if de_points is not None:
            de_score = _min_area_fast(de_points)
            if de_score > best_score:
                best_score = de_score
                best_points = de_points.copy()
            # Polish DE result
            if time.time() < deadline - 40:
                t_limit2 = time.time() + 20
                polished = _nelder_mead_full(best_points, t_limit2)
                pol_score = _min_area_fast(polished)
                if pol_score > best_score:
                    best_score = pol_score
                    best_points = polished
    
    # Phase 4: Local refinement of current best
    if time.time() < deadline - 40:
        t_limit = time.time() + 20
        refined_pts, refined_score = _local_refine(best_points, t_limit)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts
    
    # Phase 5: Dual annealing for additional global exploration
    if time.time() < deadline - 25:
        t_limit = time.time() + 15
        da_points = _dual_annealing_search(t_limit, seed=123)
        if da_points is not None:
            da_score = _min_area_fast(da_points)
            if da_score > best_score:
                best_score = da_score
                best_points = da_points.copy()
                # Quick polish
                if time.time() < deadline - 8:
                    polished = _nelder_mead_full(best_points, time.time() + 7)
                    pol_score = _min_area_fast(polished)
                    if pol_score > best_score:
                        best_score = pol_score
                        best_points = polished
    
    # Phase 6: Final refinement of worst triangles
    if time.time() < deadline - 12:
        t_limit = time.time() + 10
        refined_pts, refined_score = _refine_worst_triangles(best_points, t_limit)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts
    
    # Phase 7: Final high-precision Nelder-Mead polish
    if time.time() < deadline - 5:
        t_limit = time.time() + 4
        polished = _nelder_mead_full(best_points, t_limit)
        pol_score = _min_area_fast(polished)
        if pol_score > best_score:
            best_score = pol_score
            best_points = polished
    
    if best_points is None:
        best_points = _init_symmetric()
    
    return _clamp_to_triangle(best_points)
