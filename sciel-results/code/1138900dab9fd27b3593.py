"""Circle packing in perimeter-4 rectangle: Differential Evolution + Lloyd's Voronoi Refinement."""
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.spatial import Voronoi
import scipy.spatial.distance as dist

N = 21
_TIME_LIMIT = 140.0


def _lloyd_refine(x, y, r, w, h, max_iter=100, min_radius=1e-8):
    """
    Lloyd's algorithm for circle packing:
    1. Compute Voronoi diagram of circle centers.
    2. For each circle, compute the inradius of its Voronoi cell (clipped to rectangle).
    3. Move center to the centroid of the clipped Voronoi cell.
    4. Set radius to the inradius.
    This guarantees non-overlapping circles at every iteration.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    for it in range(max_iter):
        # Build Voronoi diagram
        pts = np.column_stack([x, y])
        try:
            vor = Voronoi(pts)
        except Exception:
            break
        
        # Rectangle corners for clipping
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        for i in range(N):
            # Get the Voronoi region for point i
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 in region or len(region) == 0:
                # Unbounded region or invalid; fall back to distance to nearest neighbor
                continue
            
            # Get vertices of this Voronoi cell
            verts = vor.vertices[region]
            
            # Clip to rectangle [0, w] x [0, h]
            # We'll compute inradius and centroid of the clipped polygon
            # For simplicity, use a sampling-based approach for robustness:
            # Find the maximum inscribed circle center and radius within the cell ∩ rectangle
            
            # Method: the inradius of a convex polygon is the max over all points in polygon
            # of min distance to edges. For a Voronoi cell, the incenter is a vertex of the
            # Voronoi dual (the Delaunay triangulation circumcenter), but we can compute it
            # more robustly by finding the point in the polygon that maximizes the minimum
            # distance to all edges of the polygon (and rectangle boundaries).
            
            # Simpler robust approach: use the Voronoi vertex that is the "center" of the cell
            # Actually, for Lloyd's algorithm, we want the centroid of the cell and the
            # inradius. Let's compute both via polygon geometry.
            
            # Clip polygon to rectangle
            poly = verts
            # Sutherland-Hodgman clipping against rectangle
            def clip_polygon_to_rect(poly, w, h):
                if len(poly) == 0:
                    return poly
                # Clip against x >= 0
                def clip_edge(poly, edge_func, inside_func):
                    if len(poly) == 0:
                        return poly
                    out = []
                    n = len(poly)
                    for k in range(n):
                        cur = poly[k]
                        nxt = poly[(k + 1) % n]
                        cur_in = inside_func(cur)
                        nxt_in = inside_func(nxt)
                        if cur_in:
                            out.append(cur)
                            if not nxt_in:
                                # Exit: find intersection
                                # edge_func(p) = 0 is the boundary
                                # parametric: cur + t*(nxt-cur)
                                t = edge_func(cur) / (edge_func(cur) - edge_func(nxt)) if edge_func(nxt) != edge_func(cur) else 0.5
                                t = np.clip(t, 0, 1)
                                out.append(cur + t * (nxt - cur))
                        elif nxt_in:
                            # Enter
                            t = edge_func(cur) / (edge_func(cur) - edge_func(nxt)) if edge_func(nxt) != edge_func(cur) else 0.5
                            t = np.clip(t, 0, 1)
                            out.append(cur + t * (nxt - cur))
                            out.append(nxt)
                    return np.array(out) if out else poly
                
                poly = clip_edge(poly, lambda p: p[0], lambda p: p[0] >= -1e-10)
                poly = clip_edge(poly, lambda p: p[0] - w, lambda p: p[0] <= w + 1e-10)
                poly = clip_edge(poly, lambda p: p[1], lambda p: p[1] >= -1e-10)
                poly = clip_edge(poly, lambda p: p[1] - h, lambda p: p[1] <= h + 1e-10)
                return poly
            
            clipped = clip_polygon_to_rect(poly, w, h)
            
            if len(clipped) < 3:
                # Degenerate cell; use small radius
                new_r[i] = min_radius
                continue
            
            # Compute centroid
            centroid = np.mean(clipped, axis=0)
            new_x[i] = centroid[0]
            new_y[i] = centroid[1]
            
            # Compute inradius: minimum distance from centroid to each edge of clipped polygon
            min_dist = np.inf
            n_verts = len(clipped)
            for k in range(n_verts):
                p1 = clipped[k]
                p2 = clipped[(k + 1) % n_verts]
                # Distance from point to line segment
                d = dist.point_to_segment(centroid, p1, p2)
                min_dist = min(min_dist, d)
            
            new_r[i] = max(min_radius, min_dist * 0.999)  # Slight shrink for safety
        
        # Check convergence
        dx = np.max(np.abs(new_x - x))
        dy = np.max(np.abs(new_y - y))
        dr = np.max(np.abs(new_r - r))
        x, y, r = new_x, new_y, new_r
        
        if dx + dy + dr < 1e-10:
            break
    
    return x, y, r


def _evaluate_packing(x, y, r, w, h):
    """Evaluate sum of radii, checking validity."""
    if w <= 0.05 or h <= 0.05:
        return 0.0
    for i in range(N):
        if r[i] <= 0:
            return 0.0
        if x[i] < r[i] - 1e-9 or x[i] > w - r[i] + 1e-9:
            return 0.0
        if y[i] < r[i] - 1e-9 or y[i] > h - r[i] + 1e-9:
            return 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            rs = r[i] + r[j]
            if d2 < rs * rs - 1e-9:
                return 0.0
    return float(np.sum(r))


def _de_objective(params):
    """
    Objective for differential evolution.
    params: [w, x_0, y_0, ..., x_20, y_20]
    We use a grid-based initialization for radii, then Lloyd's refinement.
    To make DE work, we parameterize the aspect ratio and use Lloyd's
    to determine optimal radii from positions.
    """
    w = params[0]
    h = 2.0 - w
    if w < 0.3 or h < 0.3:
        return 1e6
    
    x = params[1::2]
    y = params[2::2]
    
    # Clip positions to valid range
    x = np.clip(x, 0.01, w - 0.01)
    y = np.clip(y, 0.01, h - 0.01)
    
    # Initial radii: small
    r = np.full(N, 0.02)
    
    # Run Lloyd's refinement
    try:
        x, y, r = _lloyd_refine(x, y, r, w, h, max_iter=50)
    except Exception:
        return 1e6
    
    val = _evaluate_packing(x, y, r, w, h)
    if val == 0.0:
        return 1e6
    return -val


def _grid_init(w, h, n):
    """Generate a grid of n points in a w x h rectangle."""
    cols = int(np.ceil(np.sqrt(n * w / h)))
    rows = int(np.ceil(n / cols))
    if rows < 1:
        rows = 1
    if cols < 1:
        cols = 1
    
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        row, col = divmod(i, cols)
        x[i] = (col + 0.5) * w / cols
        y[i] = (row + 0.5) * h / rows
    return x, y


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    best_value = 0.0
    best_x = None
    best_y = None
    best_r = None
    best_w = None
    
    # Phase 1: Differential Evolution over aspect ratio and positions
    # Parameter vector: [w, x_0, y_0, ..., x_20, y_20]
    # Bounds: w in [0.4, 1.6], x in [0, 2], y in [0, 2]
    bounds = [(0.4, 1.6)]
    for _ in range(N):
        bounds.append((0.0, 2.0))  # x
        bounds.append((0.0, 2.0))  # y
    
    # Use multiple DE runs with different seeds for diversity
    rng_seeds = [42, 123, 456, 789, 101, 202, 303, 404]
    
    for seed_idx, seed in enumerate(rng_seeds):
        if time.time() > deadline - 30:
            break
        
        # Generate initial population
        rng = np.random.default_rng(seed)
        
        # Aspect ratio candidates
        w_candidates = [0.8, 1.0, 1.2, 0.7, 1.3, 0.9, 1.1, 0.85, 1.15, 0.75, 1.25]
        
        # For each aspect ratio, create a grid-based initial population
        for w0 in w_candidates[:5]:
            if time.time() > deadline - 30:
                break
            
            h0 = 2.0 - w0
            x_init, y_init = _grid_init(w0, h0, N)
            
            # Create initial population around this grid
            pop = []
            for _ in range(15):
                w_p = np.clip(w0 + rng.normal(0, 0.1), 0.4, 1.6)
                h_p = 2.0 - w_p
                x_p = np.clip(x_init + rng.normal(0, 0.05, N), 0.01, w_p - 0.01)
                y_p = np.clip(y_init + rng.normal(0, 0.05, N), 0.01, h_p - 0.01)
                params = np.zeros(1 + 2 * N)
                params[0] = w_p
                params[1::2] = x_p
                params[2::2] = y_p
                pop.append(params)
            
            # Run DE
            try:
                res = differential_evolution(
                    _de_objective, bounds,
                    seed=seed,
                    maxiter=30,
                    popsize=15,
                    tol=1e-4,
                    polish=False,
                    init=np.array(pop),
                    updating='immediate',
                    mutation=(0.3, 0.9),
                    recombination=0.7,
                    constraints=()
                )
                
                if res.success or res.fun < 1e5:
                    w_opt = res.x[0]
                    h_opt = 2.0 - w_opt
                    x_opt = np.clip(res.x[1::2], 0.01, w_opt - 0.01)
                    y_opt = np.clip(res.x[2::2], 0.01, h_opt - 0.01)
                    r_init = np.full(N, 0.02)
                    
                    # Full Lloyd's refinement
                    x_ref, y_ref, r_ref = _lloyd_refine(x_opt, y_opt, r_init, w_opt, h_opt, max_iter=100)
                    val = _evaluate_packing(x_ref, y_ref, r_ref, w_opt, h_opt)
                    
                    if val > best_value:
                        best_value = val
                        best_x = x_ref.copy()
                        best_y = y_ref.copy()
                        best_r = r_ref.copy()
                        best_w = w_opt
                        
            except Exception:
                continue
    
    # Phase 2: Additional Lloyd's refinement with different initial configurations
    if best_x is not None and time.time() < deadline - 20:
        # Try to improve by perturbing positions and re-running Lloyd's
        rng = np.random.default_rng(999)
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            x_pert = best_x + rng.normal(0, 0.02, N)
            y_pert = best_y + rng.normal(0, 0.02, N)
            w_pert = np.clip(best_w + rng.normal(0, 0.05), 0.4, 1.6)
            h_pert = 2.0 - w_pert
            
            x_pert = np.clip(x_pert, 0.01, w_pert - 0.01)
            y_pert = np.clip(y_pert, 0.01, h_pert - 0.01)
            
            r_pert = np.full(N, 0.02)
            x_new, y_new, r_new = _lloyd_refine(x_pert, y_pert, r_pert, w_pert, h_pert, max_iter=80)
            val = _evaluate_packing(x_new, y_new, r_new, w_pert, h_pert)
            
            if val > best_value:
                best_value = val
                best_x = x_new.copy()
                best_y = y_new.copy()
                best_r = r_new.copy()
                best_w = w_pert
    
    # Phase 3: Final high-iteration Lloyd's polish
    if best_x is not None and time.time() < deadline - 5:
        h_final = 2.0 - best_w
        x_final, y_final, r_final = _lloyd_refine(best_x, best_y, best_r, best_w, h_final, max_iter=200)
        val = _evaluate_packing(x_final, y_final, r_final, best_w, h_final)
        
        if val > best_value:
            best_value = val
            best_x = x_final
            best_y = y_final
            best_r = r_final
    
    # Fallback
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Safety: slight shrink to ensure strict validity
    for _ in range(10):
        r_trial = best_r * (1 - 1e-10)
        valid = True
        w = best_w
        h = 2.0 - w
        for i in range(N):
            if r_trial[i] <= 0:
                valid = False
                break
            if best_x[i] < r_trial[i] - 1e-12 or best_x[i] > w - r_trial[i] + 1e-12:
                valid = False
                break
            if best_y[i] < r_trial[i] - 1e-12 or best_y[i] > h - r_trial[i] + 1e-12:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dx = best_x[i] - best_x[j]
                    dy = best_y[i] - best_y[j]
                    d2 = dx * dx + dy * dy
                    rs = r_trial[i] + r_trial[j]
                    if d2 < rs * rs - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            best_r = r_trial
        else:
            break
    
    return np.column_stack((best_x, best_y, best_r))
