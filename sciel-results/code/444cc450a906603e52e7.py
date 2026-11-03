import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

# Pre-compute all C(11,3) = 165 triangle index combinations
_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)
N_TRI = len(_indices)


def _bary_to_cart(bary):
    """Convert barycentric (N,3) to cartesian (N,2)."""
    pts = np.empty((N, 2))
    pts[:, 0] = bary[:, 1] + 0.5 * bary[:, 2]
    pts[:, 1] = H * bary[:, 2]
    return pts


def _cart_to_bary(pts):
    """Convert cartesian (N,2) to barycentric (N,3)."""
    bary = np.empty((N, 3))
    bary[:, 2] = pts[:, 1] / H
    bary[:, 1] = pts[:, 0] - 0.5 * bary[:, 2]
    bary[:, 0] = 1.0 - bary[:, 1] - bary[:, 2]
    return bary


def _project_bary(bary, eps=1e-12):
    """Project barycentric coordinates into the valid simplex (all >= 0, sum = 1)."""
    bary = bary.copy()
    bary = np.maximum(bary, 0.0)
    s = bary.sum(axis=1, keepdims=True)
    s = np.maximum(s, eps)
    bary = bary / s
    return bary


def _all_areas_cart(pts):
    """Compute all 165 triangle areas from cartesian points (N,2)."""
    i = _idx_arr[:, 0]
    j = _idx_arr[:, 1]
    k = _idx_arr[:, 2]
    det = (pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) - (pts[j,1]-pts[i,1])*(pts[k,0]-pts[i,0])
    return 0.5 * np.abs(det)


def _all_areas_bary(bary):
    """Compute all 165 triangle areas from barycentric coordinates."""
    pts = _bary_to_cart(bary)
    return _all_areas_cart(pts)


def _min_area_bary(bary):
    return float(np.min(_all_areas_bary(bary)))


def _min_area_cart(pts):
    return float(np.min(_all_areas_cart(pts)))


def _soft_min_bary(bary, beta=80.0):
    """Smooth min of all 165 areas using log-sum-exp."""
    areas = _all_areas_bary(bary)
    min_a = np.min(areas)
    shifted = -beta * (areas - min_a)
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return min_a - log_sum / beta


def _delaunay_tris(pts):
    """Return list of Delaunay triangle index tuples (i,j,k) for interior points."""
    outer = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, H]])
    all_pts = np.vstack([pts, outer])
    try:
        tri = Delaunay(all_pts)
    except Exception:
        return []
    result = []
    for simplex in tri.simplices:
        if all(int(v) < N for v in simplex):
            result.append(tuple(int(v) for v in simplex))
    return result


def _delaunay_active_set_optimize(bary, time_limit=15.0):
    """
    Delaunay-based active-set optimization.
    Key insight: the minimum-area triangle must be a Delaunay triangle.
    We iteratively:
      1. Compute Delaunay triangulation
      2. Find the minimum-area Delaunay triangle
      3. Move its vertices to increase its area (equalize areas)
      4. Re-compute Delaunay (topology may change)
    """
    start_t = time.time()
    current = bary.copy()
    best = bary.copy()
    best_score = _min_area_bary(current)
    
    for it in range(300):
        if time.time() - start_t > time_limit:
            break
        
        pts = _bary_to_cart(current)
        dtri = _delaunay_tris(pts)
        
        if len(dtri) == 0:
            # Fallback: use all triangles
            areas = _all_areas_bary(current)
            min_a = float(np.min(areas))
            if min_a > best_score:
                best_score = min_a
                best = current.copy()
            improved = False
            for idx in range(N):
                for _ in range(5):
                    db = np.random.randn(3) * 0.01
                    new_bary = _project_bary(current + db[np.newaxis, :])
                    na = _min_area_bary(new_bary)
                    if na > min_a + 1e-12:
                        current = new_bary
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
            continue
        
        # Find min-area Delaunay triangle
        dareas = np.array([0.5 * abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) - (pts[j,1]-pts[i,1])*(pts[k,0]-pts[i,0])) for i,j,k in dtri])
        min_da = float(np.min(dareas))
        min_tri_idx = int(np.argmin(dareas))
        i, j, k = dtri[min_tri_idx]
        
        # Check if this is actually the global min
        global_min = _min_area_bary(current)
        if global_min > best_score:
            best_score = global_min
            best = current.copy()
        
        # Try to increase the area of the critical Delaunay triangle
        improved = False
        for move_idx in [i, j, k]:
            pi = current[move_idx]
            
            # Compute direction to increase this triangle's area
            if move_idx == i:
                p1, p2 = pts[j], pts[k]
            elif move_idx == j:
                p1, p2 = pts[k], pts[i]
            else:
                p1, p2 = pts[i], pts[j]
            
            edge = p2 - p1
            perp = np.array([-edge[1], edge[0]])
            
            # Determine sign: we want to move away from the edge
            v1 = pts[move_idx] - p1
            v2 = pts[move_idx] - p2
            det = v1[0]*v2[1] - v1[1]*v2[0]
            # The perpendicular direction should point away from the edge
            # Check which side the point is on
            cross = edge[0]*v1[1] - edge[1]*v1[0]
            direction = perp if cross > 0 else -perp
            
            direction = direction / (np.linalg.norm(direction) + 1e-15)
            
            # Convert cartesian direction to barycentric direction
            db2 = direction[1] / H
            db1 = direction[0] - 0.5 * db2
            db0 = -db1 - db2
            bary_dir = np.array([db0, db1, db2])
            
            for step in [0.03, 0.015, 0.008, 0.004, 0.002, 0.001, 0.0005]:
                new_bary = current.copy()
                new_bary[move_idx] = pi + step * bary_dir
                new_bary = _project_bary(new_bary)
                na = _min_area_bary(new_bary)
                if na > global_min + 1e-12:
                    current = new_bary
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try random perturbations of involved points
            for idx in [i, j, k]:
                for _ in range(15):
                    db = np.random.randn(3) * 0.005
                    new_bary = _project_bary(current + db[np.newaxis, :])
                    na = _min_area_bary(new_bary)
                    if na > global_min + 1e-12:
                        current = new_bary
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best, best_score


def _nelder_mead_bary(bary, time_limit=10.0, beta=80.0):
    """Nelder-Mead optimization in barycentric space with soft-min objective."""
    start_t = time.time()
    params = bary.flatten().copy()
    
    def obj(params):
        bary_p = params.reshape(N, 3)
        bary_p = _project_bary(bary_p)
        return -_soft_min_bary(bary_p, beta)
    
    result = minimize(
        obj,
        params,
        method='Nelder-Mead',
        options={
            'maxiter': 2000,
            'xatol': 1e-10,
            'fatol': 1e-12,
            'adaptive': True
        }
    )
    
    new_bary = _project_bary(result.x.reshape(N, 3))
    score = _min_area_bary(new_bary)
    old_score = _min_area_bary(bary)
    
    if score > old_score:
        return new_bary, score
    return bary, old_score


def _push_apart_bary(bary, beta=60.0):
    """
    Compute gradient-like direction in barycentric space that pushes points
    apart, weighted by how close each triangle is to the minimum.
    """
    pts = _bary_to_cart(bary)
    areas = _all_areas_cart(pts)
    min_a = np.min(areas)
    
    # Soft weights
    weights = np.exp(-beta * (areas - min_a))
    total = np.sum(weights)
    if total < 1e-30:
        return np.zeros((N, 3))
    weights /= total
    
    # Compute gradient in cartesian space
    grad_cart = np.zeros((N, 2))
    for t in range(N_TRI):
        i, j, k = int(_idx_arr[t, 0]), int(_idx_arr[t, 1]), int(_idx_arr[t, 2])
        w = weights[t]
        pi, pj, pk = pts[i], pts[j], pts[k]
        
        det = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0])
        sign = np.sign(det)
        
        # Gradient of signed area w.r.t. each point
        gpi = np.array([-(pk[1]-pj[1]), pk[0]-pj[0]])
        gpj = np.array([-(pi[1]-pk[1]), pi[0]-pk[0]])
        gpk = np.array([-(pj[1]-pi[1]), pj[0]-pi[0]])
        
        grad_cart[i] += w * sign * gpi
        grad_cart[j] += w * sign * gpj
        grad_cart[k] += w * sign * gpk
    
    # Convert cartesian gradient to barycentric gradient
    grad_bary = np.zeros((N, 3))
    for m in range(N):
        db2 = grad_cart[m, 1] / H
        db1 = grad_cart[m, 0] - 0.5 * db2
        db0 = -db1 - db2
        grad_bary[m] = [db0, db1, db2]
    
    return grad_bary


def _gradient_ascent_bary(bary, time_limit=10.0, beta=60.0):
    """Gradient ascent in barycentric space using push-apart gradient."""
    start_t = time.time()
    current = bary.copy()
    best = bary.copy()
    best_score = _min_area_bary(current)
    
    for it in range(150):
        if time.time() - start_t > time_limit:
            break
        
        current_score = _min_area_bary(current)
        if current_score > best_score:
            best_score = current_score
            best = current.copy()
        
        grad = _push_apart_bary(current, beta)
        
        improved = False
        for step in [0.05, 0.025, 0.01, 0.005, 0.002, 0.001, 0.0005]:
            new_bary = _project_bary(current + step * grad)
            na = _min_area_bary(new_bary)
            if na > current_score + 1e-12:
                current = new_bary
                improved = True
                break
        
        if not improved:
            break
    
    return best, best_score


def _fine_local_search(bary, time_limit=10.0):
    """Fine-grained local search on the minimum-area triangle's vertices."""
    start_t = time.time()
    current = bary.copy()
    best = bary.copy()
    best_score = _min_area_bary(current)
    
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for ss in step_sizes:
        if time.time() - start_t > time_limit:
            break
        
        for _ in range(250):
            if time.time() - start_t > time_limit:
                break
            
            pts = _bary_to_cart(current)
            areas = _all_areas_cart(pts)
            min_a = float(np.min(areas))
            
            if min_a > best_score:
                best_score = min_a
                best = current.copy()
            
            # Find involved points in min triangles
            threshold = min_a + 1e-10
            min_idx = np.where(areas <= threshold)[0]
            involved = set()
            for idx in min_idx:
                involved.add(int(_idx_arr[idx, 0]))
                involved.add(int(_idx_arr[idx, 1]))
                involved.add(int(_idx_arr[idx, 2]))
            
            improved = False
            for idx in sorted(involved):
                if time.time() - start_t > time_limit:
                    break
                for _ in range(20):
                    db = np.random.randn(3) * ss
                    new_bary = _project_bary(current + db[np.newaxis, :])
                    na = _min_area_bary(new_bary)
                    if na > min_a + 1e-12:
                        current = new_bary
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
    
    return best, best_score


def _sobol_init(n, seed=42):
    """Generate n quasi-random points in the triangle using Sobol sequence."""
    from scipy.stats import qmc
    rng = qmc.Sobol(d=2, seed=seed)
    n_gen = n * 8
    samples = rng.random(n_gen)
    u = samples[:, 0]
    v = samples[:, 1]
    mask = (u >= 0) & (v >= 0) & (u + v <= 1)
    valid_u = u[mask]
    valid_v = v[mask]
    while len(valid_u) < n:
        more = rng.random(n_gen)
        u2, v2 = more[:, 0], more[:, 1]
        mask2 = (u2 >= 0) & (v2 >= 0) & (u2 + v2 <= 1)
        valid_u = np.concatenate([valid_u, u2[mask2]])
        valid_v = np.concatenate([valid_v, v2[mask2]])
    u = valid_u[:n]
    v = valid_v[:n]
    bary = np.empty((n, 3))
    bary[:, 0] = 1.0 - u - v
    bary[:, 1] = u
    bary[:, 2] = v
    return bary


def _symmetric_init():
    """Create a symmetric initial configuration exploiting the triangle's 3-fold symmetry."""
    cx, cy = 0.5, H / 3.0
    bary = np.zeros((N, 3))
    
    # Center point
    bary[0] = [1/3, 1/3, 1/3]
    
    # 3 points on one ring
    r1 = 0.15
    angles = [0, 2*np.pi/3, 4*np.pi/3]
    for m in range(3):
        x = cx + r1 * np.cos(angles[m])
        y = cy + r1 * np.sin(angles[m])
        bary[m+1] = _cart_to_bary(np.array([[x, y]]))[0]
    
    # 3 points on another ring
    r2 = 0.3
    angles2 = [np.pi/3, np.pi, 5*np.pi/3]
    for m in range(3):
        x = cx + r2 * np.cos(angles2[m])
        y = cy + r2 * np.sin(angles2[m])
        bary[m+4] = _cart_to_bary(np.array([[x, y]]))[0]
    
    # 3 points near the edges
    r3 = 0.35
    angles3 = [np.pi/6, 5*np.pi/6, 3*np.pi/2]
    for m in range(3):
        x = cx + r3 * np.cos(angles3[m])
        y = cy + r3 * np.sin(angles3[m])
        bary[m+7] = _cart_to_bary(np.array([[x, y]]))[0]
    
    return _project_bary(bary)


def _random_init(seed=0):
    """Random initialization in barycentric coordinates."""
    np.random.seed(seed)
    bary = np.random.dirichlet([1.0, 1.0, 1.0], size=N)
    return _project_bary(bary)


def _optimize_one(bary, deadline):
    """Run the full optimization pipeline on one initial configuration."""
    remaining = deadline - time.time()
    if remaining < 5:
        return bary, _min_area_bary(bary)
    
    # Step 1: Nelder-Mead with soft min (broad exploration)
    t1 = min(8.0, remaining * 0.15)
    bary, score = _nelder_mead_bary(bary, time_limit=t1, beta=60.0)
    remaining = deadline - time.time()
    if remaining < 5:
        return bary, score
    
    # Step 2: Gradient ascent (push apart)
    t2 = min(5.0, remaining * 0.12)
    bary, score = _gradient_ascent_bary(bary, time_limit=t2, beta=60.0)
    remaining = deadline - time.time()
    if remaining < 5:
        return bary, score
    
    # Step 3: Delaunay-based active-set optimization (the key structural method)
    t3 = min(15.0, remaining * 0.3)
    bary, score = _delaunay_active_set_optimize(bary, time_limit=t3)
    remaining = deadline - time.time()
    if remaining < 5:
        return bary, score
    
    # Step 4: Nelder-Mead again with higher beta (sharper)
    t4 = min(8.0, remaining * 0.18)
    bary, score = _nelder_mead_bary(bary, time_limit=t4, beta=120.0)
    remaining = deadline - time.time()
    if remaining < 5:
        return bary, score
    
    # Step 5: Fine local search
    t5 = min(10.0, remaining * 0.22)
    bary, score = _fine_local_search(bary, time_limit=t5)
    remaining = deadline - time.time()
    if remaining < 3:
        return bary, score
    
    # Step 6: Another Delaunay pass
    t6 = min(10.0, remaining * 0.25)
    bary, score = _delaunay_active_set_optimize(bary, time_limit=t6)
    
    return bary, score


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin: 15s before the 150s kill
    
    np.random.seed(42)
    
    best_bary = None
    best_score = -np.inf
    
    # Generate diverse initializations
    inits = []
    
    # Symmetric initialization (exploits 3-fold symmetry)
    inits.append(_symmetric_init())
    
    # Sobol quasi-random
    for seed in [42, 123, 777, 2024]:
        inits.append(_sobol_init(N, seed=seed))
    
    # Random Dirichlet
    for seed in [1, 2, 3, 4]:
        inits.append(_random_init(seed=seed))
    
    # Structured grid-like
    grid_bary = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.6, 0.2, 0.2],
        [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6],
        [0.5, 0.25, 0.25],
        [0.25, 0.5, 0.25],
        [0.25, 0.25, 0.5],
        [1/3, 1/3, 1/3],
        [0.4, 0.4, 0.2],
    ])
    inits.append(_project_bary(grid_bary))
    
    # Optimize each initialization
    for i, init in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 15:
            break
        
        bary, score = _optimize_one(init, deadline)
        
        if score > best_score:
            best_score = score
            best_bary = bary.copy()
    
    # Phase 2: Refine the best result with additional passes
    if best_bary is not None:
        remaining = deadline - time.time()
        if remaining > 10:
            # Additional Delaunay refinement
            t = min(12.0, remaining * 0.35)
            bary, score = _delaunay_active_set_optimize(best_bary, time_limit=t)
            if score > best_score:
                best_score = score
                best_bary = bary.copy()
        
        remaining = deadline - time.time()
        if remaining > 8:
            # Fine local search
            t = min(8.0, remaining * 0.35)
            bary, score = _fine_local_search(best_bary, time_limit=t)
            if score > best_score:
                best_score = score
                best_bary = bary.copy()
        
        remaining = deadline - time.time()
        if remaining > 8:
            # Nelder-Mead with very high beta for final polish
            t = min(8.0, remaining * 0.4)
            bary, score = _nelder_mead_bary(best_bary, time_limit=t, beta=200.0)
            if score > best_score:
                best_score = score
                best_bary = bary.copy()
        
        remaining = deadline - time.time()
        if remaining > 5:
            # Final Delaunay pass
            t = min(5.0, remaining * 0.6)
            bary, score = _delaunay_active_set_optimize(best_bary, time_limit=t)
            if score > best_score:
                best_score = score
                best_bary = bary.copy()
    
    # Fallback
    if best_bary is None:
        best_bary = _sobol_init(N, seed=0)
    
    # Final projection and conversion to cartesian
    best_bary = _project_bary(best_bary)
    pts = _bary_to_cart(best_bary)
    
    # Ensure points are strictly inside the triangle (tiny margin)
    eps = 1e-10
    for m in range(N):
        x, y = pts[m, 0], pts[m, 1]
        if y < eps:
            y = eps
        if y > SQRT3 * x - eps:
            y = SQRT3 * x - eps
        if y > SQRT3 * (1 - x) - eps:
            y = SQRT3 * (1 - x) - eps
        if x < eps:
            x = eps
        if x > 1 - eps:
            x = 1 - eps
        pts[m, 0] = x
        pts[m, 1] = y
    
    return pts
