import numpy as np
import time
from scipy.spatial import Voronoi, Delaunay
from scipy.optimize import minimize

N = 32

def _is_valid(xy, r, tol=1e-9):
    n = N
    if np.any(r < -tol):
        return False
    if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
        return False
    if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
        return False
    dx = xy[:, 0][:, None] - xy[:, 0][None, :]
    dy = xy[:, 1][:, None] - xy[:, 1][None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    iu = np.triu_indices(n, k=1)
    if np.any(dist[iu] < min_dist[iu] - tol):
        return False
    return True

def _score(xy, r):
    return float(np.sum(r))

def _voronoi_inradius(xy):
    """For each site, compute the inradius of its Voronoi cell (distance to nearest edge/vertex)."""
    n = len(xy)
    # Add boundary points to make Voronoi bounded
    # We add points far outside to ensure the Voronoi diagram covers [0,1]^2
    margin = 10.0
    extra = np.array([
        [-margin, -margin], [1+margin, -margin],
        [-margin, 1+margin], [1+margin, 1+margin],
        [-margin, 0.5], [1+margin, 0.5],
        [0.5, -margin], [0.5, 1+margin]
    ])
    pts = np.vstack([xy, extra])
    try:
        vor = Voronoi(pts)
    except Exception:
        return np.full(n, 0.05)
    
    inradii = np.zeros(n)
    for i in range(n):
        region_id = vor.point_region[i]
        region = vor.regions[region_id]
        if -1 in region:
            inradii[i] = 0.05  # unbounded, shouldn't happen with our setup
            continue
        # Get the polygon vertices for this region
        verts = vor.vertices[region]
        if len(verts) < 3:
            inradii[i] = 0.05
            continue
        # Compute inradius: min distance from site to each edge of the cell
        center = xy[i]
        min_dist = np.inf
        for k in range(len(verts)):
            v1 = verts[k]
            v2 = verts[(k+1) % len(verts)]
            # Distance from center to line segment v1-v2
            d = _point_segment_dist(center, v1, v2)
            if d < min_dist:
                min_dist = d
        # Also check distance to boundary of unit square
        min_dist = min(min_dist, center[0], 1-center[0], center[1], 1-center[1])
        inradii[i] = min_dist
    return inradii

def _point_segment_dist(p, a, b):
    """Distance from point p to segment ab."""
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-15)
    t = np.clip(t, 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def _voronoi_objective(params):
    """Objective: -sum(r_i * inradius_i) + penalties. We want to maximize sum of radii
    subject to r_i <= inradius_i and boundary constraints."""
    n = N
    xy = params[:2*n].reshape(n, 2)
    # We don't optimize radii directly; we use r_i = alpha * inradius_i
    # This is handled in the outer loop
    return 0.0  # Placeholder

def _refine_with_lagrangian(xy, r, max_iter=300):
    """
    Use a penalty-based approach where we maximize sum(r) subject to:
    1. r_i <= dist to boundary for circle i
    2. r_i + r_j <= dist(i,j) for all pairs
    """
    n = N
    
    def objective(params):
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        r = np.maximum(r, 0.001)
        
        # Boundary violation
        b_viol = 0.0
        for i in range(n):
            b_viol += max(0, r[i] - xy[i,0])**2
            b_viol += max(0, xy[i,0] + r[i] - 1)**2
            b_viol += max(0, r[i] - xy[i,1])**2
            b_viol += max(0, xy[i,1] + r[i] - 1)**2
        
        # Overlap violation
        dx = xy[:,0][:,None] - xy[:,0][None,:]
        dy = xy[:,1][:,None] - xy[:,1][None,:]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:,None] + r[None,:]
        iu = np.triu_indices(n, k=1)
        overlap = np.maximum(0, min_dist[iu] - dist[iu])
        o_viol = np.sum(overlap**2)
        
        return -np.sum(r) + 500*b_viol + 500*o_viol
    
    def gradient(params):
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        r = np.maximum(r, 0.001)
        
        gxy = np.zeros_like(xy)
        gr = np.full(n, -1.0)
        
        # Boundary
        for i in range(n):
            v = r[i] - xy[i,0]
            if v > 0:
                gxy[i,0] += -1000*v
                gr[i] += 1000*v
            v = xy[i,0] + r[i] - 1
            if v > 0:
                gxy[i,0] += 1000*v
                gr[i] += 1000*v
            v = r[i] - xy[i,1]
            if v > 0:
                gxy[i,1] += -1000*v
                gr[i] += 1000*v
            v = xy[i,1] + r[i] - 1
            if v > 0:
                gxy[i,1] += 1000*v
                gr[i] += 1000*v
        
        # Overlap
        dx = xy[:,0][:,None] - xy[:,0][None,:]
        dy = xy[:,1][:,None] - xy[:,1][None,:]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        min_dist = r[:,None] + r[None,:]
        iu = np.triu_indices(n, k=1)
        overlap = np.maximum(0, min_dist[iu] - dist[iu])
        
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            o = overlap[idx]
            if o > 0:
                d = dist[i,j]
                factor = 1000.0 * o / d
                gxy[i,0] += factor * dx[i,j]
                gxy[i,1] += factor * dy[i,j]
                gxy[j,0] -= factor * dx[i,j]
                gxy[j,1] -= factor * dy[i,j]
                gr[i] += 1000.0 * o
                gr[j] += 1000.0 * o
        
        return np.concatenate([gxy.flatten(), gr])
    
    params = np.concatenate([xy.flatten(), r])
    try:
        res = minimize(objective, params, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})
        return res.x
    except Exception:
        return params

def _expand_radii(xy, max_iter=1000, tol=1e-12):
    """
    Given fixed centers, find maximum radii such that all constraints are satisfied.
    Use iterative water-filling: repeatedly set each r_i to min(inradius_i, min_j (dist_ij - r_j))
    """
    n = N
    r = np.zeros(n)
    
    # Compute pairwise distances
    dx = xy[:,0][:,None] - xy[:,0][None,:]
    dy = xy[:,1][:,None] - xy[:,1][None,:]
    dist = np.sqrt(dx**2 + dy**2)
    
    # Boundary distances
    bdist = np.minimum(np.minimum(xy[:,0], 1-xy[:,0]), np.minimum(xy[:,1], 1-xy[:,1]))
    
    for it in range(max_iter):
        new_r = np.zeros(n)
        for i in range(n):
            # Max radius for circle i given all other circles
            max_r = bdist[i]
            for j in range(n):
                if i != j:
                    max_r = min(max_r, dist[i,j] - r[j])
            new_r[i] = max(0, max_r)
        
        if np.allclose(new_r, r, atol=tol):
            r = new_r
            break
        # Damped update to avoid oscillation
        r = 0.5 * r + 0.5 * new_r
    
    return r

def _simulated_annealing(xy, r, rng, n_iters=5000, T0=0.05, T1=0.001, alpha=0.999):
    """
    Simulated annealing on centers and radii jointly.
    Moves: perturb one circle's center or radius.
    Accept if it improves sum(r) or with probability exp(-delta/T) if it decreases.
    Constraint: maintain validity.
    """
    n = N
    best_xy = xy.copy()
    best_r = r.copy()
    best_score = _score(best_xy, best_r)
    
    cur_xy = xy.copy()
    cur_r = r.copy()
    cur_score = _score(cur_xy, cur_r)
    
    T = T0
    for it in range(n_iters):
        T *= alpha
        if T < T1:
            T = T1
        
        # Random move
        i = rng.randint(n)
        move_type = rng.randint(3)
        
        new_xy = cur_xy.copy()
        new_r = cur_r.copy()
        
        if move_type == 0:
            # Perturb center
            new_xy[i,0] += rng.normal(0, 0.01)
            new_xy[i,1] += rng.normal(0, 0.01)
        elif move_type == 1:
            # Perturb radius
            new_r[i] += rng.normal(0, 0.005)
            new_r[i] = max(0.001, min(0.5, new_r[i]))
        else:
            # Swap two circles' positions
            j = rng.randint(n)
            if j != i:
                new_xy[i], new_xy[j] = new_xy[j].copy(), new_xy[i].copy()
                new_r[i], new_r[j] = new_r[j], new_r[i]
        
        # Clip to boundary
        for k in range(n):
            new_xy[k,0] = np.clip(new_xy[k,0], new_r[k], 1-new_r[k])
            new_xy[k,1] = np.clip(new_xy[k,1], new_r[k], 1-new_r[k])
        
        # Check validity
        if _is_valid(new_xy, new_r):
            new_score = _score(new_xy, new_r)
            delta = new_score - cur_score
            if delta > 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                cur_xy = new_xy
                cur_r = new_r
                cur_score = new_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_xy = cur_xy.copy()
                    best_r = cur_r.copy()
        else:
            # Try to repair by shrinking
            for shrink in range(10):
                new_r *= 0.9
                if _is_valid(new_xy, new_r):
                    new_score = _score(new_xy, new_r)
                    delta = new_score - cur_score
                    if delta > 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                        cur_xy = new_xy
                        cur_r = new_r
                        cur_score = new_score
                        if cur_score > best_score:
                            best_score = cur_score
                            best_xy = cur_xy.copy()
                            best_r = cur_r.copy()
                    break
    
    return best_xy, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    rng = np.random.RandomState(42)
    
    best_xy = None
    best_r = None
    best_score = -np.inf
    
    def update_best(xy, r):
        nonlocal best_xy, best_r, best_score
        if _is_valid(xy, r):
            s = _score(xy, r)
            if s > best_score:
                best_score = s
                best_xy = xy.copy()
                best_r = r.copy()
    
    # Phase 1: Generate diverse initial configurations
    configs = []
    
    # Grid 6x6 (use 32 of 36)
    side = 6
    xy = np.zeros((N, 2))
    for i in range(N):
        row = i // side
        col = i % side
        xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
    configs.append(xy.copy())
    
    # Perturbed grid
    xy2 = xy.copy()
    xy2 += rng.uniform(-0.03, 0.03, size=(N, 2))
    configs.append(xy2)
    
    # Hexagonal
    xy3 = np.zeros((N, 2))
    for i in range(N):
        row = i // 6
        col = i % 6
        x = (col + 0.5)/6 + (0.25 if row % 2 == 1 else 0)
        y = (row + 0.5)/6
        xy3[i] = [x, y]
    configs.append(xy3)
    
    # 8x4 grid
    xy4 = np.zeros((N, 2))
    for i in range(N):
        row = i // 8
        col = i % 8
        xy4[i] = [(col + 0.5)/8, (row + 0.5)/4]
    configs.append(xy4)
    
    # 4x8 grid
    xy5 = np.zeros((N, 2))
    for i in range(N):
        row = i // 4
        col = i % 4
        xy5[i] = [(col + 0.5)/4, (row + 0.5)/8]
    configs.append(xy5)
    
    # Random
    for _ in range(5):
        xy_r = rng.uniform(0.1, 0.9, size=(N, 2))
        configs.append(xy_r)
    
    # Phase 2: For each config, expand radii and refine
    for cfg_idx, xy_init in enumerate(configs):
        if time.time() > deadline - 30:
            break
        
        # Expand radii
        r_init = _expand_radii(xy_init, max_iter=500)
        update_best(xy_init, r_init)
        
        # Refine with L-BFGS
        params = np.concatenate([xy_init.flatten(), r_init])
        refined = _refine_with_lagrangian(xy_init, r_init, max_iter=200)
        xy_ref = refined[:2*N].reshape(N, 2)
        r_ref = refined[2*N:]
        update_best(xy_ref, r_ref)
        
        # Re-expand radii after refinement
        r_re = _expand_radii(xy_ref, max_iter=500)
        update_best(xy_ref, r_re)
        
        # Simulated annealing
        if time.time() < deadline - 20:
            xy_sa, r_sa = _simulated_annealing(xy_ref, r_re, rng, n_iters=3000, T0=0.03, T1=0.001)
            update_best(xy_sa, r_sa)
            
            # Re-expand after SA
            r_sa2 = _expand_radii(xy_sa, max_iter=500)
            update_best(xy_sa, r_sa2)
    
    # Phase 3: Voronoi-based approach
    if best_xy is not None and time.time() < deadline - 20:
        # Use current best as starting point
        xy_v = best_xy.copy()
        r_v = best_r.copy()
        
        # Iterate: compute inradii, set r_i = inradius_i, recompute
        for it in range(200):
            if time.time() > deadline - 10:
                break
            inradii = _voronoi_inradius(xy_v)
            new_r = inradii * 0.999  # Small margin
            new_r = np.clip(new_r, 0.001, 0.5)
            
            if _is_valid(xy_v, new_r):
                update_best(xy_v, new_r)
            
            # Perturb centers slightly
            xy_v = best_xy + rng.normal(0, 0.005, size=(N, 2))
            xy_v = np.clip(xy_v, 0.01, 0.99)
            r_v = _expand_radii(xy_v, max_iter=300)
            update_best(xy_v, r_v)
        
        # Final refinement
        if best_xy is not None:
            params = np.concatenate([best_xy.flatten(), best_r])
            refined = _refine_with_lagrangian(best_xy, best_r, max_iter=500)
            xy_f = refined[:2*N].reshape(N, 2)
            r_f = refined[2*N:]
            update_best(xy_f, r_f)
            
            r_f2 = _expand_radii(xy_f, max_iter=1000)
            update_best(xy_f, r_f2)
    
    # Phase 4: Final polish
    if best_xy is not None and time.time() < deadline - 5:
        # Multiple random perturbations + refine
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            xy_p = best_xy + rng.normal(0, 0.002, size=(N, 2))
            r_p = best_r.copy()
            xy_p = np.clip(xy_p, 0.005, 0.995)
            
            params = np.concatenate([xy_p.flatten(), r_p])
            refined = _refine_with_lagrangian(xy_p, r_p, max_iter=200)
            xy_r = refined[:2*N].reshape(N, 2)
            r_r = refined[2*N:]
            update_best(xy_r, r_r)
            
            r_r2 = _expand_radii(xy_r, max_iter=500)
            update_best(xy_r, r_r2)
    
    if best_xy is not None:
        # Final clip to ensure validity
        r_final = np.clip(best_r, 0, 0.5)
        xy_final = best_xy.copy()
        xy_final[:,0] = np.clip(xy_final[:,0], r_final, 1-r_final)
        xy_final[:,1] = np.clip(xy_final[:,1], r_final, 1-r_final)
        
        if _is_valid(xy_final, r_final):
            return np.column_stack([xy_final, r_final])
        else:
            # Shrink until valid
            for _ in range(100):
                r_final *= 0.95
                if _is_valid(xy_final, r_final):
                    return np.column_stack([xy_final, r_final])
    
    # Fallback
    side = 6
    cells = []
    for i in range(N):
        row = i // side
        col = i % side
        cells.append([(col+0.5)/side, (row+0.5)/side, 0.5/side - 0.002])
    return np.array(cells)
