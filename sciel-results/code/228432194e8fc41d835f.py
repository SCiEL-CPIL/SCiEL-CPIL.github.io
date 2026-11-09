# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _get_tri_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=int)

def _triangle_areas(pts, tri_idx):
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
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _compute_score(pts, tri_idx):
    if not np.all(np.isfinite(pts)):
        return 0.0
    areas = _triangle_areas(pts, tri_idx)
    min_area = areas.min()
    hull_area = _hull_area(pts)
    if hull_area <= 1e-9:
        return 0.0
    return min_area / hull_area

def _grad_min_area(pts, tri_idx, active_tol=1e-3):
    """
    Compute gradient of the minimum triangle area.
    Returns: (min_area, gradient_vector)
    """
    areas = _triangle_areas(pts, tri_idx)
    min_area = areas.min()
    
    # Identify active triangles: those with area close to min
    # Using a small tolerance to include nearly-minimal triangles for stability
    active_mask = areas < (min_area * (1 + active_tol) + 1e-9)
    active_indices = np.where(active_mask)[0]
    
    if len(active_indices) == 0:
        return min_area, np.zeros_like(pts.flatten())
    
    grad_sum = np.zeros_like(pts)
    
    for idx in active_indices:
        i, j, k = tri_idx[idx]
        p_i = pts[i]
        p_j = pts[j]
        p_k = pts[k]
        
        # Area = 0.5 * |det(p_j - p_i, p_k - p_i)|
        # Let v1 = p_j - p_i, v2 = p_k - p_i
        # Area = 0.5 * |v1_x v2_y - v1_y v2_x|
        
        v1 = p_j - p_i
        v2 = p_k - p_i
        det_val = v1[0] * v2[1] - v1[1] * v2[0]
        
        # Gradient of |det| is sign(det) * grad(det)
        # det = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
        # d(det)/dx_i = -(y_k - y_i) + (y_j - y_i) = y_j - y_k
        # d(det)/dy_i = -(x_k - x_i) + (x_j - x_i) = x_j - x_k
        # d(det)/dx_j = y_k - y_i
        # d(det)/dy_j = -(x_k - x_i)
        # d(det)/dx_k = -(y_j - y_i)
        # d(det)/dy_k = x_j - x_i
        
        sign_det = np.sign(det_val)
        if sign_det == 0:
            sign_det = 1.0
            
        grad_det = np.zeros_like(pts)
        
        grad_det[i, 0] = sign_det * (p_j[1] - p_k[1])
        grad_det[i, 1] = sign_det * (p_j[0] - p_k[0])
        
        grad_det[j, 0] = sign_det * (p_k[1] - p_i[1])
        grad_det[j, 1] = sign_det * (-(p_k[0] - p_i[0]))
        
        grad_det[k, 0] = sign_det * (-(p_j[1] - p_i[1]))
        grad_det[k, 1] = sign_det * (p_j[0] - p_i[0])
        
        grad_sum += grad_det * 0.5 # Factor of 0.5 from area definition
    
    # Average the gradients of active triangles
    # This approximates the subgradient of the min function
    grad_min = grad_sum / len(active_indices)
    
    return min_area, grad_min.flatten()

def _grad_hull_area(pts):
    """
    Compute gradient of the convex hull area.
    Returns: (hull_area, gradient_vector)
    """
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        hull_verts = hull.vertices
    except Exception:
        return 0.0, np.zeros_like(pts.flatten())
        
    grad_hull = np.zeros_like(pts)
    
    # The hull vertices are in counter-clockwise order (usually)
    # We need to find the previous and next vertices for each hull vertex
    # ConvexHull.vertices is not guaranteed to be ordered in all versions, 
    # but typically it is. To be safe, we can sort them by angle or use the simplices.
    # Actually, hull.vertices returns indices. We can get the ordered boundary by 
    # traversing the simplices or just assuming the order if it's a 2D convex hull.
    # In scipy, hull.vertices is the set of vertices. 
    # Let's get the ordered vertices.
    # For 2D, we can sort by angle around centroid.
    centroid = pts[hull_verts].mean(axis=0)
    angles = np.arctan2(pts[hull_verts, 1] - centroid[1], pts[hull_verts, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    ordered_hull_verts = hull_verts[sorted_indices]
    
    n_hull = len(ordered_hull_verts)
    
    for idx in range(n_hull):
        v_idx = ordered_hull_verts[idx]
        prev_idx = ordered_hull_verts[(idx - 1) % n_hull]
        next_idx = ordered_hull_verts[(idx + 1) % n_hull]
        
        p_prev = pts[prev_idx]
        p_next = pts[next_idx]
        
        # Shoelace formula contribution for vertex v_idx:
        # A = 0.5 * sum (x_i y_{i+1} - x_{i+1} y_i)
        # Term involving v_idx:
        # From (prev, v): x_prev y_v - x_v y_prev
        # From (v, next): x_v y_next - x_next y_v
        
        # d/dx_v: y_next - y_prev
        # d/dy_v: -(x_next - x_prev) = x_prev - x_next
        
        grad_hull[v_idx, 0] = 0.5 * (p_next[1] - p_prev[1])
        grad_hull[v_idx, 1] = 0.5 * (p_prev[0] - p_next[0])
        
    return hull_area, grad_hull.flatten()

def _soft_min_objective(params, tri_idx, n, beta=50.0):
    """
    Objective for L-BFGS-B polish.
    """
    pts = params.reshape(n, 2)
    if not np.all(np.isfinite(pts)):
        return 1e6
    areas = _triangle_areas(pts, tri_idx)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-9:
        return 1e6
        
    # Soft min of areas
    shifted = -beta * areas
    max_shifted = shifted.max()
    if max_shifted < -1000:
        soft_min_val = 0.0
    else:
        soft_min_val = -(max_shifted + np.log(np.exp(shifted - max_shifted).sum())) / beta
        
    score = soft_min_val / hull_area
    return -score

def heilbronn_convex13() -> np.ndarray:
    n = 13
    tri_idx = _get_tri_indices(n)
    
    start_time = time.time()
    deadline = start_time + 580
    
    # Initial candidates
    candidates = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(reg_pts)
    
    # 2. Perturbed regular 13-gon
    np.random.seed(42)
    for _ in range(5):
        base = reg_pts.copy()
        noise = np.random.normal(0, 0.1, base.shape)
        candidates.append(base + noise)
        
    # 3. Triangular lattice subset
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    lattice_pts = []
    R = 3
    for i in range(-R, R+1):
        for j in range(-R, R+1):
            p = i * a1 + j * a2
            lattice_pts.append(p)
    lattice_pts = np.array(lattice_pts)
    lattice_pts = lattice_pts - lattice_pts.mean(axis=0)
    dists = np.linalg.norm(lattice_pts, axis=1)
    sorted_idx = np.argsort(dists)
    lat_pts = lattice_pts[sorted_idx[:n]]
    h_area = _hull_area(lat_pts)
    if h_area > 0:
        lat_pts = lat_pts / np.sqrt(h_area)
    candidates.append(lat_pts)
    
    # Evaluate initial candidates
    best_score = -1.0
    best_pts = None
    for c in candidates:
        s = _compute_score(c, tri_idx)
        if s > best_score:
            best_score = s
            best_pts = c.copy()
            
    # Subgradient Ascent Phase
    # We will run this from the best candidate and a few others
    subgrad_starts = [best_pts]
    for c in candidates[:3]:
        if not np.allclose(c, best_pts):
            subgrad_starts.append(c)
            
    for start_pts in subgrad_starts:
        if time.time() > deadline - 120:
            break
            
        pts = start_pts.copy()
        # Center
        pts -= pts.mean(axis=0)
        
        # Adaptive step size
        step_size = 0.01
        min_step = 0.0001
        patience = 0
        
        for it in range(500):
            if time.time() > deadline - 120:
                break
                
            min_area, grad_min = _grad_min_area(pts, tri_idx)
            hull_area, grad_hull = _grad_hull_area(pts)
            
            if hull_area < 1e-9:
                break
                
            # Gradient of score = (grad_min * hull_area - min_area * grad_hull) / hull_area^2
            grad_score = (grad_min * hull_area - min_area * grad_hull) / (hull_area ** 2)
            
            # Normalize gradient to prevent huge steps
            grad_norm = np.linalg.norm(grad_score)
            if grad_norm > 0:
                grad_score = grad_score / grad_norm
                
            # Update
            new_pts = pts + step_size * grad_score
            
            # Center new points
            new_pts -= new_pts.mean(axis=0)
            
            # Check score
            new_score = _compute_score(new_pts, tri_idx)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                pts = new_pts
                patience = 0
            else:
                pts = new_pts # Keep moving? Or revert? 
                # Reverting is safer for subgradient
                pts = pts - step_size * grad_score
                pts -= pts.mean(axis=0)
                patience += 1
                
            if patience > 50:
                step_size *= 0.8
                patience = 0
                if step_size < min_step:
                    break
                    
    # Local Refinement Phase (L-BFGS-B on Soft Min)
    # Use the best point found by subgradient as a start
    local_starts = [best_pts]
    # Add some perturbations
    for _ in range(5):
        perturbed = best_pts + np.random.normal(0, 0.05, best_pts.shape)
        local_starts.append(perturbed)
        
    for start_pts in local_starts:
        if time.time() > deadline - 30:
            break
            
        try:
            res = minimize(
                _soft_min_objective, 
                start_pts.flatten(), 
                args=(tri_idx, n, 50.0),
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 1e5:
                res_pts = res.x.reshape(-1, 2)
                res_score = _compute_score(res_pts, tri_idx)
                if res_score > best_score:
                    best_score = res_score
                    best_pts = res_pts.copy()
        except Exception:
            pass

    # Final Polish: Coordinate Descent / Nudge
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            areas = _triangle_areas(best_pts, tri_idx)
            min_area = areas.min()
            close_tris = np.where(areas < min_area * 1.05)[0]
            
            improved = False
            for tri_i in close_tris[:3]:
                i, j, k = tri_idx[tri_i]
                p_i = best_pts[i]
                p_j = best_pts[j]
                p_k = best_pts[k]
                centroid = (p_i + p_j + p_k) / 3.0
                
                step = 0.005
                trial = best_pts.copy()
                trial[i] += (p_i - centroid) * step
                trial[j] += (p_j - centroid) * step
                trial[k] += (p_k - centroid) * step
                
                new_score = _compute_score(trial, tri_idx)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = trial
                    improved = True
            if not improved:
                break

    # Normalize to unit hull area for consistency (optional, but good for stability)
    hull_area = _hull_area(best_pts)
    if hull_area > 1e-9:
        best_pts = best_pts / np.sqrt(hull_area)
        
    if not np.all(np.isfinite(best_pts)):
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_pts

# EVOLVE-BLOCK-END
