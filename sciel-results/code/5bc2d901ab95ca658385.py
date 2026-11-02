import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13
DEADLINE = 145.0  # Safety margin before 150s kill

# Pre-compute all triples for fast vectorized scoring
_triple_i = []
_triple_j = []
_triple_k = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _triple_i.append(_i)
            _triple_j.append(_j)
            _triple_k.append(_k)

_triple_i = np.array(_triple_i, dtype=np.int32)
_triple_j = np.array(_triple_j, dtype=np.int32)
_triple_k = np.array(_triple_k, dtype=np.int32)
_num_triples = len(_triple_i)

def _compute_areas(pts):
    """Compute all triangle areas for a point set."""
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-12

def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    
    areas = _compute_areas(pts)
    min_area = np.min(areas)
    
    if min_area < 1e-15:
        return 0.0
    
    return min_area / hull_area

def _get_active_indices(pts, areas, min_area, tol=1e-5):
    """
    Get indices of triangles that are close to the minimum.
    Also returns the set of point indices involved in these triangles.
    """
    active_mask = areas <= (min_area + tol)
    active_triple_indices = np.where(active_mask)[0]
    
    if len(active_triple_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    # Get unique point indices involved
    pts_i = _triple_i[active_triple_indices]
    pts_j = _triple_j[active_triple_indices]
    pts_k = _triple_k[active_triple_indices]
    
    all_pts = np.concatenate([pts_i, pts_j, pts_k])
    unique_pts = np.unique(all_pts)
    
    return active_triple_indices, unique_pts

def _try_improve_point(pts, idx, active_triples, areas, min_area, hull_area, step_size):
    """
    Try to move point idx to improve the minimum area.
    Returns (new_pts, improved_bool)
    """
    current_score = min_area / hull_area
    
    # Find all active triangles involving idx
    mask_i = (_triple_i[active_triples] == idx)
    mask_j = (_triple_j[active_triples] == idx)
    mask_k = (_triple_k[active_triples] == idx)
    
    # We want to move idx in a direction that increases the area of the smallest triangle involving it.
    # The gradient of area w.r.t. point i in triangle (i,j,k) is perpendicular to (j-k).
    # Area = 0.5 * |cross(Pj-Pi, Pk-Pi)|
    # dArea/dPi is roughly proportional to the normal of the triangle.
    
    # Let's collect gradients from the worst triangles involving this point.
    # To keep it simple and robust, we just try a few candidate directions:
    # 1. Normal to the smallest triangle involving idx.
    # 2. Normal to the second smallest.
    # 3. Random directions.
    
    # Identify triangles involving idx and sort by area
    involved_triples = []
    for t_idx in active_triples:
        if _triple_i[t_idx] == idx:
            involved_triples.append((t_idx, 'i'))
        elif _triple_j[t_idx] == idx:
            involved_triples.append((t_idx, 'j'))
        elif _triple_k[t_idx] == idx:
            involved_triples.append((t_idx, 'k'))
            
    if not involved_triples:
        return pts, False

    # Sort involved triples by area (ascending)
    involved_triples.sort(key=lambda x: areas[x[0]])
    
    best_new_pts = None
    best_new_score = current_score
    
    # Try moving based on the top 3 worst triangles involving this point
    for t_idx, role in involved_triples[:3]:
        i, j, k = _triple_i[t_idx], _triple_j[t_idx], _triple_k[t_idx]
        
        # Get coordinates
        Pi = pts[i]
        Pj = pts[j]
        Pk = pts[k]
        
        # Vector from Pi to Pj and Pi to Pk
        v1 = Pj - Pi
        v2 = Pk - Pi
        
        # Cross product (z-component) gives signed area * 2
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        
        # Gradient of Area w.r.t. Pi:
        # Area = 0.5 * |cross|
        # If cross > 0, increasing cross increases area.
        # d(cross)/dPi = -d(cross)/dPj = ... 
        # Actually, simpler: The direction to increase area is the outward normal of the triangle at vertex Pi.
        # The normal vector n is perpendicular to the edges.
        # For triangle (i,j,k), the normal pointing "out" (away from the third vertex relative to the edge opposite?)
        # Let's use the fact that moving Pi in the direction of the normal of the triangle increases the area.
        # The normal to the plane in 2D is just a 2D vector perpendicular to the edge.
        
        # Direction to increase area of triangle (i,j,k) by moving i:
        # It's the vector perpendicular to (j-k) ? No.
        # Area = 0.5 * | (Pj-Pi) x (Pk-Pi) |
        # Let f = (Pj-Pi) x (Pk-Pi).
        # df/dPi = - (Pj-Pi) x d(Pk-Pi)/dPi - (Pk-Pi) x d(Pj-Pi)/dPi ... this is getting messy.
        
        # Geometric intuition:
        # To increase the area of triangle (i,j,k) by moving i, move i in the direction of the altitude from i to side jk, extended.
        # i.e., away from the line segment jk.
        
        # Vector from i to jk line?
        # Let's just compute the gradient numerically or use the normal.
        # The normal to the triangle at vertex i, pointing such that area increases:
        # If we rotate (j-i) by 90 degrees, we get a normal.
        # Let's try both directions of the normal to the edge jk.
        
        # Vector j-k
        v_jk = Pj - Pk
        # Perpendicular vectors
        n1 = np.array([-v_jk[1], v_jk[0]])
        n1_norm = np.linalg.norm(n1)
        if n1_norm < 1e-9:
            continue
        n1 = n1 / n1_norm
        
        # Which direction increases area?
        # Move i slightly in n1 direction and check area.
        # Or simpler: The sign of the cross product tells us the orientation.
        # If cross > 0, the orientation is CCW. Moving i "out" (away from jk) increases area.
        # The vector from i to the line jk is along the normal.
        
        # Let's just test both n1 and -n1 with a small step.
        for sign in [1, -1]:
            dir_vec = sign * n1
            # Also try the other edges' normals? No, just the opposite edge is most relevant for area change w.r.t. vertex.
            
            new_Pi = Pi + step_size * dir_vec
            new_pts = pts.copy()
            new_pts[idx] = new_Pi
            
            # Quick check: does this point stay "reasonable"?
            # We don't strictly enforce convexity, but if it goes far, hull area might drop.
            
            new_areas = _compute_areas(new_pts)
            new_min_area = np.min(new_areas)
            
            # Only compute hull area if min area improved significantly?
            # Hull area is expensive. Let's estimate.
            # If new_min_area > min_area + tol, it's likely an improvement.
            if new_min_area > min_area + 1e-8:
                new_hull_area = _hull_area(new_pts)
                if new_hull_area > 1e-12:
                    new_score = new_min_area / new_hull_area
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_pts = new_pts
                        
    if best_new_pts is not None:
        return best_new_pts, True
    return pts, False

def _local_search(pts, max_iters=500, time_limit=None, start_time=None):
    """
    Perform local search by targeting the points in the minimum area triangles.
    """
    current_pts = pts.copy()
    current_areas = _compute_areas(current_pts)
    current_min_area = np.min(current_areas)
    current_hull_area = _hull_area(current_pts)
    current_score = current_min_area / current_hull_area
    
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 20
    
    for i in range(max_iters):
        if time_limit is not None:
            if (time.time() - start_time) > time_limit:
                break
                
        # Identify active triangles (close to min)
        active_triples, active_points = _get_active_indices(current_pts, current_areas, current_min_area, tol=1e-6)
        
        if len(active_points) == 0:
            break
            
        improved = False
        
        # Try to improve each active point
        # Shuffle to avoid bias
        rng = np.random.RandomState(i)
        rng.shuffle(active_points)
        
        for idx in active_points:
            new_pts, did_improve = _try_improve_point(
                current_pts, idx, active_triples, current_areas, 
                current_min_area, current_hull_area, step_size
            )
            
            if did_improve:
                current_pts = new_pts
                current_areas = _compute_areas(current_pts)
                current_min_area = np.min(current_areas)
                current_hull_area = _hull_area(current_pts)
                current_score = current_min_area / current_hull_area
                improved = True
                no_improve_count = 0
                # Reduce step size slightly on improvement? Or keep it.
                # Let's keep step size constant for a bit, then decay.
                break # Restart loop to update active sets
            
        if not improved:
            no_improve_count += 1
            # Decay step size
            step_size *= 0.9
            if step_size < 1e-5:
                break
            if no_improve_count > max_no_improve:
                break
                
    return current_pts, current_score

def _make_initial_guesses():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p1 = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(p1.flatten())
    
    # 2. 1 center + 12 outer
    p2 = np.zeros((N, 2))
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    p2[1:, 0] = np.cos(angles)
    p2[1:, 1] = np.sin(angles)
    inits.append(p2.flatten())
    
    # 3. 1 center + 6 mid + 6 outer
    p3 = np.zeros((N, 2))
    angles_m = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    p3[1:7, 0] = 0.5 * np.cos(angles_m)
    p3[1:7, 1] = 0.5 * np.sin(angles_m)
    angles_o = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    p3[7:, 0] = np.cos(angles_o)
    p3[7:, 1] = np.sin(angles_o)
    inits.append(p3.flatten())
    
    # 4. Random
    rng = np.random.RandomState(42)
    p4 = rng.uniform(-1, 1, size=(N, 2)).flatten()
    inits.append(p4)
    
    # 5. 1 center + 5 inner + 7 outer
    p5 = np.zeros((N, 2))
    angles_in = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    p5[1:6, 0] = 0.4 * np.cos(angles_in)
    p5[1:6, 1] = 0.4 * np.sin(angles_in)
    angles_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    p5[6:, 0] = np.cos(angles_out)
    p5[6:, 1] = np.sin(angles_out)
    inits.append(p5.flatten())
    
    # 6. 2 center-ish + 11 outer
    p6 = np.zeros((N, 2))
    p6[0] = [0.2, 0]
    p6[1] = [-0.2, 0]
    angles = np.linspace(0, 2 * np.pi, 11, endpoint=False)
    p6[2:, 0] = np.cos(angles)
    p6[2:, 1] = np.sin(angles)
    inits.append(p6.flatten())
    
    # 7. Another random
    rng2 = np.random.RandomState(123)
    p7 = rng2.uniform(-1, 1, size=(N, 2)).flatten()
    inits.append(p7)
    
    return inits

def construct():
    t_start = time.time()
    
    best_params = None
    best_score = -np.inf
    
    # 1. Evaluate initial guesses and run local search on them
    inits = _make_initial_guesses()
    
    for p in inits:
        if (time.time() - t_start) > (DEADLINE - 30.0):
            break
            
        pts = p.reshape(N, 2)
        s = _score_points(pts)
        
        # Run local search to improve this start
        refined_pts, refined_score = _local_search(pts, max_iters=100, time_limit=DEADLINE - 30.0, start_time=t_start)
        
        if refined_score > best_score:
            best_score = refined_score
            best_params = refined_pts.flatten()
            
    # 2. Simulated Annealing with Targeted Moves
    # We use the best result so far as the starting point.
    # We will mix random kicks with targeted moves.
    
    current_params = best_params.copy()
    current_pts = current_params.reshape(N, 2)
    current_score = best_score
    
    # SA Parameters
    T0 = 0.05
    T_end = 1e-5
    cooling_rate = 0.995
    max_iters = 5000
    
    rng = np.random.RandomState(123)
    step_size = 0.05
    
    for i in range(max_iters):
        if (time.time() - t_start) > (DEADLINE - 10.0):
            break
            
        T = T0 * (cooling_rate ** i)
        
        # Strategy: 50% chance random kick, 50% chance targeted local search step
        use_targeted = rng.random() < 0.5
        
        if use_targeted:
            # Perform a quick local search step (1-5 iterations)
            new_pts, new_score = _local_search(current_pts, max_iters=5, time_limit=0.1, start_time=time.time())
            neighbor_params = new_pts.flatten()
            neighbor_score = new_score
        else:
            # Random kick
            perturbation = rng.normal(0, step_size, size=(2 * N))
            neighbor_params = current_params + perturbation
            neighbor_pts = neighbor_params.reshape(N, 2)
            neighbor_score = _score_points(neighbor_pts)
            
        # Acceptance
        if neighbor_score > current_score:
            accept = True
        else:
            delta = current_score - neighbor_score
            if delta < 1e-12:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta / T)
                
        if accept:
            current_params = neighbor_params
            current_pts = neighbor_pts
            current_score = neighbor_score
            
            if current_score > best_score:
                best_score = current_score
                best_params = current_params.copy()
                
        # Adaptive step size
        step_size = 0.05 * (T / T0) + 0.01

    # 3. Final Polish
    # Use L-BFGS-B on a sharp soft-min for the final best point
    def _neg_soft_score_polish(params, beta=500.0):
        pts = params.reshape(N, 2)
        hull_area = _hull_area(pts)
        if hull_area <= 1e-12:
            return 1.0
        areas = _compute_areas(pts)
        min_area = np.min(areas)
        # Soft min
        # We want to maximize min_area / hull_area.
        # Minimize - (min_area / hull_area).
        # Approximation: - (exp(-beta * min_area) * ... ) is hard.
        # Just use hard score for polish? L-BFGS-B needs smooth.
        # Let's use a very high beta soft min.
        vals = -beta * areas
        v_max = np.max(vals)
        log_sum = v_max + np.log(np.sum(np.exp(vals - v_max)))
        soft_min_val = -log_sum / beta
        return -(soft_min_val / hull_area)

    if (time.time() - t_start) < (DEADLINE - 5.0):
        try:
            res = minimize(
                lambda x: _neg_soft_score_polish(x, beta=500.0),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-14}
            )
            if res.success:
                new_pts = res.x.reshape(N, 2)
                new_score = _score_points(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_params = res.x
        except Exception:
            pass

    # One last local search on the very best
    if (time.time() - t_start) < (DEADLINE - 2.0):
        final_pts, final_score = _local_search(best_params.reshape(N, 2), max_iters=50, time_limit=DEADLINE - 2.0, start_time=t_start)
        if final_score > best_score:
            best_score = final_score
            best_params = final_pts.flatten()

    return best_params.reshape(N, 2)
