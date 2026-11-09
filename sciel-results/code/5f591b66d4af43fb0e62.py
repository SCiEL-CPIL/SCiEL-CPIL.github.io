import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize, dual_annealing

N = 11
H = np.sqrt(3) / 2
# Vertices of the unit equilateral triangle
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

def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle."""
    points = points.copy()
    for i in range(N):
        x, y = points[i]
        # y is bounded by [0, H]
        y = max(0.0, min(H, y))
        # x is bounded by the slanted sides
        # Left side: y = sqrt(3) * x  => x = y / sqrt(3)
        # Right side: y = sqrt(3) * (1 - x) => x = 1 - y / sqrt(3)
        x_min = y / np.sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        if x_min > x_max:
            x = 0.5
            y = 0.0
        else:
            x = max(x_min, min(x_max, x))
        points[i] = [x, y]
    return points

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

def _inside_penalty(points, margin=0.001):
    """Smooth penalty for points outside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    # Line 1: y >= 0
    viol1 = np.maximum(0.0, -y + margin)
    # Line 2: y <= sqrt(3) * x  =>  y - sqrt(3)*x <= 0
    viol2 = np.maximum(0.0, y - np.sqrt(3) * x + margin)
    # Line 3: y <= sqrt(3) * (1 - x) => y - sqrt(3)*(1-x) <= 0
    viol3 = np.maximum(0.0, y - np.sqrt(3) * (1.0 - x) + margin)
    return np.sum(viol1**2) + np.sum(viol2**2) + np.sum(viol3**2)

def _is_inside(points, tol=1e-8):
    """Check if all points are inside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    return (np.all(y >= -tol) and
            np.all(y <= np.sqrt(3) * x + tol) and
            np.all(y <= np.sqrt(3) * (1.0 - x) + tol))

def _soft_min_area(params, temperature=0.01):
    """
    Differentiable soft-minimum of triangle areas using log-sum-exp.
    This creates a smooth landscape for gradient-based optimization.
    """
    points = params.reshape(N, 2)
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    cross = np.abs(dx1 * dy2 - dy1 * dx2)
    areas = 0.5 * cross
    
    # Soft minimum: -T * log(sum(exp(-areas/T)))
    # As T -> 0, this approaches the true minimum
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -temperature * log_sum
    
    return soft_min

def _objective_soft(params, temperature=0.01):
    """
    Objective: negative soft-min area + boundary penalty.
    This is differentiable and suitable for L-BFGS-B.
    """
    points = params.reshape(N, 2)
    pen = _inside_penalty(points)
    soft_min = _soft_min_area(params, temperature)
    return -soft_min + 100.0 * pen

def _objective(params):
    """Objective: negative min area + boundary penalty (for non-gradient methods)."""
    points = params.reshape(N, 2)
    pen = _inside_penalty(points)
    if pen > 1e-4:
        return -_min_area_fast(points) + 1000.0 * pen
    return -_min_area_fast(points)

def _l_bfgsb_refine(points, time_limit, temperature=0.01):
    """
    Use L-BFGS-B with the soft-min objective to refine the configuration.
    Start with a larger temperature and anneal it down.
    """
    start = time.time()
    best_points = points.copy()
    best_score = _min_area_fast(points)
    
    # Anneal temperature from large to small
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    
    for temp in temperatures:
        if time.time() > time_limit:
            break
        
        def obj(params):
            return _objective_soft(params, temp)
        
        def grad(params):
            # Numerical gradient (L-BFGS-B can also estimate, but we provide it)
            points = params.reshape(N, 2)
            p1 = points[_idx_i]
            p2 = points[_idx_j]
            p3 = points[_idx_k]
            dx1 = p2[:, 0] - p1[:, 0]
            dy1 = p2[:, 1] - p1[:, 1]
            dx2 = p3[:, 0] - p1[:, 0]
            dy2 = p3[:, 1] - p1[:, 1]
            cross = np.abs(dx1 * dy2 - dy1 * dx2)
            areas = 0.5 * cross
            
            # Gradient of soft-min
            scaled = -areas / temp
            max_scaled = np.max(scaled)
            exp_scaled = np.exp(scaled - max_scaled)
            weights = exp_scaled / np.sum(exp_scaled)
            
            # d(soft_min)/d(area_k) = -weights[k]
            # d(area_k)/d(point) via chain rule
            grad_points = np.zeros((N, 2))
            
            for t in range(_n_tris):
                i, j, k = _idx_i[t], _idx_j[t], _idx_k[t]
                w = weights[t]
                # area = 0.5 * |cross|
                # d(area)/d(point_i) = 0.5 * sign(cross) * (dy2, -dx2) etc.
                c = cross[t]
                if abs(c) < 1e-15:
                    continue
                s = np.sign(c)
                # d(cross)/d(p1) = -(dy2, -dx2) = (-dy2, dx2) ... let me recompute
                # cross = dx1*dy2 - dy1*dx2
                # d(cross)/d(p1.x) = -dy2, d(cross)/d(p1.y) = dx2
                # d(cross)/d(p2.x) = dy2, d(cross)/d(p2.y) = -dx1
                # d(cross)/d(p3.x) = -dy1, d(cross)/d(p3.y) = dx1
                grad_points[i, 0] += w * s * (-dy2[t])
                grad_points[i, 1] += w * s * (dx2[t])
                grad_points[j, 0] += w * s * (dy2[t])
                grad_points[j, 1] += w * s * (-dx1[t])
                grad_points[k, 0] += w * s * (-dy1[t])
                grad_points[k, 1] += w * s * (dx1[t])
            
            # Multiply by 0.5 for area
            grad_points *= 0.5
            
            # Add boundary penalty gradient
            x = points[:, 0]
            y = points[:, 1]
            viol1 = np.maximum(0.0, -y + 0.001)
            viol2 = np.maximum(0.0, y - np.sqrt(3) * x + 0.001)
            viol3 = np.maximum(0.0, y - np.sqrt(3) * (1.0 - x) + 0.001)
            
            # d(pen)/d(y) from viol1
            g_y = -2.0 * viol1
            # d(pen)/d(x) from viol2: -sqrt(3)*2*viol2, d(pen)/d(y): 2*viol2
            g_x = -np.sqrt(3) * 2.0 * viol2
            g_y += 2.0 * viol2
            # d(pen)/d(x) from viol3: sqrt(3)*2*viol3, d(pen)/d(y): 2*viol3
            g_x += np.sqrt(3) * 2.0 * viol3
            g_y += 2.0 * viol3
            
            grad_points[:, 0] += g_x
            grad_points[:, 1] += g_y
            
            # Objective is -soft_min + 100*pen, so negate the soft_min gradient
            # and scale penalty gradient by 100
            return (-grad_points + 100.0 * np.column_stack([g_x, g_y])).flatten()
        
        # Use L-BFGS-B
        try:
            result = minimize(
                obj,
                best_points.flatten(),
                jac=grad,
                method='L-BFGS-B',
                bounds=[(0, 1), (0, H)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            new_points = _clamp_to_triangle(result.x.reshape(N, 2))
            new_score = _min_area_fast(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        except Exception:
            pass
    
    return best_points, best_score

def _local_refine_worst_triangle(points, time_limit):
    """
    Iteratively refine by optimizing the worst triangle's points.
    This is a block coordinate descent method.
    """
    start = time.time()
    best_points = points.copy()
    best_score = _min_area_fast(points)
    
    for iteration in range(1000):
        if time.time() > time_limit:
            break
        
        min_area, worst_idx = _min_area_and_idx(best_points)
        
        if min_area > best_score + 1e-12:
            best_score = min_area
            best_points = points.copy()
        
        i, j, k = _idx_i[worst_idx], _idx_j[worst_idx], _idx_k[worst_idx]
        tri_points = best_points[[i, j, k]].copy()
        
        combined_mask = _point_masks[i] | _point_masks[j] | _point_masks[k]
        
        def _tri_obj(tri_params):
            tp = tri_params.reshape(3, 2)
            x, y = tp[:, 0], tp[:, 1]
            if np.any(y < -0.1) or np.any(y > H + 0.1):
                return 1000.0
            if np.any(y > np.sqrt(3)*x + 0.1) or np.any(y > np.sqrt(3)*(1-x) + 0.1):
                return 1000.0
            
            full = best_points.copy()
            full[i], full[j], full[k] = tp[0], tp[1], tp[2]
            
            p1 = full[_idx_i]
            p2 = full[_idx_j]
            p3 = full[_idx_k]
            dx1 = p2[:, 0] - p1[:, 0]
            dy1 = p2[:, 1] - p1[:, 1]
            dx2 = p3[:, 0] - p1[:, 0]
            dy2 = p3[:, 1] - p1[:, 1]
            cross = np.abs(dx1 * dy2 - dy1 * dx2)
            areas = 0.5 * cross
            
            return -np.min(areas[combined_mask])
        
        try:
            res = minimize(_tri_obj, tri_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            
            new_tri = res.x.reshape(3, 2)
            new_tri = _clamp_to_triangle(new_tri)
            
            test_points = best_points.copy()
            test_points[i], test_points[j], test_points[k] = new_tri[0], new_tri[1], new_tri[2]
            
            if _is_inside(test_points, tol=1e-8):
                new_score = _min_area_fast(test_points)
                if new_score > best_score:
                    best_points = test_points
                    best_score = new_score
        except Exception:
            pass

    return best_points, best_score

def _init_symmetric():
    """Generate a good symmetric starting configuration."""
    points = np.zeros((N, 2))
    candidates = []
    for i in range(5):
        for j in range(5):
            u = i / 4.0
            v = j / 4.0
            w = 1.0 - u - v
            if w >= -1e-9:
                w = max(0.0, w)
                x = v * 1.0 + w * 0.5
                y = w * H
                if x >= -1e-9 and x <= 1 + 1e-9 and y >= -1e-9 and y <= H + 1e-9:
                    candidates.append([x, y])
    
    candidates = np.array(candidates)
    if len(candidates) < N:
        while len(candidates) < N:
            u = np.random.random()
            v = np.random.random()
            if u + v <= 1:
                x = v + 0.5 * (1 - u - v)
                y = H * (1 - u - v)
                candidates = np.vstack([candidates, [[x, y]]])
    
    selected = [candidates[0]]
    for _ in range(N - 1):
        dists = np.min([np.linalg.norm(candidates - s, axis=1) for s in selected], axis=0)
        next_idx = np.argmax(dists)
        selected.append(candidates[next_idx])
        
    pts = np.array(selected)
    pts += np.random.normal(0, 0.01, size=pts.shape)
    return _clamp_to_triangle(pts)

def _init_random(rng):
    """Generate random initialization inside the triangle."""
    pts = np.zeros((N, 2))
    for i in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        x = v + 0.5 * (1 - u - v)
        y = H * (1 - u - v)
        pts[i] = [x, y]
    return pts

def _init_lattice(rng):
    """Generate a lattice-based initialization with perturbation."""
    pts = []
    # Try different grid densities
    for density in [4, 5, 6]:
        candidates = []
        for i in range(density + 1):
            for j in range(density + 1):
                u = i / density
                v = j / density
                w = 1.0 - u - v
                if w >= -1e-9:
                    w = max(0.0, w)
                    x = v + 0.5 * w
                    y = w * H
                    if x >= -1e-9 and x <= 1 + 1e-9 and y >= -1e-9 and y <= H + 1e-9:
                        candidates.append([x, y])
        
        candidates = np.array(candidates)
        if len(candidates) >= N:
            # Greedy selection for max min-distance
            selected = [candidates[0]]
            for _ in range(N - 1):
                dists = np.min([np.linalg.norm(candidates - s, axis=1) for s in selected], axis=0)
                next_idx = np.argmax(dists)
                selected.append(candidates[next_idx])
            
            pts = np.array(selected)
            pts += rng.normal(0, 0.02, size=pts.shape)
            pts = _clamp_to_triangle(pts)
            return pts
    
    # Fallback to random
    return _init_random(rng)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin: 150s limit
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Generate diverse initializations
    inits = []
    inits.append(_init_symmetric())
    inits.append(_init_lattice(rng))
    for _ in range(4):
        inits.append(_init_random(rng))
    
    # Evaluate all inits
    for init in inits:
        score = _min_area_fast(init)
        if score > best_score:
            best_score = score
            best_points = init.copy()
    
    # Phase 1: L-BFGS-B refinement from each good init
    for init in inits[:5]:
        if time.time() > deadline - 30:
            break
        
        t_limit = time.time() + 10
        refined_pts, refined_score = _l_bfgsb_refine(init, t_limit)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # Phase 2: Local refinement on best
    if time.time() < deadline - 20:
        t_limit = time.time() + 15
        refined_pts, refined_score = _local_refine_worst_triangle(best_points, t_limit)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # Phase 3: More L-BFGS-B refinement with different temperatures
    if time.time() < deadline - 15:
        t_limit = time.time() + 12
        refined_pts, refined_score = _l_bfgsb_refine(best_points, t_limit)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # Phase 4: Final local refinement
    if time.time() < deadline - 5:
        t_limit = time.time() + 4
        refined_pts, refined_score = _local_refine_worst_triangle(best_points, t_limit)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # Phase 5: One more L-BFGS-B pass if time permits
    if time.time() < deadline - 3:
        t_limit = time.time() + 2
        refined_pts, refined_score = _l_bfgsb_refine(best_points, t_limit)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    if best_points is None:
        best_points = _init_symmetric()
        
    return _clamp_to_triangle(best_points)
