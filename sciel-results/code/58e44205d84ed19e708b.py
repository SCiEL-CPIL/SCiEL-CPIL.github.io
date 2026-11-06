"""Program for covering the sphere with 130 equal caps."""
import time
import numpy as np
from scipy.optimize import minimize
import math

N = 130

# EVOLVE-BLOCK-START
def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n, seed_noise=0.0):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if seed_noise > 0:
        rng = np.random.default_rng(42)
        noise = rng.normal(0, seed_noise, (n, 3))
        pts = pts + noise
        pts = _normalize(pts)
    return pts

def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facets."""
    pts = _normalize(pts)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
    except Exception:
        return 180.0

    max_rad = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        if len(verts) >= 3:
            v0, v1, v2 = verts[0], verts[1], verts[2]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
        else:
            continue
        
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-15:
            continue
        normal = normal / norm_len
        
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        for v in verts:
            cos_ang = np.clip(np.dot(v, normal), -1.0, 1.0)
            ang = np.arccos(cos_ang)
            if ang > max_rad:
                max_rad = ang
    
    return math.degrees(max_rad)

def _facet_circumradius(pts, simplex):
    """Compute circumradius of a facet (angular distance from circumcenter to vertices)."""
    verts = pts[simplex]
    if len(verts) < 3:
        return 0.0, None, None
    
    # Compute circumcenter of the facet (in 3D, the point equidistant from all vertices on the plane)
    # For a triangle, use the standard formula
    if len(verts) == 3:
        a, b, c = verts[0], verts[1], verts[2]
        # Side lengths
        a_len = np.linalg.norm(b - c)
        b_len = np.linalg.norm(a - c)
        c_len = np.linalg.norm(a - b)
        
        # Circumcenter in barycentric coordinates
        # Using the formula: circumcenter = (a^2(b-c) + b^2(c-a) + c^2(a-b)) / (2 * cross product area)
        # Actually, let's use the standard approach:
        # Solve for circumcenter on the plane of the triangle
        
        # Normal to the plane
        e1 = b - a
        e2 = c - a
        normal = np.cross(e1, e2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-15:
            return 0.0, None, None
        normal = normal / norm_len
        
        # Circumcenter: point on plane equidistant from a, b, c
        # |circ - a|^2 = |circ - b|^2
        # |circ - a|^2 = |circ - c|^2
        # Let circ = a + s*e1 + t*e2
        # |s*e1 + t*e2|^2 = |(s-1)*e1 + t*e2|^2
        # s^2|e1|^2 + 2st e1.e2 + t^2|e2|^2 = (s-1)^2|e1|^2 + 2(s-1)t e1.e2 + t^2|e2|^2
        # s^2|e1|^2 + 2st e1.e2 = (s^2 - 2s + 1)|e1|^2 + (2st - 2t) e1.e2
        # 0 = -2s|e1|^2 + |e1|^2 - 2t e1.e2
        # 2s|e1|^2 + 2t e1.e2 = |e1|^2 ... (i)
        
        # |s*e1 + t*e2|^2 = |s*e1 + (t-1)*e2|^2
        # s^2|e1|^2 + 2st e1.e2 + t^2|e2|^2 = s^2|e1|^2 + 2s(t-1) e1.e2 + (t-1)^2|e2|^2
        # 2st e1.e2 + t^2|e2|^2 = (2st - 2s) e1.e2 + (t^2 - 2t + 1)|e2|^2
        # 0 = -2s e1.e2 - 2t|e2|^2 + |e2|^2
        # 2s e1.e2 + 2t|e2|^2 = |e2|^2 ... (ii)
        
        e1_sq = np.dot(e1, e1)
        e2_sq = np.dot(e2, e2)
        e1_e2 = np.dot(e1, e2)
        
        # Solve linear system:
        # 2*e1_sq * s + 2*e1_e2 * t = e1_sq
        # 2*e1_e2 * s + 2*e2_sq * t = e2_sq
        
        A = np.array([[2*e1_sq, 2*e1_e2], [2*e1_e2, 2*e2_sq]])
        b_vec = np.array([e1_sq, e2_sq])
        
        try:
            st = np.linalg.solve(A, b_vec)
        except np.linalg.LinAlgError:
            return 0.0, None, None
        
        s, t = st[0], st[1]
        circ = a + s * e1 + t * e2
        
        # Normalize circumcenter to unit sphere (it should be close)
        circ_norm = np.linalg.norm(circ)
        if circ_norm < 1e-15:
            return 0.0, None, None
        circ = circ / circ_norm
        
        # Circumradius is the angular distance from circ to any vertex
        cos_rad = np.clip(np.dot(circ, a), -1.0, 1.0)
        rad = np.arccos(cos_rad)
        
        return rad, circ, normal
    else:
        # For higher-order simplices, use the first 3 vertices as approximation
        return _facet_circumradius(pts, simplex[:3])

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin

    # 1. Initialization
    P = _fibonacci_sphere(N, seed_noise=0.005)
    
    # 2. Prepare samples for optimization
    S_opt = 15000
    samples_opt = _fibonacci_sphere(S_opt)
    
    # 3. Phase 1: Exponential max surrogate optimization
    # Objective: sum(exp(lambda * min_dist)) - sharply penalizes large gaps
    lambda_val = 15.0
    
    def objective_and_grad(P_flat):
        P = P_flat.reshape(-1, 3)
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        P = P / norms
        
        dots = samples_opt @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        min_dists = np.min(dists, axis=1)
        nearest_idx = np.argmin(dists, axis=1)
        
        loss = np.sum(np.exp(lambda_val * min_dists))
        
        # Gradient
        dots_nearest = dots[np.arange(S_opt), nearest_idx]
        denom = np.sqrt(np.maximum(1e-12, 1.0 - dots_nearest**2))
        
        weights = lambda_val * np.exp(lambda_val * min_dists)
        grad_factor = -weights / denom
        
        grad_P = np.zeros_like(P)
        np.add.at(grad_P, nearest_idx, grad_factor[:, None] * samples_opt)
        
        # Project to tangent space
        grad_dot_P = np.sum(grad_P * P, axis=1, keepdims=True)
        grad_tangent = grad_P - grad_dot_P * P
        
        norms_flat = np.linalg.norm(P_flat.reshape(-1, 3), axis=1, keepdims=True)
        norms_flat[norms_flat < 1e-15] = 1.0
        grad_P_raw = grad_tangent / norms_flat
        
        return loss, grad_P_raw.flatten()
    
    time_check = [False]
    def time_callback(xk):
        if time.time() > deadline - 30.0:
            time_check[0] = True
            raise ValueError("Time limit approaching")
    
    try:
        res = minimize(
            objective_and_grad,
            P.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 80, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False},
            callback=time_callback
        )
        if res.x is not None:
            P = res.x.reshape(-1, 3)
            P = _normalize(P)
    except Exception:
        pass
    
    if time.time() > deadline - 40.0:
        return P
    
    # 4. Phase 2: Convex hull facet-driven refinement
    # Directly optimize the true objective by shrinking the largest facet circumradius
    from scipy.spatial import ConvexHull
    
    max_iter_hull = 200
    for it in range(max_iter_hull):
        if time.time() > deadline - 15.0:
            break
        
        try:
            hull = ConvexHull(P)
        except Exception:
            break
        
        # Find the facet with the largest circumradius
        best_rad = 0.0
        best_simplex = None
        best_circ = None
        best_normal = None
        
        for simplex in hull.simplices:
            rad, circ, normal = _facet_circumradius(P, simplex)
            if rad > best_rad:
                best_rad = rad
                best_simplex = simplex
                best_circ = circ
                best_normal = normal
        
        if best_simplex is None or best_circ is None:
            break
        
        # Move vertices of the worst facet toward the circumcenter direction
        # The circumcenter is the "worst point" for this facet
        # We want to move the vertices so that the circumradius decreases
        # Move each vertex toward the circumcenter (on the sphere)
        
        for v_idx in best_simplex:
            v = P[v_idx]
            # Direction from v to best_circ (tangent direction)
            proj = np.dot(v, best_circ)
            grad_dir = best_circ - proj * v
            grad_norm = np.linalg.norm(grad_dir)
            
            if grad_norm < 1e-8:
                continue
            
            grad_dir = grad_dir / grad_norm
            
            # Try a few step sizes
            best_step = 0.0
            best_new_rad = best_rad
            
            for step_size in [0.05, 0.02, 0.01, 0.005]:
                new_v = v + step_size * grad_dir
                new_v = _normalize(new_v.reshape(1, 3))[0]
                
                old_v = P[v_idx].copy()
                P[v_idx] = new_v
                
                # Recompute circumradius of this facet
                new_rad, _, _ = _facet_circumradius(P, best_simplex)
                
                if new_rad < best_new_rad:
                    best_new_rad = new_rad
                    best_step = step_size
                    best_v = new_v.copy()
                else:
                    P[v_idx] = old_v
            
            if best_step > 0:
                P[v_idx] = best_v
            else:
                P[v_idx] = v  # Keep original
    
    # 5. Phase 3: Greedy polish with dense samples
    S_polish = 50000
    samples_polish = _fibonacci_sphere(S_polish)
    
    def get_min_dists(P):
        dots = samples_polish @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        nearest_idx = np.argmin(dists, axis=1)
        return min_dists, nearest_idx
    
    min_dists, _ = get_min_dists(P)
    current_max_dist = np.max(min_dists)
    
    rng = np.random.default_rng(42)
    
    for _ in range(300):
        if time.time() > deadline - 5.0:
            break
        
        min_dists, nearest_idx = get_min_dists(P)
        worst_idx = np.argmax(min_dists)
        worst_sample = samples_polish[worst_idx]
        
        p_idx = nearest_idx[worst_idx]
        p_old = P[p_idx].copy()
        
        # Move p_old towards worst_sample
        proj = np.dot(p_old, worst_sample)
        grad_dir = worst_sample - proj * p_old
        grad_norm = np.linalg.norm(grad_dir)
        
        if grad_norm < 1e-6:
            continue
        
        grad_dir = grad_dir / grad_norm
        
        # Add small random noise to escape local minima
        random_dir = rng.standard_normal(3)
        random_dir = random_dir - np.dot(random_dir, p_old) * p_old
        random_norm = np.linalg.norm(random_dir)
        if random_norm > 1e-6:
            random_dir = random_dir / random_norm
        else:
            random_dir = np.zeros(3)
            
        best_step = 0.0
        best_new_max = current_max_dist
        best_p = p_old.copy()
        
        for step_size in [0.1, 0.05, 0.02, 0.01, 0.005]:
            new_p = p_old + step_size * (0.8 * grad_dir + 0.2 * random_dir)
            new_p = _normalize(new_p.reshape(1, 3))[0]
            
            P[p_idx] = new_p
            min_dists_temp, _ = get_min_dists(P)
            new_max = np.max(min_dists_temp)
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_step = step_size
                best_p = new_p.copy()
            else:
                P[p_idx] = p_old
        
        if best_step > 0:
            P[p_idx] = best_p
            current_max_dist = best_new_max
        else:
            P[p_idx] = p_old
    
    return P
# EVOLVE-BLOCK-END
