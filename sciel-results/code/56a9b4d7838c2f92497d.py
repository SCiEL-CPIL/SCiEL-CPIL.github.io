"""Combined program for covering the sphere with 130 equal caps."""
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
        if len(verts) == 3:
            v0, v1, v2 = verts
            # Circumcenter of triangle on sphere
            # The center of the minimal enclosing circle of 3 points on a sphere
            # is along the normal. We can solve for the circumradius.
            a = np.linalg.norm(v1 - v2)
            b = np.linalg.norm(v0 - v2)
            c = np.linalg.norm(v0 - v1)
            s = (a + b + c) / 2.0
            area = math.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
            if area < 1e-10:
                continue
            R_tri = (a * b * c) / (4.0 * area)
            # Check if circumcenter is on the sphere side
            normal = np.cross(v1 - v0, v2 - v0)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-10:
                continue
            normal /= normal_norm
            center = normal * (1.0 / math.cos(math.acos(np.clip(np.dot(normal, v0), -1.0, 1.0))))
            # Actually, simpler: compute distance from center to v0
            # But R_tri is the radius of the circle in the plane.
            # The spherical radius is the angle.
            # cos(angle) = 1 / (1 + R_tri^2) ? No.
            # Let's just compute angle from center to v0.
            # Center is unit vector.
            # The circumcenter of a spherical triangle lies on the normal.
            # Distance from origin to circumcenter is 1/sin(delta) where delta is spherical radius?
            # Let's use the formula: R_spherical = arccos(1 - 2 * Area_spherical / (4*pi)) ... complicated.
            # Easier: compute angle between normal and v0.
            # Is the circumcenter simply the normalized normal?
            # Only for equilateral.
            # Let's stick to the sample-based or a simpler heuristic if exact is hard.
            # For now, approximate max_rad by sampling if we can't easily compute exact circumradius.
            pass
        # Fallback to sample-based if hull logic is too complex or slow
        # For speed, we'll use a hybrid: hull for validation, samples for optimization.
    
    # Let's revert to sample-based for the radius calculation in the loss function
    # because exact circumradius computation for all facets is tricky and potentially slower
    # or prone to numerical errors for near-degenerate facets.
    # However, for the FINAL evaluation, we should use the exact definition if possible.
    # The task says: "largest circumradius of the facets of the convex hull".
    # This implies we MUST compute circumradius of hull facets.
    
    # Let's implement a robust circumradius calculation.
    max_rad = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        n_v = len(verts)
        if n_v == 3:
            v0, v1, v2 = verts
            # Solve for center c such that |c - v0| = |c - v1| = |c - v2| and c is on sphere?
            # No, c is inside the sphere? No, the facet is on the convex hull.
            # The covering radius is the max angular distance from any point on sphere to nearest point.
            # For a convex hull facet, the furthest point on the sphere from the points in the facet
            # is the circumcenter of the spherical triangle.
            
            # Compute normal
            e1 = v1 - v0
            e2 = v2 - v0
            n = np.cross(e1, e2)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            
            # The circumcenter of the spherical triangle is along n.
            # Let c = n (unit vector).
            # Check distance from c to v0, v1, v2.
            # If c is the center, then dist(c, v0) should be minimal for the cap?
            # Actually, the furthest point in the cap defined by the triangle is the circumcenter
            # IF the triangle is acute. If obtuse, the circumcenter might be outside the triangle,
            # and the furthest point is one of the vertices?
            # No, the covering radius is the radius of the largest empty circle.
            # For a spherical triangle, the largest empty circle center is either the circumcenter
            # (if acute) or the midpoint of the longest edge (if obtuse).
            
            # Let's compute the circumcenter c = n (or -n, pick the one closer to centroid).
            centroid = (v0 + v1 + v2) / 3.0
            if np.dot(n, centroid) < 0:
                n = -n
            
            # Compute angular distances from n to vertices
            d0 = np.arccos(np.clip(np.dot(n, v0), -1.0, 1.0))
            d1 = np.arccos(np.clip(np.dot(n, v1), -1.0, 1.0))
            d2 = np.arccos(np.clip(np.dot(n, v2), -1.0, 1.0))
            max_d = max(d0, d1, d2)
            
            # Check if this is the circumcenter.
            # The circumradius R satisfies cos(R) = (1 + cos(a)cos(b) + cos(c)) / ...
            # Simpler: check if n is inside the triangle.
            # If n is inside, the radius is max_d.
            # If n is outside, the largest empty circle might be centered elsewhere.
            # A robust way:
            # 1. Try circumcenter n. Radius R_c = max_d.
            # 2. Try midpoints of edges.
            # For a spherical triangle, the largest empty circle is determined by:
            # - If acute: circumcenter.
            # - If obtuse: midpoint of longest side.
            
            # Check angles of the spherical triangle.
            # cos(a) = dot(v1, v2) ... no, angles at vertices.
            # Angle at v0: between v1 and v2, projected to tangent plane at v0.
            t1 = v1 - np.dot(v1, v0) * v0
            t2 = v2 - np.dot(v2, v0) * v0
            t1 /= np.linalg.norm(t1)
            t2 /= np.linalg.norm(t2)
            angle_0 = np.arccos(np.clip(np.dot(t1, t2), -1.0, 1.0))
            
            t1 = v0 - np.dot(v0, v1) * v1
            t2 = v2 - np.dot(v2, v1) * v1
            t1 /= np.linalg.norm(t1)
            t2 /= np.linalg.norm(t2)
            angle_1 = np.arccos(np.clip(np.dot(t1, t2), -1.0, 1.0))
            
            t1 = v0 - np.dot(v0, v2) * v2
            t2 = v1 - np.dot(v1, v2) * v2
            t1 /= np.linalg.norm(t1)
            t2 /= np.linalg.norm(t2)
            angle_2 = np.arccos(np.clip(np.dot(t1, t2), -1.0, 1.0))
            
            if angle_0 < np.pi/2 and angle_1 < np.pi/2 and angle_2 < np.pi/2:
                # Acute: use circumcenter
                radius = max_d
            else:
                # Obtuse: use midpoint of longest edge
                # Longest edge is opposite the largest angle.
                max_angle = max(angle_0, angle_1, angle_2)
                if max_angle == angle_0:
                    # Edge v1-v2
                    mid = v1 + v2
                    mid = _normalize(mid.reshape(1, 3))[0]
                    radius = np.arccos(np.clip(np.dot(mid, v1), -1.0, 1.0))
                elif max_angle == angle_1:
                    # Edge v0-v2
                    mid = v0 + v2
                    mid = _normalize(mid.reshape(1, 3))[0]
                    radius = np.arccos(np.clip(np.dot(mid, v0), -1.0, 1.0))
                else:
                    # Edge v0-v1
                    mid = v0 + v1
                    mid = _normalize(mid.reshape(1, 3))[0]
                    radius = np.arccos(np.clip(np.dot(mid, v0), -1.0, 1.0))
            
            if radius > max_rad:
                max_rad = radius
        elif n_v == 4:
            # Tetrahedron facet (should not happen on sphere convex hull unless points are coplanar)
            # Treat as multiple triangles or ignore for now
            pass

    return math.degrees(max_rad)

def _fibonacci_sphere(n, seed_noise=0.0, seed=42):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if seed_noise > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, seed_noise, (n, 3))
        pts = pts + noise
        pts = _normalize(pts)
    return pts

def _soft_max_loss(P_flat, samples, lam=10.0):
    """
    Soft-max covering loss using log-sum-exp for stability.
    """
    P = P_flat.reshape(-1, 3)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    Pn = P / norms
    n = len(Pn)

    # 1. Soft-max covering loss
    dots = samples @ Pn.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    nearest = np.argmin(dists, axis=1)
    
    # Log-sum-exp trick
    max_min_dists = np.max(min_dists)
    exp_terms = np.exp(lam * (min_dists - max_min_dists))
    log_sum_exp = np.log(np.sum(exp_terms))
    cover_loss = (1.0 / lam) * (lam * max_min_dists + log_sum_exp)
    
    # Gradient
    # d/dP_i sum_j exp(lam * min_dist_j)
    # Only the nearest point to each sample contributes.
    weights = lam * exp_terms / np.sum(exp_terms)
    # Wait, d(LSE)/dx_i = sum(exp(x_i - LSE)) = weights
    # So gradient of cover_loss wrt dist_j is weights_j
    # d(dist_j)/dP_k = - (P_k - (P_k.P_j)P_j) / sin(dist_j) for k = nearest[j]
    # Wait, dist_j = arccos(P_n[j] . y_j)
    # d(dist_j)/dP_n[j] = - (y_j - (P_n[j].y_j)P_n[j]) / sqrt(1 - (P_n[j].y_j)^2)
    
    dots_nearest = dots[np.arange(len(samples)), nearest]
    denom = np.sqrt(np.maximum(1e-12, 1.0 - dots_nearest**2))
    grad_factor = -weights / denom
    
    grad_P = np.zeros_like(Pn)
    # For each sample j, add contribution to P_nearest[j]
    tangent = samples - dots_nearest[:, None] * Pn[np.arange(len(samples)), nearest]
    grad_contrib = (grad_factor[:, None] * tangent)
    np.add.at(grad_P, nearest, grad_contrib)

    # Project gradient to tangent space
    grad_dot = np.sum(grad_P * Pn, axis=1, keepdims=True)
    grad_tan = grad_P - grad_dot * Pn
    
    grad_flat = (grad_tan / norms).flatten()
    
    return cover_loss, grad_flat

def _fast_local_search(P, deadline, max_iters=200):
    """
    Fast local search using exact covering radius.
    Moves the point responsible for the worst facet.
    """
    n = len(P)
    rng = np.random.default_rng(42)
    
    current_max = _covering_radius_deg(P)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Find the worst facet
        # We need to know which facet is the worst to identify which point to move.
        # _covering_radius_deg doesn't return the facet.
        # Let's modify _covering_radius_deg or write a helper to return the worst point index.
        
        # For now, we'll use a simpler heuristic:
        # Compute the exact radius, and if it improves, keep it.
        # To find the point to move, we can approximate the "worst sample" from a dense sample set,
        # or we can try moving each point slightly and see if the exact radius improves.
        # Trying each point is O(N * Cost_Hull). N=130, Cost_Hull is small.
        # 130 hull computations per iteration. If hull takes 1ms, that's 130ms per iter.
        # 100 iters = 13s. This is feasible within 150s budget if we have 50-60s left for local search.
        
        improved = False
        for i in range(n):
            p_old = P[i].copy()
            
            # Try a few random directions
            for _ in range(5):
                rand = rng.standard_normal(3)
                rand = rand - np.dot(rand, p_old) * p_old
                rand_norm = np.linalg.norm(rand)
                if rand_norm < 1e-8:
                    continue
                rand = rand / rand_norm
                
                best_step = 0.0
                best_max = current_max
                best_p = None
                
                steps = [0.05, 0.02, 0.01, 0.005]
                for step in steps:
                    new_p = p_old + step * rand
                    new_p = _normalize(new_p.reshape(1, 3))[0]
                    P[i] = new_p
                    new_max = _covering_radius_deg(P)
                    if new_max < best_max:
                        best_max = new_max
                        best_step = step
                        best_p = new_p.copy()
                    else:
                        P[i] = p_old
                
                if best_p is not None:
                    P[i] = best_p
                    current_max = best_max
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # No improvement found with small random steps.
            # Try a larger random step to escape local minimum?
            # Or break if stuck for a while.
            # Let's just break to save time.
            break
            
    return P, current_max

def construct():
    start_time = time.time()
    # Safety margin: 10s before the 150s limit
    deadline = start_time + 140.0
    
    best_P = None
    best_score = float('inf')
    
    # Restart parameters
    restarts = [
        (0.03, 42),
        (0.05, 123),
        (0.08, 999),
        (0.0, 777)
    ]
    
    total_time_budget = deadline - start_time
    time_per_restart = total_time_budget / len(restarts)
    
    for r_idx, (noise, seed) in enumerate(restarts):
        current_deadline = start_time + (r_idx + 1) * time_per_restart
        current_deadline = min(current_deadline, deadline - 5.0)
        
        if time.time() > current_deadline - 2.0:
            break
            
        # 1. Initialize
        P = _fibonacci_sphere(N, seed_noise=noise, seed=seed)
        
        # 2. Optimization Phase
        # Use a moderate number of samples for optimization (20k)
        S_opt = 20000
        samples_opt = _fibonacci_sphere(S_opt, seed=seed+100)
        
        # L-BFGS optimization
        opt_deadline = current_deadline - 15.0 # Reserve 15s for local search
        
        def time_callback(xk):
            if time.time() > opt_deadline:
                raise ValueError("Time limit approaching")
        
        try:
            res = minimize(
                lambda x: _soft_max_loss(x, samples_opt, lam=10.0),
                P.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False},
                callback=time_callback
            )
            if res.x is not None:
                P = _normalize(res.x.reshape(-1, 3))
        except Exception:
            pass
            
        # 3. Local Search Phase
        # Use exact covering radius
        local_deadline = current_deadline - 2.0
        
        if time.time() < local_deadline:
            P, current_max = _fast_local_search(P, local_deadline, max_iters=50)
            
            # Evaluate with exact metric
            current_score = _covering_radius_deg(P)
            
            if current_score < best_score:
                best_score = current_score
                best_P = P.copy()
                
    # 4. Final Polish
    if best_P is not None and time.time() < deadline - 5.0:
        P_best, _ = _fast_local_search(best_P, deadline - 1.0, max_iters=50)
        final_score = _covering_radius_deg(P_best)
        
        if final_score < best_score:
            best_P = P_best
            best_score = final_score

    return _normalize(best_P)

# EVOLVE-BLOCK-END
