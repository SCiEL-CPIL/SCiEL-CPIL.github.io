import numpy as np
import time
from scipy.optimize import minimize

def _vertex_score(distances_sorted):
    d = distances_sorted
    n = len(d)
    if n != 19:
        return 0.0
    if np.any(d < 0.05):
        return 0.0
    diffs = d[1:] - d[:-1]
    min_diff = np.min(diffs)
    k = int(np.argmax(diffs == min_diff))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        four_d = d[0:4]
    elif k == 17:
        four_d = d[15:19]
    else:
        four_d = d[k-1:k+3]
    if np.any(four_d <= 0):
        return 0.0
    ratios = np.maximum(four_d / D, D / four_d)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
    return float(score)

def _compute_min_vertex_score(polygon):
    n = len(polygon)
    if n != 20:
        return 0.0
    if not np.all(np.isfinite(polygon)):
        return 0.0
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist
    
    # Check convexity
    cross_products = np.zeros(n)
    for i in range(n):
        e1 = p[(i+1) % n] - p[i]
        e2 = p[(i+2) % n] - p[(i+1) % n]
        cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
    if np.any(np.abs(cross_products) < 1e-10):
        return 0.0
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return 0.0
    # Check total turn
    total_turn = 0.0
    for i in range(n):
        v1 = p[(i+1) % n] - p[i]
        v2 = p[(i+2) % n] - p[(i+1) % n]
        ang1 = np.arctan2(v1[1], v1[0])
        ang2 = np.arctan2(v2[1], v2[0])
        turn = ang2 - ang1
        while turn > np.pi:
            turn -= 2 * np.pi
        while turn < -np.pi:
            turn += 2 * np.pi
        total_turn += turn
    if not np.isclose(abs(total_turn), 2 * np.pi, atol=1e-6):
        return 0.0

    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    min_score = 1.0
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        score = _vertex_score(d_sorted)
        min_score = min(min_score, score)
    return min_score

def _soft_min_vertex_score(polygon, temperature=0.05):
    n = len(polygon)
    if n != 20:
        return 0.0
    if not np.all(np.isfinite(polygon)):
        return 0.0
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist
    
    # Convexity check (soft)
    cross_products = np.zeros(n)
    for i in range(n):
        e1 = p[(i+1) % n] - p[i]
        e2 = p[(i+2) % n] - p[(i+1) % n]
        cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
    if np.any(np.abs(cross_products) < 1e-8):
        return 0.0
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return 0.0
        
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    scores = np.zeros(n)
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        scores[i] = _vertex_score(d_sorted)
    
    # Soft min
    neg_scores = -scores
    max_neg = np.max(neg_scores)
    if max_neg >= 0:
        log_sum_exp_neg = max_neg + temperature * np.log(np.sum(np.exp((neg_scores - max_neg) / temperature)))
        smooth_min_score = -log_sum_exp_neg
    else:
        smooth_min_score = -max_neg
        
    return max(0.0, smooth_min_score)

def _polygon_from_edges(lengths, angle_deltas):
    n = 20
    base_angle = 2.0 * np.pi / n
    # Compute total delta sum to ensure closure of angles
    # We pass 19 deltas, the 20th is determined to sum to 0
    if len(angle_deltas) == n - 1:
        deltas = np.zeros(n)
        deltas[:n-1] = angle_deltas
        deltas[n-1] = -np.sum(angle_deltas)
    else:
        deltas = angle_deltas
        
    angles = base_angle + deltas
    
    # Check if angles are valid for strict convexity
    # They must be > 0 and < pi
    if np.any(angles < 1e-4) or np.any(angles > np.pi - 1e-4):
        return None
        
    # Construct edge vectors
    # Cumulative angles
    cum_angles = np.cumsum(angles)
    # The direction of edge k is determined by the sum of previous turns?
    # Let's define the turn at vertex k as angles[k].
    # If we start with edge 0 along x-axis, its angle is 0.
    # Vertex 0 is at (0,0).
    # Edge 0 goes to Vertex 1.
    # Turn at Vertex 1 is angles[1]?
    # Standard definition: The exterior angle at vertex k is the turn from edge k-1 to edge k.
    # Let exterior angles be $\theta_k$.
    # Direction of edge k is $\phi_k = \sum_{j=0}^{k-1} \theta_j$?
    # If $\theta_j$ are the turns at vertices 0, 1, ..., n-1.
    # Sum of $\theta_j$ must be $2\pi$.
    # My `angles` array represents the turns.
    
    # Let's set $\phi_0 = 0$.
    # $\phi_{k} = \sum_{j=0}^{k-1} \theta_j$.
    # Edge k vector: $L_k e^{i \phi_k}$.
    
    phi = np.zeros(n)
    for k in range(1, n):
        phi[k] = phi[k-1] + angles[k-1]
        
    edges = np.stack([lengths * np.cos(phi), lengths * np.sin(phi)], axis=1)
    
    # Vertices
    vertices = np.zeros((n, 2))
    for k in range(1, n):
        vertices[k] = vertices[k-1] + edges[k-1]
        
    # Check closure (should be close to 0 if sum of turns is 2pi and lengths are consistent? 
    # No, lengths don't affect closure of angles, but the polygon must close.
    # The SC mapping guarantees closure.
    # Here, we just construct an open chain. We need to check if it closes.
    # Actually, for a convex polygon defined by turns and lengths, it closes if the vector sum is 0.
    # We are optimizing lengths too, so the optimizer can adjust lengths to close the polygon?
    # No, the sum of turns is fixed to 2pi by construction.
    # The polygon closes if $\sum L_k e^{i \phi_k} = 0$.
    # This is a constraint on L and deltas.
    # We can add a penalty for non-closure.
    
    closure_error = np.linalg.norm(vertices[-1]) # distance from end to start (which is 0)
    if closure_error > 1e-6:
        # Scale down or return penalty
        # For soft score, we return 0 if not closed
        return None
        
    return vertices

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 615.0
    
    n = 20
    best_poly = None
    best_score = 0.0
    
    # Initial guess: Regular polygon
    # Lengths all equal, deltas all 0
    L0 = np.ones(n)
    D0 = np.zeros(n-1)
    
    # Check regular polygon
    # For regular polygon, closure is exact.
    # Let's verify
    v_reg = _polygon_from_edges(L0, D0)
    if v_reg is not None:
        s = _compute_min_vertex_score(v_reg)
        if s > best_score:
            best_score = s
            best_poly = v_reg.copy()

    # Objective function
    def _objective(params, temp=0.05):
        if time.time() > deadline:
            return 100.0
        L = params[:n]
        D = params[n:]
        
        # Bounds check
        if np.any(L < 0.1) or np.any(L > 20.0):
            return 10.0
        if np.any(D < -0.3) or np.any(D > 0.3):
            return 10.0
            
        poly = _polygon_from_edges(L, D)
        if poly is None:
            return 10.0
            
        s = _soft_min_vertex_score(poly, temperature=temp)
        return -s

    # Phase 1: L-BFGS-B
    x0 = np.concatenate([L0, D0])
    
    try:
        result = minimize(
            _objective,
            x0,
            args=(0.05,),
            method='L-BFGS-B',
            bounds=[(0.1, 20.0)] * n + [(-0.3, 0.3)] * (n-1),
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        L_opt = result.x[:n]
        D_opt = result.x[n:]
        poly_opt = _polygon_from_edges(L_opt, D_opt)
        
        if poly_opt is not None:
            s = _compute_min_vertex_score(poly_opt)
            if s > best_score:
                best_score = s
                best_poly = poly_opt.copy()
                
    except Exception as e:
        pass

    if time.time() > deadline - 100:
        return best_poly if best_poly is not None else _regular_polygon()

    # Phase 2: Polish with Nelder-Mead on a smaller scale around the best solution
    # We need to map the parameter space to ensure we stay within bounds
    # NM doesn't handle bounds well, so we use a transformation or just run it and check
    
    def _objective_nm(params, temp=0.02):
        L = params[:n]
        D = params[n:]
        
        # Soft bounds
        L_pen = 0
        if np.any(L < 0.1): L_pen += 10 * np.sum(np.clip(0.1 - L, 0, None)**2)
        if np.any(L > 20.0): L_pen += 10 * np.sum(np.clip(L - 20.0, 0, None)**2)
        D_pen = 0
        if np.any(D < -0.3): D_pen += 10 * np.sum(np.clip(-0.3 - D, 0, None)**2)
        if np.any(D > 0.3): D_pen += 10 * np.sum(np.clip(D - 0.3, 0, None)**2)
        
        poly = _polygon_from_edges(L, D)
        if poly is None:
            return 10.0 + L_pen + D_pen
            
        s = _soft_min_vertex_score(poly, temperature=temp)
        return -s + L_pen + D_pen

    x0_nm = np.concatenate([L_opt, D_opt])
    
    try:
        result_nm = minimize(
            _objective_nm,
            x0_nm,
            args=(0.02,),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        
        L_nm = result_nm.x[:n]
        D_nm = result_nm.x[n:]
        poly_nm = _polygon_from_edges(L_nm, D_nm)
        
        if poly_nm is not None:
            s = _compute_min_vertex_score(poly_nm)
            if s > best_score:
                best_score = s
                best_poly = poly_nm.copy()
                
    except Exception as e:
        pass

    if best_poly is None:
        # Fallback
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_poly = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        
    return best_poly

def _regular_polygon():
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)
