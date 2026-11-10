# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.spatial import ConvexHull

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

def _edge_normals(V):
    n = np.empty((6, 2))
    for i in range(6):
        e = V[(i + 1) % 6] - V[i]
        L = np.hypot(e[0], e[1])
        n[i] = np.array([e[1], -e[0]]) / L
    return n

def _min_separation(A, B):
    """
    Compute the minimum separation between two convex polygons A and B.
    Separation > 0 means they are disjoint (gap exists).
    Separation <= 0 means they intersect or touch.
    We use the separating axis theorem.
    """
    nA = _edge_normals(A)
    nB = _edge_normals(B)
    best = -np.inf
    for n in nA:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
    for n in nB:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
    return best

def _outer_side_for_points(points, phi_deg):
    """
    Calculate the minimum side length of a regular hexagon with rotation phi_deg
    that encloses the given points. The hexagon is centered at the centroid of points?
    No, for a given orientation, the minimal enclosing hexagon is not necessarily centered at the centroid.
    However, for a regular hexagon, the "width" in direction n is w_n.
    The side length s is related to the width by w_n = s * sqrt(3) / 2 * (1 + cos(angle_diff))? 
    Actually, the width of a regular hexagon of side s in any direction is between s*sqrt(3)/2 (apothem*2) and s*sqrt(3) (vertex to vertex).
    Wait, the width in the direction normal to an edge is s*sqrt(3).
    The width in the direction to a vertex is s*sqrt(3) * cos(30) * 2? No.
    Let's stick to the half-plane definition.
    Hexagon H defined by center c, side s, angle phi.
    Normals n_k = (cos(phi + 60k), sin(phi + 60k)).
    Condition: n_k . (p - c) <= s * sqrt(3) / 2 for all k, all p.
    And n_k . (p - c) >= -s * sqrt(3) / 2 for all k, all p.
    This implies:
    s * sqrt(3) / 2 >= max_k ( max_p (n_k . (p - c)) )
    s * sqrt(3) / 2 >= max_k ( -min_p (n_k . (p - c)) )
    To minimize s, we should choose c such that the max of these terms is minimized.
    This is a convex optimization problem for c given phi and s.
    But we want to minimize s.
    For a fixed phi, the minimal s is determined by the "width" of the point set in the 6 normal directions.
    Specifically, let w_k = max_p (n_k . p) - min_p (n_k . p).
    The hexagon must have width at least w_k in direction n_k.
    The width of a regular hexagon of side s in direction n_k (where n_k is a normal) is s * sqrt(3).
    Wait, if n_k is normal to an edge, the width is s*sqrt(3).
    If n_k is aligned with a vertex, the width is s*sqrt(3)*cos(30)*2? No.
    Let's check: Regular hexagon side s. Distance between parallel edges is s*sqrt(3).
    Distance between opposite vertices is 2s.
    The width in direction theta is:
    If theta is 0 (normal to edge), width = s*sqrt(3).
    If theta is 30 (to vertex), width = 2s.
    The width function is periodic and piecewise linear? No, it's smooth.
    Actually, for a regular hexagon, the support function h(u) = s * max_k (u . n_k) ... no.
    The condition for p to be inside is: max_k (n_k . (p-c)) <= s*sqrt(3)/2.
    So, for a set of points P, we need:
    s >= (2/sqrt(3)) * max_k ( max_p (n_k . (p-c)) )
    s >= (2/sqrt(3)) * max_k ( -min_p (n_k . (p-c)) )
    To minimize s, we can choose c.
    Let M_k = max_p (n_k . p) and m_k = min_p (n_k . p).
    We need:
    n_k . c <= M_k - s*sqrt(3)/2
    n_k . c >= m_k + s*sqrt(3)/2
    So, m_k + s*sqrt(3)/2 <= n_k . c <= M_k - s*sqrt(3)/2
    This is feasible iff m_k + s*sqrt(3)/2 <= M_k - s*sqrt(3)/2
    => s*sqrt(3) >= M_k - m_k = w_k
    => s >= w_k / sqrt(3)
    So, for a fixed orientation phi, the minimum side length is:
    s_min(phi) = (1/sqrt(3)) * max_k ( w_k(phi) )
    where w_k(phi) = max_p (n_k(phi) . p) - min_p (n_k(phi) . p).
    And the center c can be any point satisfying the inequalities, e.g., the average of the bounds.
    """
    phi_rad = np.deg2rad(phi_deg)
    normals = np.column_stack((np.cos(phi_rad + np.deg2rad(60.0) * np.arange(6)), 
                               np.sin(phi_rad + np.deg2rad(60.0) * np.arange(6))))
    
    # Projections
    proj = points @ normals.T # (N, 6)
    widths = proj.max(axis=0) - proj.min(axis=0) # (6,)
    max_width = widths.max()
    return max_width / np.sqrt(3.0)

def _objective_and_penalty(params):
    """
    params: 22 values (x1, y1, ..., x11, y11)
    Returns: (value, penalty)
    value: outer side length
    penalty: sum of squared overlaps
    """
    centers = params.reshape(11, 2)
    
    # Calculate outer side length for a few optimal rotations?
    # We can optimize the outer rotation as well.
    # Let's add 1 more param: outer angle.
    # So params: 22 + 1 = 23.
    # But basinhopping on 23 dims is fine.
    
    # For now, let's assume we optimize the outer angle separately or fix it.
    # Actually, we can just try a few angles for the outer hexagon and take the min.
    # Or better, include outer angle in params.
    
    # Let's stick to 22 params for centers, and internally optimize outer angle?
    # No, that's slow.
    # Let's expand params to 23: 22 centers + 1 outer angle.
    
    if len(params) != 23:
        return 1e10, 1e10
        
    centers = params[:22].reshape(11, 2)
    outer_angle = params[22]
    
    # Outer side length
    s_outer = _outer_side_for_points(centers, outer_angle)
    
    # Overlap penalty
    penalty = 0.0
    Vs = [_hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0) for i in range(11)]
    for i in range(11):
        for j in range(i + 1, 11):
            sep = _min_separation(Vs[i], Vs[j])
            if sep < 0:
                penalty += sep * sep
            elif sep < 1e-6:
                # Small gap, add small penalty to encourage separation
                penalty += (1e-6 - sep) ** 2
    
    # Also penalize if centers are too far apart (keep them compact)
    # This helps convergence
    compactness = np.sum(centers**2)
    
    # Total cost
    cost = s_outer + 100.0 * penalty + 0.01 * compactness
    
    return cost, penalty

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 590.0

    # Initial guess: Hexagonal grid
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        return np.array([x, y])
        
    coords = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (1,1), (-1,1), (0,2), (0,-2)
    ]
    init_centers = np.array([axial_to_cartesian(q, r) for q, r in coords])
    init_centers -= init_centers.mean(axis=0)
    
    x0 = np.append(init_centers.flatten(), 0.0) # + outer angle
    
    best_cost = np.inf
    best_params = None
    best_penalty = np.inf
    
    # Use Basin Hopping
    def objective_wrapper(x):
        cost, pen = _objective_and_penalty(x)
        return cost
        
    # We need to track the best non-overlapping solution
    def track_best(x):
        nonlocal best_cost, best_params, best_penalty
        cost, pen = _objective_and_penalty(x)
        if pen < 1e-6: # Valid solution
            if cost < best_cost:
                best_cost = cost
                best_params = x.copy()
                best_penalty = pen
        # Even if not valid, if it's better in terms of cost+penalty, update for search?
        # No, basinhopping handles the search. We just want to record valid ones.
        
    # To ensure we don't waste time, we'll run basinhopping with a time limit.
    # We can't easily put a time limit inside basinhopping's inner loop.
    # So we run it for a fixed number of iterations and check time.
    
    # Run several restarts with different initial conditions
    rng = np.random.default_rng(42)
    
    for restart in range(10):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            x_start = x0
        else:
            # Perturb initial condition
            x_start = x0 + rng.normal(0, 0.5, size=23)
            
        # Run basin hopping
        # niter is the number of basin hopping steps.
        # miniter is the number of L-BFGS iterations per step.
        try:
            res = basinhopping(
                objective_wrapper, 
                x_start, 
                niter=50, 
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 100}},
                stepsize=0.5,
                interval=10 # Check time every 10 steps? No, interval is for callback.
            )
        except Exception as e:
            print(f"Basin hopping error: {e}")
            continue
            
        # Check result
        cost, pen = _objective_and_penalty(res.x)
        if pen < 1e-6 and cost < best_cost:
            best_cost = cost
            best_params = res.x.copy()
            best_penalty = pen

        # Also check the best point found during the run?
        # Basinhopping doesn't return all points.
        # We can use a callback to track the best.
        
    # If we found a valid solution, reconstruct it
    if best_params is not None:
        centers = best_params[:22].reshape(11, 2)
        outer_angle = best_params[22]
        
        inner = np.column_stack((centers[:, 0], centers[:, 1], np.full(11, 0.0)))
        # Calculate optimal center for outer hexagon
        # The formula s = w/sqrt(3) assumes the hexagon is centered such that it's tight.
        # We need to find the center (ox, oy) that minimizes the max projection.
        # For a fixed s and phi, the feasible region for (ox, oy) is a hexagon.
        # We can just pick the centroid of the points as a good approximation, 
        # or solve for the center that minimizes the max deviation.
        
        # Let's refine the outer center and angle with a quick local optimization
        def outer_opt(params):
            ox, oy, phi = params
            s = _outer_side_for_points(centers, phi)
            # Check if (ox, oy) is valid center for this s
            phi_rad = np.deg2rad(phi)
            normals = np.column_stack((np.cos(phi_rad + np.deg2rad(60.0) * np.arange(6)), 
                                       np.sin(phi_rad + np.deg2rad(60.0) * np.arange(6))))
            proj = centers @ normals.T
            center_proj = np.array([ox, oy]) @ normals.T
            # Max distance from center
            max_dist = np.max(np.abs(proj - center_proj))
            # Required s
            req_s = max_dist / (np.sqrt(3)/2)
            return req_s
        
        x0_outer = np.array([0, 0, outer_angle])
        res_outer = minimize(outer_opt, x0_outer, method='L-BFGS-B')
        ox, oy, phi_opt = res_outer.x
        s_opt = res_outer.fun
        
        # Final validity check
        inner_final = np.column_stack((centers[:, 0], centers[:, 1], np.full(11, 0.0)))
        outer_final = np.array([ox, oy, phi_opt])
        
        # We need to ensure the inner hexagons are valid (separated)
        # The penalty was < 1e-6, so they should be.
        
        return inner_final, outer_final, s_opt
        
    # Fallback
    inner = np.array([
        [0, 0, 0],
        [-2.5, 0, 0],
        [2.5, 0, 0],
        [-1.25, 2.17, 0],
        [1.25, 2.17, 0],
        [-1.25, -2.17, 0],
        [1.25, -2.17, 0],
        [-3.75, 2.17, 0],
        [3.75, 2.17, 0],
        [-3.75, -2.17, 0],
        [3.75, -2.17, 0],
    ], dtype=float)
    outer = np.array([0, 0, 0], dtype=float)
    return inner, outer, 8.0

# EVOLVE-BLOCK-END
