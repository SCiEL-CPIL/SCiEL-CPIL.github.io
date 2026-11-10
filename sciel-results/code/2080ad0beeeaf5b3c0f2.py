# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _signed_distance_pair(v1, v2):
    """Compute max of signed distances from edges of hex1 to edges of hex2 and vice versa.
    Returns negative if overlapping, positive if separated."""
    # For two convex polygons, they are separated iff there exists an edge normal
    # where the projections are disjoint. We compute the minimum gap over all 12 edge normals.
    # Actually, for non-intersection: for EVERY edge normal of both polygons, the projections
    # must be disjoint OR they touch. The separation condition is:
    # max over all 12 normals of (min_proj_other - max_proj_self) > 0
    # This is equivalent to: the polygons don't overlap.
    
    # Compute for each of the 6 edges of hex1: project both hexes onto the normal
    # For each edge of hex1, check if hex2 is entirely on the outside
    # For each edge of hex2, check if hex1 is entirely on the outside
    
    # A pair of convex polygons are disjoint iff there exists a separating axis.
    # For hexagons, the separating axes are the 12 edge normals (6 from each).
    
    # The "gap" along a normal n is: min(dot(v2, n)) - max(dot(v1, n))
    # If this is > 0 for ANY normal, they're separated.
    # The minimum gap over all separating axes that separate them gives the actual distance.
    
    # For our purposes, we need: the minimum over all 12 normals of the "overlap" to be <= 0
    # where overlap along normal n = max(dot(v1,n)) - min(dot(v2,n)) if n points from 1 to 2
    # Actually let me think more carefully.
    
    # Two convex polygons P1, P2 are disjoint iff there exists a unit normal n such that
    # max_{p in P1} dot(p,n) < min_{q in P2} dot(q,n)  OR  max_{p in P2} dot(p,n) < min_{q in P1} dot(q,n)
    
    # The separating axes for two hexagons are the 12 edge normals.
    # For each normal n (pointing outward from one hex), compute:
    #   gap = min_{q in P2} dot(q,n) - max_{p in P1} dot(p,n)
    # If gap > 0 for any n, they're separated.
    # The "penetration" is the minimum over all 12 normals of:
    #   max(0, max_{p in P1} dot(p,n) - min_{q in P2} dot(q,n))
    # But this isn't quite right either.
    
    # Let me use a different approach: compute the minimum distance between the two hexagons.
    # If min_distance > 0, they're disjoint.
    
    # Actually, for the barrier function, I'll compute: for each of the 12 edge normals,
    # the "violation" is how much the projections overlap. The pair is valid if at least
    # one normal has zero overlap (i.e., the projections are disjoint).
    
    # Simpler: compute the minimum over all 12 normals of the "separation" where
    # separation(n) = min(dot(v2,n)) - max(dot(v1,n)) for normals from v1, and
    # separation(n) = min(dot(v1,n)) - max(dot(v2,n)) for normals from v2.
    # The pair is disjoint iff max over all 12 of separation > 0.
    # The "penetration depth" is -max over all 12 of separation (if negative, they overlap).
    
    # Let me compute all 12 edge normals
    n1 = []
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1) % 6]
        e = p2 - p1
        n = np.array([-e[1], e[0]])
        n = n / (np.linalg.norm(n) + 1e-15)
        # Ensure normal points outward: check midpoint of edge
        mid = (p1 + p2) / 2
        center1 = np.mean(v1, axis=0)
        if np.dot(n, mid - center1) < 0:
            n = -n
        n1.append(n)
    
    n2 = []
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i+1) % 6]
        e = p2 - p1
        n = np.array([-e[1], e[0]])
        n = n / (np.linalg.norm(n) + 1e-15)
        center2 = np.mean(v2, axis=0)
        mid = (p1 + p2) / 2
        if np.dot(n, mid - center2) < 0:
            n = -n
        n2.append(n)
    
    all_normals = n1 + n2
    
    # For each normal, compute the separation
    # separation = min_{q in other} dot(q, n) - max_{p in self} dot(p, n)
    # But we need to be careful about direction.
    # For a normal from hex1, we want: min dot(v2, n) - max dot(v1, n) > 0 means separated
    # For a normal from hex2, we want: min dot(v1, n) - max dot(v2, n) > 0 means separated
    
    separations = []
    for i, n in enumerate(all_normals):
        if i < 6:
            # Normal from hex1
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            sep = np.min(proj2) - np.max(proj1)
        else:
            # Normal from hex2
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            sep = np.min(proj1) - np.max(proj2)
        separations.append(sep)
    
    # The pair is disjoint iff max(separations) > 0
    # The "penetration" is -max(separations) if max < 0, else 0
    max_sep = np.max(separations)
    penetration = -max_sep  # positive if overlapping
    
    return penetration

def _outer_constraint(v_outer, v_inner):
    """Check if inner hexagon is inside outer hexagon.
    Returns 0 if inside, positive if violating."""
    # For each edge of outer hex, check if all vertices of inner are on the correct side
    center = np.mean(v_outer, axis=0)
    violations = []
    for i in range(6):
        p1 = v_outer[i]
        p2 = v_outer[(i+1) % 6]
        e = p2 - p1
        n = np.array([-e[1], e[0]])
        n = n / (np.linalg.norm(n) + 1e-15)
        # Ensure normal points inward
        mid = (p1 + p2) / 2
        if np.dot(n, center - mid) < 0:
            n = -n
        # All inner vertices should satisfy dot(v - p1, n) >= 0
        dots = np.dot(v_inner - p1, n)
        min_dot = np.min(dots)
        if min_dot < 0:
            violations.append(-min_dot)
    
    return max(violations) if violations else 0.0

def _objective(params, n_inner=11):
    """Objective: minimize outer side length + barrier terms for constraints."""
    # params layout: [outer_x, outer_y, outer_angle, outer_side, then for each inner: x, y, angle]
    outer_x = params[0]
    outer_y = params[1]
    outer_angle = params[2]
    outer_side = params[3]
    
    if outer_side < 1.0:
        return 1e6
    
    v_outer = _hex_vertices(outer_x, outer_y, outer_side, outer_angle)
    
    total_barrier = 0.0
    for i in range(n_inner):
        ix = params[4 + 3*i]
        iy = params[5 + 3*i]
        iangle = params[6 + 3*i]
        v_inner = _hex_vertices(ix, iy, 1.0, iangle)
        
        # Outer constraint
        total_barrier += _outer_constraint(v_outer, v_inner) ** 2
    
    # Inner-inner constraints
    for i in range(n_inner):
        ix = params[4 + 3*i]
        iy = params[5 + 3*i]
        iangle = params[6 + 3*i]
        v_i = _hex_vertices(ix, iy, 1.0, iangle)
        
        for j in range(i+1, n_inner):
            jx = params[4 + 3*j]
            jy = params[5 + 3*j]
            jangle = params[6 + 3*j]
            v_j = _hex_vertices(jx, jy, 1.0, jangle)
            
            pen = _signed_distance_pair(v_i, v_j)
            if pen > 0:
                total_barrier += pen ** 2
    
    # Objective: minimize outer_side + barrier
    return outer_side + 100.0 * total_barrier

def _gradient(params, n_inner=11):
    """Numerical gradient."""
    eps = 1e-7
    grad = np.zeros_like(params)
    f0 = _objective(params, n_inner)
    for i in range(len(params)):
        params_p = params.copy()
        params_p[i] += eps
        grad[i] = (_objective(params_p, n_inner) - f0) / eps
    return grad

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580  # 580s to leave margin
    
    n_inner = 11
    
    # Initial guess: hexagonal arrangement
    # Center at origin, 6 around it, 4 more
    h = np.sqrt(3) / 2  # height of unit hexagon
    d = 2.0  # distance between centers of adjacent hexagons (side=1, so distance = 2*apothem = sqrt(3))
    # Actually for unit hexagons, the distance between centers when touching is 2 * apothem = sqrt(3)
    # Apothem of unit hex = sqrt(3)/2, so center-to-center = 2 * sqrt(3)/2 = sqrt(3)
    d = np.sqrt(3)
    
    # Hexagonal lattice positions
    # Row 0: center
    # Row 1: 6 neighbors
    # Row 2: 4 more
    
    init_positions = [
        [0, 0],
        [d, 0],
        [d/2, d*h],  # upper right
        [-d/2, d*h],  # upper left
        [-d, 0],
        [-d/2, -d*h],  # lower left
        [d/2, -d*h],   # lower right
        # 4 more: extend the pattern
        [3*d/2, d*h],   # far upper right
        [-3*d/2, d*h],  # far upper left
        [3*d/2, -d*h],  # far lower right
        [-3*d/2, -d*h], # far lower left
    ]
    
    # Build initial params
    params0 = np.zeros(4 + 3*n_inner)
    params0[0] = 0  # outer_x
    params0[1] = 0  # outer_y
    params0[2] = 0  # outer_angle
    params0[3] = 5.0  # outer_side initial guess
    
    for i in range(n_inner):
        params0[4 + 3*i] = init_positions[i][0]
        params0[5 + 3*i] = init_positions[i][1]
        params0[6 + 3*i] = 0  # angle
    
    # Use L-BFGS-B with numerical gradient
    best_params = params0.copy()
    best_obj = _objective(params0, n_inner)
    
    # Try multiple restarts with different initial configurations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            p0 = params0.copy()
        else:
            # Perturb the initial guess
            p0 = params0.copy()
            np.random.seed(42 + restart)
            for i in range(n_inner):
                p0[4 + 3*i] += np.random.uniform(-0.5, 0.5)
                p0[5 + 3*i] += np.random.uniform(-0.5, 0.5)
                p0[6 + 3*i] = np.random.uniform(0, 60)
            p0[3] = 4.5 + restart * 0.2
        
        try:
            result = minimize(
                _objective, p0, args=(n_inner,),
                method='L-BFGS-B',
                jac=_gradient,
                options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception as e:
            pass
        
        if time.time() > deadline - 30:
            break
    
    # Extract best solution
    outer_x = best_params[0]
    outer_y = best_params[1]
    outer_angle = best_params[2]
    outer_side = best_params[3]
    
    inner_hex_data = np.zeros((n_inner, 3))
    for i in range(n_inner):
        inner_hex_data[i, 0] = best_params[4 + 3*i]
        inner_hex_data[i, 1] = best_params[5 + 3*i]
        inner_hex_data[i, 2] = best_params[6 + 3*i]
    
    outer_hex_data = np.array([outer_x, outer_y, outer_angle])
    
    # Ensure outer_side is reasonable
    if outer_side < np.sqrt(11) - 0.01:
        outer_side = np.sqrt(11) + 0.01
    
    return inner_hex_data, outer_hex_data, float(outer_side)

# EVOLVE-BLOCK-END
