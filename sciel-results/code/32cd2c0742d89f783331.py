"""Optimized hexagon packing using smooth repulsion potential + L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

# Precompute unit hexagon vertices (flat-top orientation, side=1)
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees
_HEX_ANGLES = np.deg2rad(np.arange(6) * 60.0)
_HEX_VERTS = np.column_stack([np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)])

# Support function of unit hexagon in direction (ux, uy)
# For a regular hexagon with vertices at unit distance from center,
# support in direction u = max over vertices of dot(v, u)
def _hex_support(ux, uy):
    dots = _HEX_VERTS[0] * ux + _HEX_VERTS[1] * uy
    return np.max(dots)

def _hex_support_vec(ux, uy):
    """Vectorized support for arrays of ux, uy."""
    dots = _HEX_VERTS[0][:, None] * ux[None, :] + _HEX_VERTS[1][:, None] * uy[None, :]
    return np.max(dots, axis=0)

def _collision_distance(c1, c2, a1, a2):
    """
    Minimum distance between two unit hexagons at centers c1, c2 with angles a1, a2.
    Positive = separated, negative = overlapping.
    Uses support function approach: distance = max over directions of (support1 + support2 - |c1-c2|_dir)
    Actually: two convex bodies separated iff for ALL directions u, 
    support_A(u) + support_B(-u) <= |c1-c2|
    Collision depth = max over u of (support_A(u) + support_B(-u) - |c1-c2|)
    """
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    d = np.sqrt(dx*dx + dy*dy)
    if d < 1e-12:
        return 0.0  # centers coincide, definitely overlapping
    
    # Check 12 directions: 6 for hex1, 6 for hex2 (negated)
    # For hex at angle a, support in direction (ux,uy) = max_v dot(R(a)*v, (ux,uy))
    # = max_v dot(v, R(-a)*(ux,uy))
    # So support_hex(a, u) = _hex_support(u_x*cos(a)-u_y*sin(a), u_x*sin(a)+u_y*cos(a))
    
    # Directions to check: for each vertex normal of hex1 and hex2
    # Vertex normals of unit hex (flat-top): directions to vertices
    # Actually for a regular hexagon, the support function's maximum changes at the edge normals
    # Edge normals are at 30, 90, 150, 210, 270, 330 degrees for flat-top
    
    # Simpler: check all 12 vertex directions of both hexagons
    # For hex1 rotated by a1: vertex directions are a1 + 0, a1+60, ..., a1+300
    # For hex2 rotated by a2: vertex directions are a2 + 0, ..., a2+300
    # But we need support in the direction FROM c2 TO c1 and vice versa
    
    # The separation condition: for all unit vectors u,
    # h1(u) + h2(-u) <= |c1 - c2|
    # where h1(u) = max_{v in hex1} dot(v, u)
    
    # Critical directions are vertex normals of both hexagons
    # Hex1 vertex normals (outward): at angles a1 + k*60
    # Hex2 vertex normals: at angles a2 + k*60
    
    # For the separation, we need:
    # h1(u) + h2(-u) <= d for all u
    # The max of h1(u) + h2(-u) - d over u gives the overlap
    
    # Critical u values: directions where h1 or h2 has a kink
    # h1(u) has kinks at u perpendicular to edges of hex1
    # For flat-top hex, edges are at angles 30, 90, 150, 210, 270, 330 from center
    # Edge normals (outward) at: 0, 60, 120, 180, 240, 300 (vertex directions)
    # Wait, let me think again.
    
    # Flat-top hexagon vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    # Edge from v0 to v1: direction (0.5-sqrt(3)/2*0... let me just use the vertex directions as normals
    
    # For a regular hexagon, the support function h(u) = max dot(v_i, u)
    # The maximum switches between vertices when u is perpendicular to an edge
    # The edge between v_i and v_{i+1} is perpendicular to the bisector direction
    # For our hex, the "normal" directions where support switches are at 30, 90, 150, 210, 270, 330
    
    # But for the sum h1(u) + h2(-u), critical directions include:
    # - Directions where h1 changes: 30+a1, 90+a1, ...
    # - Directions where h2 changes: 30+a2, 90+a2, ...
    
    # Let's just check these 12 directions plus the direction of (c1-c2)
    a1r = np.deg2rad(a1)
    a2r = np.deg2rad(a2)
    
    # Edge normal angles for flat-top hex (offset by 30 from vertex angles)
    edge_normals_1 = a1r + np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    edge_normals_2 = a2r + np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    
    all_angles = np.concatenate([edge_normals_1, edge_normals_2])
    
    ux = np.cos(all_angles)
    uy = np.sin(all_angles)
    
    # h1(u) for hex1 at angle a1:
    # h1(u) = max_i dot(R(a1)*v_i, u) = max_i dot(v_i, R(-a1)*u)
    cos_a1 = np.cos(a1r)
    sin_a1 = np.sin(a1r)
    # R(-a1)*u = (ux*cos(a1)+uy*sin(a1), -ux*sin(a1)+uy*cos(a1))
    h1 = _hex_support_vec(ux*cos_a1 + uy*sin_a1, -ux*sin_a1 + uy*cos_a1)
    
    # h2(-u) for hex2 at angle a2:
    # h2(-u) = max_i dot(R(a2)*v_i, -u) = -min_i dot(R(a2)*v_i, u) = -h2(u) ... no
    # h2(-u) = max_i dot(R(a2)*v_i, -u) = max_i (-dot(R(a2)*v_i, u)) = -min_i dot(R(a2)*v_i, u)
    # But min dot = -max dot(-v_i, ... ) ... 
    # Actually h2(-u) = max_i dot(R(a2)*v_i, -u) = max_i dot(R(a2)*(-v_i), u)
    # Since hexagon is centrally symmetric, -v_i is also a vertex
    # So h2(-u) = h2(u) for centrally symmetric body! No wait, that's not right.
    # h2(-u) = max_i dot(w_i, -u) where w_i = R(a2)*v_i
    # = -min_i dot(w_i, u)
    # For a centrally symmetric convex body centered at origin, min dot(w_i, u) = -max dot(w_i, -u) = -h2(-u)... circular
    # Let me just compute directly: h2(-u) = max_i dot(R(a2)*v_i, (-ux, -uy))
    cos_a2 = np.cos(a2r)
    sin_a2 = np.sin(a2r)
    # R(-a2)*(-u) = (-ux*cos(a2)-uy*sin(a2), ux*sin(a2)-uy*cos(a2))
    h2_neg_u = _hex_support_vec(-ux*cos_a2 - uy*sin_a2, ux*sin_a2 - uy*cos_a2)
    
    # Separation in direction u: the "gap" is d - (h1(u) + h2(-u))
    # Wait, I need to be more careful.
    # Body 1 is at c1, body 2 is at c2.
    # For body 1: points are c1 + R(a1)*v for v in unit hex
    # For body 2: points are c2 + R(a2)*w for w in unit hex
    # They are separated iff for all u: max_{p in body1} dot(p,u) <= min_{q in body2} dot(q,u)
    # max_{p in body1} dot(p,u) = dot(c1,u) + h1(u)
    # min_{q in body2} dot(q,u) = dot(c2,u) - h2(-u)  [since min dot(q,u) = dot(c2,u) + min dot(R(a2)w, u) = dot(c2,u) - h2(-u)]
    # So separated iff: dot(c1,u) + h1(u) <= dot(c2,u) - h2(-u) for all u
    # i.e., h1(u) + h2(-u) <= dot(c2-c1, u) = dot(c2-c1, u)
    # Let d_vec = c2 - c1, then we need h1(u) + h2(-u) <= dot(d_vec, u) for all u
    # Overlap = max_u [h1(u) + h2(-u) - dot(d_vec, u)]
    
    d_vec = np.array([c2[0] - c1[0], c2[1] - c1[1]])
    dot_du = d_vec[0]*ux + d_vec[1]*uy
    
    overlap = np.max(h1 + h2_neg_u - dot_du)
    
    # Also check the reverse: we need h2(u) + h1(-u) <= dot(c1-c2, u)
    # This is equivalent by symmetry, but let's verify with the same computation
    # Actually the condition is symmetric, so one check suffices if we're careful
    # Let me just return the max overlap
    return overlap

def _objective(params):
    """
    params: 3N + 4 = [x1,y1,a1, x2,y2,a2, ..., xN,yN,aN, cx,cy, S, theta]
    Objective: S + penalty for collisions
    """
    start = time.time()
    
    positions = params[:3*N].reshape(N, 3)
    cx, cy, S, theta = params[3*N:3*N+4]
    
    # Check outer hexagon containment
    # Outer hexagon: center (cx,cy), side S, rotation theta
    # A point p is inside if for all 6 edge normals n_i: dot(p - center, n_i) <= S * apothem
    # Apothem = S * sqrt(3)/2
    # Edge normals for flat-top hex at rotation theta:
    # The 6 outward normals are at angles theta + 30, theta+90, ... (edge normals)
    # Actually for a regular hexagon with side S, the distance from center to each edge is S*sqrt(3)/2
    # The 6 edge outward normal directions (for flat-top, theta=0): 
    #   30, 90, 150, 210, 270, 330 degrees
    # Wait, let me reconsider. Flat-top hex has vertices at 0,60,...,300
    # Edges connect consecutive vertices. The outward normal of each edge points away from center.
    # Edge between vertex at 0 and vertex at 60: midpoint direction is 30 degrees
    # So outward normals are at 30, 90, 150, 210, 270, 330 for flat-top.
    
    theta_r = np.deg2rad(theta)
    edge_normal_angles = theta_r + np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    en_x = np.cos(edge_normal_angles)
    en_y = np.sin(edge_normal_angles)
    
    apothem = S * np.sqrt(3) / 2
    
    total_penalty = 0.0
    
    # Check each inner hexagon is inside outer
    for i in range(N):
        px, py, pa = positions[i]
        pa_r = np.deg2rad(pa)
        
        # For inner hex at (px,py) with angle pa, side 1:
        # Its 6 vertices are at (px,py) + R(pa)*v_j
        # All vertices must satisfy: dot(vertex - (cx,cy), n_k) <= apothem for all k
        # Equivalently: dot((px,cx), n_k) + max_j dot(R(pa)*v_j, n_k) <= apothem
        # max_j dot(R(pa)*v_j, n_k) = support of inner hex in direction n_k
        # = _hex_support(n_kx*cos(pa)+n_ky*sin(pa), -n_kx*sin(pa)+n_ky*cos(pa))
        
        cos_pa = np.cos(pa_r)
        sin_pa = np.sin(pa_r)
        
        for k in range(6):
            nkx, nky = en_x[k], en_y[k]
            # Support of inner hex in direction n_k
            sup = _hex_support(nkx*cos_pa + nky*sin_pa, -nkx*sin_pa + nky*cos_pa)
            # Center contribution
            center_dot = (px - cx)*nkx + (py - cy)*nky
            if center_dot + sup > apothem:
                total_penalty += (center_dot + sup - apothem) ** 2
    
    # Check pairwise collisions between inner hexagons
    for i in range(N):
        for j in range(i+1, N):
            c1 = positions[i][:2]
            c2 = positions[j][:2]
            a1 = positions[i][2]
            a2 = positions[j][2]
            
            # Quick reject
            dx = c1[0] - c2[0]
            dy = c1[1] - c2[1]
            d = np.sqrt(dx*dx + dy*dy)
            if d > 4.0:  # Max distance between two unit hexagons is 2*sqrt(3) ~ 3.46
                continue
            
            overlap = _collision_distance(c1, c2, a1, a2)
            if overlap > 0:
                total_penalty += overlap ** 2
    
    # Objective: S + penalty
    obj = S + 10.0 * total_penalty
    
    if time.time() - start > 110:
        # Return a large value to signal timeout
        return 1e6
    
    return obj

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_obj = np.inf
    best_params = None
    
    # Initial guess: place hexagons in a reasonable pattern
    # Try a few different initial configurations
    
    np.random.seed(42)
    
    # Configuration 1: linear arrangement
    def init_linear():
        params = np.zeros(3*N + 4)
        for i in range(N):
            params[3*i] = (i - N/2 + 0.5) * 2.2
            params[3*i+1] = 0.0
            params[3*i+2] = 0.0
        params[3*N] = 0.0  # cx
        params[3*N+1] = 0.0  # cy
        params[3*N+2] = 15.0  # S initial
        params[3*N+3] = 0.0  # theta
        return params
    
    # Configuration 2: two rows
    def init_two_rows():
        params = np.zeros(3*N + 4)
        for i in range(N):
            row = i // 6
            col = i % 6
            params[3*i] = (col - 2.5) * 2.1
            params[3*i+1] = (row - 0.5) * 3.5
            params[3*i+2] = 0.0
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 12.0
        params[3*N+3] = 0.0
        return params
    
    # Configuration 3: circular arrangement
    def init_circular():
        params = np.zeros(3*N + 4)
        for i in range(N):
            angle = 2*np.pi*i/N
            r = 3.0
            params[3*i] = r*np.cos(angle)
            params[3*i+1] = r*np.sin(angle)
            params[3*i+2] = 0.0
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 10.0
        params[3*N+3] = 0.0
        return params
    
    initializations = [init_linear, init_two_rows, init_circular]
    
    for init_fn in initializations:
        if time.time() > deadline:
            break
        
        x0 = init_fn()
        
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
                
        except Exception:
            continue
        
        if time.time() > deadline - 10:
            break
    
    if best_params is None:
        # Fallback: return the seed-like answer
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract results
    positions = best_params[:3*N].reshape(N, 3)
    cx, cy, S, theta = best_params[3*N:3*N+4]
    
    # Ensure S is reasonable (at least the minimum needed)
    S = max(S, 3.0)
    
    inner = [[float(positions[i][0]), float(positions[i][1]), float(positions[i][2])] for i in range(N)]
    
    return inner, (float(cx), float(cy)), float(S), float(theta)
