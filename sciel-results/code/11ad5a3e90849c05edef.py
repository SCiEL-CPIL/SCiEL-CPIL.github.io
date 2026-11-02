import numpy as np
import time
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin

def _hex_vertices(center, side, angle):
    """Returns vertices of a regular hexagon."""
    angles = angle + np.arange(6) * np.pi / 3.0
    return center + side * np.column_stack([np.cos(angles), np.sin(angles)])

def _dist_point_to_segment(p, a, b):
    """Returns (distance, closest_point) from point p to segment ab."""
    ab = b - a
    ap = p - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq == 0:
        return np.linalg.norm(ap), a
    t = np.dot(ap, ab) / ab_len_sq
    t = np.clip(t, 0, 1)
    proj = a + t * ab
    dist = np.linalg.norm(p - proj)
    return dist, proj

def _min_dist_hex_hex_grad(H1, H2):
    """
    Computes min distance between two hexagons and the gradient with respect to H1's center and angle,
    and H2's center and angle.
    Returns: (dist, grad_H1_center, grad_H1_angle, grad_H2_center, grad_H2_angle)
    Note: We only need the gradient for the objective function which sums over pairs.
    """
    min_d = float('inf')
    best_grad1_c = np.zeros(2)
    best_grad1_a = 0.0
    best_grad2_c = np.zeros(2)
    best_grad2_a = 0.0
    
    # Vertices and edges
    V1 = H1['verts']
    V2 = H2['verts']
    
    # Check vertex of H1 to edge of H2
    for i in range(6):
        p = V1[i]
        a = V2[i]
        b = V2[(i + 1) % 6]
        d, proj = _dist_point_to_segment(p, a, b)
        if d < min_d:
            min_d = d
            # Gradient w.r.t p (which is V1[i])
            # grad_p = (p - proj) / d
            if d > 1e-8:
                grad_p = (p - proj) / d
            else:
                # If touching, gradient is undefined, use normal to edge
                edge_vec = b - a
                edge_len = np.linalg.norm(edge_vec)
                if edge_len > 1e-8:
                    normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
                    # Ensure normal points from H2 to H1
                    if np.dot(normal, p - (a+b)/2) < 0:
                        normal = -normal
                    grad_p = normal
                else:
                    grad_p = np.zeros(2)
            
            # Chain rule: p = C1 + R(angle1) * local_v
            # d p / d C1 = I
            # d p / d angle1 = perp(p - C1)
            local_v = p - H1['center']
            perp_local_v = np.array([-local_v[1], local_v[0]])
            
            best_grad1_c = grad_p
            best_grad1_a = np.dot(grad_p, perp_local_v)
            
            # Gradient w.r.t H2
            # The distance function is d(p, segment).
            # If p is outside the segment range, the closest point is a or b.
            # If p projects inside, the closest point is on the edge.
            # The gradient w.r.t the segment endpoints is complex.
            # Approximation: Assume the "active" feature is the edge itself.
            # If the projection is strictly inside the edge, moving the edge doesn't change the closest point locally?
            # Actually, if we move the edge, the distance changes.
            # For a smooth approximation, we can assume the gradient w.r.t H2 is -grad_p if the edge is fixed? No.
            # Let's use a simpler heuristic: The force is repulsive along the vector (proj - p).
            # grad w.r.t a and b depends on t.
            # To keep it simple and robust, we can just use the vector (proj - p) as the direction to move H2 away.
            # This is a valid subgradient for the distance function.
            best_grad2_c = -grad_p
            best_grad2_a = 0.0 # Rotation gradient for H2 is harder, set to 0 for now or approximate
            
    # Check vertex of H2 to edge of H1
    for i in range(6):
        p = V2[i]
        a = V1[i]
        b = V1[(i + 1) % 6]
        d, proj = _dist_point_to_segment(p, a, b)
        if d < min_d:
            min_d = d
            if d > 1e-8:
                grad_p = (p - proj) / d
            else:
                edge_vec = b - a
                edge_len = np.linalg.norm(edge_vec)
                if edge_len > 1e-8:
                    normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
                    if np.dot(normal, p - (a+b)/2) < 0:
                        normal = -normal
                    grad_p = normal
                else:
                    grad_p = np.zeros(2)
            
            # Gradient w.r.t H2 center
            local_v2 = p - H2['center']
            perp_local_v2 = np.array([-local_v2[1], local_v2[0]])
            
            best_grad1_c = -grad_p
            best_grad1_a = 0.0
            best_grad2_c = grad_p
            best_grad2_a = np.dot(grad_p, perp_local_v2)
            
    return min_d, best_grad1_c, best_grad1_a, best_grad2_c, best_grad2_a

def _min_dist_hex_outer_grad(H, S):
    """
    Computes min distance from hexagon H to outer hexagon boundary and gradient.
    Outer hexagon: Center (0,0), Side S, Rotation 0.
    Apothem A = S * sqrt(3) / 2.
    Normals n_k at angles k*pi/3.
    """
    A = S * np.sqrt(3) / 2.0
    min_d = float('inf')
    best_grad_c = np.zeros(2)
    best_grad_a = 0.0
    
    for k in range(6):
        angle = k * np.pi / 3.0
        n = np.array([np.cos(angle), np.sin(angle)])
        
        # Max dot product of H's vertices with n
        # H is convex, so max is at a vertex
        dots = np.dot(H['verts'], n)
        max_dot = np.max(dots)
        
        # Distance to boundary line n.x = A is A - max_dot
        d = A - max_dot
        
        if d < min_d:
            min_d = d
            # Gradient w.r.t H center
            # max_dot = n . (C + R a)
            # d(max_dot)/dC = n
            # So d(d)/dC = -n
            best_grad_c = -n
            
            # Gradient w.r.t angle
            # Let v be the active vertex.
            # max_dot = n . (C + R(theta) v_local)
            # d/dtheta = n . (perp(C + R v_local)) = n . perp(v_global)
            # v_global = C + R v_local
            # perp(v_global) is perpendicular to v_global
            # n . perp(v_global) = - n_perp . v_global ?
            # n = (cos, sin). perp(v) = (-vy, vx).
            # n . perp(v) = -cos*vy + sin*vx
            # This is the component of v perpendicular to n.
            
            # Find active vertex
            idx = np.argmax(dots)
            v_global = H['verts'][idx]
            perp_v = np.array([-v_global[1], v_global[0]])
            best_grad_a = np.dot(n, perp_v)
            
    return min_d, best_grad_c, best_grad_a

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        # Fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    # Binary search for S
    S_low = np.sqrt(11.0) * 1.01
    S_high = 10.0
    
    best_S = S_high
    best_inner = None
    
    def objective_and_grad(x, S):
        """
        x: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN]
        """
        centers = x[0::3]
        angles = x[2::3]
        
        hexes = []
        for i in range(N):
            c = np.array([centers[i], angles[i] * 0 + x[2*i+1]]) # Wait, x layout: [x, y, angle]
            # Let's use flat array: [x0, y0, a0, x1, y1, a1, ...]
            c = np.array([x[3*i], x[3*i+1]])
            a = x[3*i+2]
            hexes.append({
                'center': c,
                'angle': a,
                'verts': _hex_vertices(c, 1.0, a)
            })
        
        penalty = 0.0
        grad = np.zeros_like(x)
        
        # Boundary constraints
        for i in range(N):
            d, gc, ga = _min_dist_hex_outer_grad(hexes[i], S)
            if d < 0:
                penalty += d * d
                grad[3*i] += 2 * d * gc[0]
                grad[3*i+1] += 2 * d * gc[1]
                grad[3*i+2] += 2 * d * ga
        
        # Pairwise constraints
        for i in range(N):
            for j in range(i + 1, N):
                d, g1c, g1a, g2c, g2a = _min_dist_hex_hex_grad(hexes[i], hexes[j])
                if d < 0:
                    penalty += d * d
                    grad[3*i] += 2 * d * g1c[0]
                    grad[3*i+1] += 2 * d * g1c[1]
                    grad[3*i+2] += 2 * d * g1a
                    
                    grad[3*j] += 2 * d * g2c[0]
                    grad[3*j+1] += 2 * d * g2c[1]
                    grad[3*j+2] += 2 * d * g2a
        
        return penalty, grad

    def is_feasible(S, max_restarts=5):
        """
        Try to find a configuration with penalty close to 0.
        """
        for restart in range(max_restarts):
            if time.time() - START_TIME > TIME_LIMIT:
                return False, None
            
            # Initial guess
            if restart == 0:
                # Grid-like initialization
                x = np.zeros(N * 3)
                for i in range(N):
                    # Arrange in a hexagonal grid pattern
                    row = i // 6
                    col = i % 6
                    x[3*i] = col * 2.1
                    x[3*i+1] = row * 1.8
                    x[3*i+2] = 0.0
            else:
                # Random initialization within a box
                box_size = S
                x = np.random.uniform(-box_size/2, box_size/2, N * 3)
                x[2::3] = np.random.uniform(0, 2 * np.pi, N)
            
            res = minimize(
                lambda v: objective_and_grad(v, S)[0],
                x,
                jac=lambda v: objective_and_grad(v, S)[1],
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            final_penalty = res.fun
            if final_penalty < 1e-6:
                return True, res.x
            elif final_penalty < best_penalty_global[0]:
                best_penalty_global[0] = final_penalty
                best_config_global[0] = res.x
        
        return False, best_config_global[0]

    best_penalty_global = [float('inf')]
    best_config_global = [None]
    
    # Binary search
    for _ in range(20):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        S_mid = (S_low + S_high) / 2.0
        feasible, config = is_feasible(S_mid)
        
        if feasible:
            S_high = S_mid
            best_S = S_mid
            best_inner = config
        else:
            S_low = S_mid
            # If not feasible, keep the best config found for S_low?
            # We want the best S that IS feasible.
            # So we only update best_inner when feasible.
            
    # If best_inner is None, we failed to find a feasible config even for S_high.
    # Return a valid, non-overlapping configuration with a larger S.
    if best_inner is None:
        # Fallback: Linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Format output
    inner_list = []
    for i in range(N):
        inner_list.append([float(best_inner[3*i]), float(best_inner[3*i+1]), float(best_inner[3*i+2] * 180.0 / np.pi)])
    
    return inner_list, (0.0, 0.0), float(best_S), 0.0

# EVOLVE-BLOCK-START
def construct():
    # Call the main logic
    return _construct_impl()
# EVOLVE-BLOCK-END

# We need to wrap the logic in a function or keep it in construct.
# The prompt requires construct() to be the entry point.
# I will refactor to put everything in construct or a helper called by construct.

def _construct_impl():
    if time.time() - START_TIME > TIME_LIMIT:
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    S_low = np.sqrt(11.0) * 1.01
    S_high = 10.0
    
    best_S = S_high
    best_inner = None
    
    def objective_and_grad(x, S):
        centers = x[0::3]
        angles = x[2::3]
        
        hexes = []
        for i in range(N):
            c = np.array([x[3*i], x[3*i+1]])
            a = x[3*i+2]
            hexes.append({
                'center': c,
                'angle': a,
                'verts': _hex_vertices(c, 1.0, a)
            })
        
        penalty = 0.0
        grad = np.zeros_like(x)
        
        for i in range(N):
            d, gc, ga = _min_dist_hex_outer_grad(hexes[i], S)
            if d < 0:
                penalty += d * d
                grad[3*i] += 2 * d * gc[0]
                grad[3*i+1] += 2 * d * gc[1]
                grad[3*i+2] += 2 * d * ga
        
        for i in range(N):
            for j in range(i + 1, N):
                d, g1c, g1a, g2c, g2a = _min_dist_hex_hex_grad(hexes[i], hexes[j])
                if d < 0:
                    penalty += d * d
                    grad[3*i] += 2 * d * g1c[0]
                    grad[3*i+1] += 2 * d * g1c[1]
                    grad[3*i+2] += 2 * d * g1a
                    
                    grad[3*j] += 2 * d * g2c[0]
                    grad[3*j+1] += 2 * d * g2c[1]
                    grad[3*j+2] += 2 * d * g2a
        
        return penalty, grad

    def is_feasible(S, max_restarts=5):
        best_local_penalty = float('inf')
        best_local_config = None
        
        for restart in range(max_restarts):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            if restart == 0:
                x = np.zeros(N * 3)
                for i in range(N):
                    row = i // 6
                    col = i % 6
                    x[3*i] = col * 2.1
                    x[3*i+1] = row * 1.8
                    x[3*i+2] = 0.0
            else:
                box_size = S
                x = np.random.uniform(-box_size/2, box_size/2, N * 3)
                x[2::3] = np.random.uniform(0, 2 * np.pi, N)
            
            res = minimize(
                lambda v: objective_and_grad(v, S)[0],
                x,
                jac=lambda v: objective_and_grad(v, S)[1],
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            final_penalty = res.fun
            if final_penalty < best_local_penalty:
                best_local_penalty = final_penalty
                best_local_config = res.x
        
        if best_local_config is not None and best_local_penalty < 1e-6:
            return True, best_local_config
        else:
            return False, best_local_config

    for _ in range(20):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        S_mid = (S_low + S_high) / 2.0
        feasible, config = is_feasible(S_mid)
        
        if feasible:
            S_high = S_mid
            best_S = S_mid
            best_inner = config
        else:
            S_low = S_mid

    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner_list = []
    for i in range(N):
        inner_list.append([float(best_inner[3*i]), float(best_inner[3*i+1]), float(best_inner[3*i+2] * 180.0 / np.pi)])
    
    return inner_list, (0.0, 0.0), float(best_S), 0.0

# The actual construct function to be exposed
def construct():
    return _construct_impl()
