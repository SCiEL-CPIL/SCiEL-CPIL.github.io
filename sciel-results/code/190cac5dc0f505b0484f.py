"""Continuous gradient-based hexagon packing optimization."""
import time
import numpy as np
import random

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 140.0
SQRT3 = np.sqrt(3)

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def _min_side_and_grad(positions, angles, cx, cy, outer_angle):
    """Compute min outer side and its gradient w.r.t. positions and angles.
    The side is max over all vertices of linear functions, so grad is subgradient."""
    pos = np.asarray(positions, dtype=np.float64)
    ang = np.radians(np.asarray(angles, dtype=np.float64))
    base = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_ang = ang[:, None] + base[None, :]
    vx = pos[:, 0:1] + np.cos(total_ang)
    vy = pos[:, 1:2] + np.sin(total_ang)
    # Rotate into outer frame
    a = np.radians(outer_angle)
    ca, sa = np.cos(a), np.sin(a)
    dx = vx - cx
    dy = vy - cy
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # Six half-plane constraints: lx <= S, -lx <= S, lx+ly/sqrt3 <= S, etc.
    c = np.column_stack([
        lx, -lx,
        lx + ly / SQRT3, -lx + ly / SQRT3,
        -lx - ly / SQRT3, lx - ly / SQRT3
    ])  # shape (N*6, 6)
    S = np.max(c)
    # Subgradient: pick the argmax constraint
    idx = np.argmax(c, axis=1)
    # Gradient of constraint k w.r.t. (lx, ly)
    # c0: lx -> d/dlx=1, d/dly=0
    # c1: -lx -> -1, 0
    # c2: lx+ly/sqrt3 -> 1, 1/sqrt3
    # c3: -lx+ly/sqrt3 -> -1, 1/sqrt3
    # c4: -lx-ly/sqrt3 -> -1, -1/sqrt3
    # c5: lx-ly/sqrt3 -> 1, -1/sqrt3
    dcdlx = np.array([1, -1, 1, -1, -1, 1], dtype=np.float64)
    dcdly = np.array([0, 0, 1/SQRT3, 1/SQRT3, -1/SQRT3, -1/SQRT3], dtype=np.float64)
    # For the argmax vertex, gradient of S w.r.t. that vertex's (lx,ly)
    # But we need gradient w.r.t. original (x,y) and angle
    # This is complex; use numerical gradient instead for simplicity
    return float(S)

def _separation_penalty(positions, angles, mu=50.0):
    """Smooth penalty for overlaps: sum of exp(mu * overlap_depth) for each pair.
    Overlap depth approximated by negative of min distance between hexagons."""
    n = len(positions)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            # Approximate: distance between centers minus sum of "radii"
            # For hexagons, use inradius = sqrt(3)/2 * side
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            # Minimum center distance for non-overlap is 2*side = 2.0
            overlap = 2.0 - dist
            if overlap > 0:
                penalty += np.exp(mu * overlap)
    return penalty

def _objective(positions, angles, cx, cy, outer_angle, mu=50.0):
    """Total objective: outer side + overlap penalty."""
    S = _min_side_and_grad(positions, angles, cx, cy, outer_angle)
    pen = _separation_penalty(positions, angles, mu)
    return S + 0.01 * pen

def _grad_descent(positions, angles, cx, cy, outer_angle, time_limit, lr=0.05, mu=50.0):
    """Gradient descent with numerical gradients."""
    pos = np.asarray(positions, dtype=np.float64).copy()
    ang = np.asarray(angles, dtype=np.float64).copy()
    n = len(pos)
    
    best_obj = _objective(pos, ang, cx, cy, outer_angle, mu)
    best_pos = pos.copy()
    best_ang = ang.copy()
    
    # Numerical gradient computation
    def compute_grad(p, a):
        g = np.zeros_like(p)
        ga = np.zeros_like(a)
        eps = 1e-5
        for i in range(n):
            for d in range(2):
                p_plus = p.copy()
                p_plus[i, d] += eps
                f_plus = _objective(p_plus, a, cx, cy, outer_angle, mu)
                p_minus = p.copy()
                p_minus[i, d] -= eps
                f_minus = _objective(p_minus, a, cx, cy, outer_angle, mu)
                g[i, d] = (f_plus - f_minus) / (2 * eps)
            a_plus = a.copy()
            a_plus[i] += eps
            f_aplus = _objective(p, a_plus, cx, cy, outer_angle, mu)
            a_minus = a.copy()
            a_minus[i] -= eps
            f_aminus = _objective(p, a_minus, cx, cy, outer_angle, mu)
            ga[i] = (f_aplus - f_aminus) / (2 * eps)
        return g, ga
    
    step = lr
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < time_limit and no_improve < max_no_improve:
        g, ga = compute_grad(pos, ang)
        
        # Normalize gradient
        g_norm = np.sqrt(np.sum(g**2) + np.sum(ga**2))
        if g_norm < 1e-10:
            break
        
        # Gradient step
        pos_new = pos - step * g
        ang_new = ang - step * ga / 10.0  # Angle steps smaller
        
        obj_new = _objective(pos_new, ang_new, cx, cy, outer_angle, mu)
        
        if obj_new < best_obj:
            best_obj = obj_new
            best_pos = pos_new.copy()
            best_ang = ang_new.copy()
            no_improve = 0
            # Adaptive step increase
            step = min(step * 1.05, 0.5)
        else:
            no_improve += 1
            step *= 0.95
        
        pos = pos_new
        ang = ang_new
    
    return best_pos, best_ang, best_obj

def _initial_packing():
    """Hexagonal grid initial packing."""
    positions = []
    angles = []
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    # First ring
    for i in range(6):
        a = i * np.pi / 3
        positions.append([2.0 * np.cos(a), 2.0 * np.sin(a)])
        angles.append(0.0)
    # Second ring: 4 more
    for i in range(4):
        a = i * np.pi / 3
        positions.append([4.0 * np.cos(a), 4.0 * np.sin(a)])
        angles.append(0.0)
    return np.array(positions[:N], dtype=np.float64), np.array(angles[:N], dtype=np.float64)

def _multi_start_gradient(cx, cy, outer_angle, time_limit):
    """Multiple random restarts with gradient descent."""
    best_pos = None
    best_ang = None
    best_obj = float('inf')
    
    n_restarts = 0
    while time.time() < time_limit - 10.0:
        # Generate initial configuration
        if n_restarts == 0:
            pos, ang = _initial_packing()
        else:
            pos, ang = _initial_packing()
            # Random perturbation
            pos += random.gauss(0, 0.5, pos.shape)
            ang += random.gauss(0, 0.3, ang.shape)
        
        # Run gradient descent with time budget
        t_budget = min(30.0, time_limit - time.time() - 5.0)
        if t_budget < 5.0:
            break
        
        try:
            p_res, a_res, obj_res = _grad_descent(
                pos, ang, cx, cy, outer_angle,
                time_limit=time.time() + t_budget,
                lr=0.05, mu=50.0
            )
            if obj_res < best_obj:
                best_obj = obj_res
                best_pos = p_res
                best_ang = a_res
        except Exception:
            pass
        
        n_restarts += 1
    
    if best_pos is None:
        pos, ang = _initial_packing()
        return pos, ang, _objective(pos, ang, cx, cy, outer_angle)
    
    return best_pos, best_ang, best_obj

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-6):
    cx, cy = outer_center
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            # Check overlap
            dx = inner[j][0] - inner[i][0]
            dy = inner[j][1] - inner[i][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 2.0 - tol:
                return False
    # Check containment
    a = np.radians(outer_angle)
    ca, sa = np.cos(a), np.sin(a)
    for x, y, ang in inner:
        for k in range(6):
            theta = np.radians(ang) + k * np.pi / 3
            vx = x + np.cos(theta)
            vy = y + np.sin(theta)
            dx = vx - cx
            dy = vy - cy
            lx = ca * dx - sa * dy
            ly = sa * dx + ca * dy
            if lx > outer_side + tol or -lx > outer_side + tol:
                return False
            if lx + ly/SQRT3 > outer_side + tol or -lx + ly/SQRT3 > outer_side + tol:
                return False
            if -lx - ly/SQRT3 > outer_side + tol or lx - ly/SQRT3 > outer_side + tol:
                return False
    return True

def construct():
    global START_TIME, DEADLINE
    
    outer_cx, outer_cy = 0.0, 0.0
    outer_angle = 0.0
    
    # Main gradient-based optimization
    pos, ang, obj = _multi_start_gradient(
        outer_cx, outer_cy, outer_angle,
        time_limit=DEADLINE - 5.0
    )
    
    # Final side computation
    best_side = _min_side_and_grad(pos, ang, outer_cx, outer_cy, outer_angle)
    
    inner_list = [[float(pos[i][0]), float(pos[i][1]), float(ang[i])] for i in range(N)]
    
    # Verify
    if not _check_validity(inner_list, (outer_cx, outer_cy), best_side, outer_angle):
        # Fallback
        pos_init, ang_init = _initial_packing()
        inner_list = [[float(pos_init[i][0]), float(pos_init[i][1]), float(ang_init[i])] for i in range(N)]
        best_side = _min_side_and_grad(pos_init, ang_init, outer_cx, outer_cy, outer_angle)
    
    return inner_list, (float(outer_cx), float(outer_cy)), float(best_side), float(outer_angle)
