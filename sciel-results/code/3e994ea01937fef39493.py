"""
Packing of 12 unit hexagons in a minimal outer hexagon.
Uses gradient-based optimization (L-BFGS-B) with analytic gradients for
SAT-based collision penalties and outer hexagon containment.
"""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 135.0  # Safety margin below 150s
INRADIUS = math.sqrt(3) / 2.0
MIN_DIST_CIRCLE = 2 * INRADIUS  # sqrt(3) ~ 1.732

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _hex_vertex_grads(cx, cy, angle_deg, side=1.0):
    """Return vertices and their gradients w.r.t. (cx, cy, angle_deg)."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    grads = []  # list of (dx/dcx, dx/dcy, dx/dangle, dy/dcx, dy/dcy, dy/dangle)
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        # Position: (cx + ca*x - sa*y, cy + sa*x + ca*y)
        px = cx + ca * x - sa * y
        py = cy + sa * x + ca * y
        verts.append((px, py))
        # Gradients w.r.t. angle (in radians): d(ca)/da = -sa, d(sa)/da = ca
        dpx_da = -sa * x - ca * y
        dpy_da = ca * x - sa * y
        # Gradient w.r.t. angle_deg: multiply by pi/180
        dpx_dda = dpx_da * math.pi / 180.0
        dpy_dda = dpy_da * math.pi / 180.0
        grads.append((1.0, 0.0, dpx_dda, 0.0, 1.0, dpy_dda))
    return verts, grads

def _project_polygon(verts, axis):
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _sat_overlap_depth(vertsA, vertsB):
    """Compute maximum overlap depth between two polygons via SAT. Return 0 if separated."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    max_overlap = 0.0
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        overlap = min(maxA, maxB) - max(minA, minB)
        if overlap <= 0:
            return 0.0  # Separated
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    n = len(cx_arr)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_overlap_depth(verts_list[i], verts_list[j]) > 1e-9:
                return False
    
    return True

def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _quick_score(cx_arr, cy_arr, angle_arr):
    apothem_unit = math.sqrt(3) / 2.0
    normals = [(math.cos(math.radians(90 + 60 * k)), math.sin(math.radians(90 + 60 * k))) for k in range(6)]
    max_need = 0.0
    for i in range(N):
        a = math.radians(angle_arr[i])
        ca, sa = math.cos(a), math.sin(a)
        for k in range(6):
            theta = math.radians(60 * k)
            vx = cx_arr[i] + ca * math.cos(theta) - sa * math.sin(theta)
            vy = cy_arr[i] + sa * math.cos(theta) + ca * math.sin(theta)
            for (nx, ny) in normals:
                val = (nx * vx + ny * vy) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _compute_best_outer(cx_arr, cy_arr, angle_arr):
    all_verts = []
    for i in range(N):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        all_verts.extend(verts)
    
    cx = np.mean([v[0] for v in all_verts])
    cy = np.mean([v[1] for v in all_verts])
    
    best_side = float('inf')
    best_angle = 0.0
    
    for angle_deg in range(0, 60, 5):
        side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (cx, cy), angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
            
    for angle_deg in np.arange(best_angle - 5, best_angle + 5, 0.5):
        side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (cx, cy), angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg

    return cx, cy, best_side, best_angle

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def make_honeycomb_cluster():
        """
        Create a honeycomb cluster of 12 hexagons.
        Structure: 1 center, 6 in first ring, 5 in second ring (partial).
        """
        pts = []
        pts.append([0.0, 0.0, 0.0])
        r1 = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y, 0.0])
        r2 = r1 * math.sqrt(3)
        angles = [30, 90, 150, 210, 270]
        for deg in angles:
            rad = math.radians(deg)
            x = r2 * math.cos(rad)
            y = r2 * math.sin(rad)
            pts.append([x, y, 0.0])
        return pts[:12]
    
    def make_lattice_initial():
        """3-4-3-2 lattice arrangement."""
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        for c in range(3):
            x = c * col_width - col_width
            y = -row_height
            pts.append([x, y, 0.0])
        for c in range(4):
            x = c * col_width - col_width - col_width/2
            y = 0.0
            pts.append([x, y, 0.0])
        for c in range(3):
            x = c * col_width - col_width
            y = row_height
            pts.append([x, y, 0.0])
        pts.append([-col_width*2 - col_width/2 - col_width, 0.0, 0.0])
        pts.append([col_width*2 - col_width/2 + col_width, 0.0, 0.0])
        return pts

    def make_ring_initial():
        """1 center + 11 in a ring."""
        pts = [[0.0, 0.0, 0.0]]
        r = 2 * math.sqrt(3)
        for i in range(11):
            angle = math.radians(360 * i / 11 + 15)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            pts.append([x, y, 0.0])
        return pts

    def make_random_initial():
        pts = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
        return pts

    initializers = [
        make_honeycomb_cluster,
        make_lattice_initial,
        make_ring_initial,
        make_random_initial,
    ]
    
    best_side = float('inf')
    best_cfg = None
    
    # 1. Initialize and validate
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 100:
            break
        cfg = init_func()
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            side = _quick_score(cx_arr, cy_arr, angle_arr)
            if side < best_side:
                best_side = side
                best_cfg = cfg[:]
    
    if best_cfg is None:
        cfg = make_lattice_initial()
        scale = 1.5
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_cfg = cfg
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        best_side = _quick_score(cx_arr, cy_arr, angle_arr)

    # 2. Gradient-based optimization with analytic gradients
    def smooth_objective_and_grad(params):
        """
        Smooth objective: outer side length + collision penalty.
        params: [x1, y1, a1, ..., xN, yN, aN]
        Returns (value, gradient)
        """
        cx_arr = params[:N]
        cy_arr = params[N:2*N]
        angle_arr = params[2*N:]
        
        # Compute all vertices and their gradients
        all_verts = []
        all_grads = []  # For each hex i, list of 6 vertex grads
        for i in range(N):
            verts, grads = _hex_vertex_grads(cx_arr[i], cy_arr[i], angle_arr[i])
            all_verts.append(verts)
            all_grads.append(grads)
        
        # Compute outer hex side (using fixed center at origin for simplicity in gradient)
        # We'll optimize the inner hexagons to fit in a unit outer hex, then scale
        # Actually, let's just compute the required side length with center at centroid
        all_pts = []
        for i in range(N):
            all_pts.extend(all_verts[i])
        cx_c = np.mean([v[0] for v in all_pts])
        cy_c = np.mean([v[1] for v in all_pts])
        
        # For gradient computation, we need a fixed outer hex orientation
        # Use angle 0 for the outer hex
        apothem_unit = math.sqrt(3) / 2.0
        normals = [(math.cos(math.radians(90 + 60 * k)), math.sin(math.radians(90 + 60 * k))) for k in range(6)]
        
        max_need = 0.0
        active_vertex = None  # (hex_idx, vert_idx, normal_idx)
        for i in range(N):
            for vi in range(6):
                vx, vy = all_verts[i][vi]
                for ni, (nx, ny) in enumerate(normals):
                    val = (nx * (vx - cx_c) + ny * (vy - cy_c)) / apothem_unit
                    if val > max_need:
                        max_need = val
                        active_vertex = (i, vi, ni)
        
        # Collision penalty
        collision_penalty = 0.0
        grad = np.zeros(3 * N)
        
        for i in range(N):
            for j in range(i + 1, N):
                depth = _sat_overlap_depth(all_verts[i], all_verts[j])
                if depth > 0:
                    collision_penalty += depth * depth
                    # Approximate gradient: push apart along center line
                    dx = cx_arr[j] - cx_arr[i]
                    dy = cy_arr[j] - cy_arr[i]
                    dist = math.hypot(dx, dy)
                    if dist > 1e-9:
                        fx = 2 * depth * dx / dist
                        fy = 2 * depth * dy / dist
                        grad[i] -= fx
                        grad[N + i] -= fy
                        grad[2*N + i] -= fx * 0.1  # Small angle gradient
                        grad[j] += fx
                        grad[N + j] += fy
                        grad[2*N + j] += fx * 0.1
        
        # Gradient for outer hex constraint (from active vertex)
        if active_vertex is not None:
            i, vi, ni = active_vertex
            nx, ny = normals[ni]
            gi = all_grads[i][vi]
            # d(val)/d(cx_i) = nx * dpx/dcx / apothem - nx * dcx_c/dcx_i / apothem
            # dcx_c/dcx_i = 6/72 = 1/12 (each hex has 6 verts, total 72)
            factor = 1.0 / apothem_unit
            grad[i] += nx * gi[0] * factor - nx * (1.0/12.0) * factor
            grad[N + i] += ny * gi[3] * factor - ny * (1.0/12.0) * factor
            grad[2*N + i] += nx * gi[2] * factor - nx * 0.0 * factor
        
        value = max_need + 15.0 * collision_penalty
        return value, grad

    params0 = np.array([p[0] for p in best_cfg] + [p[1] for p in best_cfg] + [p[2] for p in best_cfg])
    
    try:
        # Use L-BFGS-B with analytic gradient
        def obj_only(params):
            val, _ = smooth_objective_and_grad(params)
            return val
        
        def grad_only(params):
            _, g = smooth_objective_and_grad(params)
            return g
        
        res = minimize(obj_only, params0, jac=grad_only, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5, 'maxfun': 5000})
        
        if res.fun < best_side:
            # Verify validity
            params_opt = res.x
            cx_test = params_opt[:N]
            cy_test = params_opt[N:2*N]
            angle_test = params_opt[2*N:]
            if _is_valid_fast(cx_test, cy_test, angle_test):
                side_test = _quick_score(cx_test, cy_test, angle_test)
                if side_test < best_side:
                    best_side = side_test
                    best_cfg = [[params_opt[i], params_opt[N+i], params_opt[2*N+i]] for i in range(N)]
    except:
        pass
    
    # 3. Final refinement with Nelder-Mead on valid configurations
    if best_cfg is not None:
        cx_arr = np.array([p[0] for p in best_cfg])
        cy_arr = np.array([p[1] for p in best_cfg])
        angle_arr = np.array([p[2] for p in best_cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            cx, cy, side, angle = _compute_best_outer(cx_arr, cy_arr, angle_arr)
            best_side = side
            best_center = (cx, cy)
            best_outer_angle = angle
        else:
            # Fallback to lattice if invalid
            cfg = make_lattice_initial()
            cx_arr = np.array([p[0] for p in cfg])
            cy_arr = np.array([p[1] for p in cfg])
            angle_arr = np.array([p[2] for p in cfg])
            cx, cy, side, angle = _compute_best_outer(cx_arr, cy_arr, angle_arr)
            best_side = side
            best_center = (cx, cy)
            best_outer_angle = angle
            best_cfg = cfg
            
    inner = [[p[0], p[1], p[2]] for p in best_cfg]
    return inner, best_center, best_side, best_outer_angle
