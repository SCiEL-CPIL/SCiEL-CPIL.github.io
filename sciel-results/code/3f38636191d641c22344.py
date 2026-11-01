import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

def hex_vertices(cx, cy, side, angle_rad):
    """Return vertices of a regular hexagon as (6,2) array."""
    angles = angle_rad + np.radians(np.arange(0, 360, 60))
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def polygons_disjoint_sat(poly1, poly2, tol=1e-9):
    """Check if two convex polygons are disjoint using SAT. Returns True if disjoint."""
    for poly in (poly1, poly2):
        for i in range(len(poly)):
            v1 = poly[i]
            v2 = poly[(i + 1) % len(poly)]
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            proj1 = poly1 @ normal
            proj2 = poly2 @ normal
            if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                return True
    return False

def all_hexagons_disjoint(hexes, tol=1e-9):
    """Check all pairs of hexagons are disjoint."""
    n = len(hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not polygons_disjoint_sat(hexes[i], hexes[j], tol):
                return False
    return True

def required_outer_side(inner_hexes, outer_center, outer_angle_rad):
    """Compute minimum outer hexagon side to contain all inner hexagons."""
    oc = np.array(outer_center)
    max_apothem = 0.0
    for i in range(N):
        for v in inner_hexes[i]:
            rel = v - oc
            for k in range(6):
                a = outer_angle_rad + k * np.pi / 3
                d = rel[0] * np.cos(a) + rel[1] * np.sin(a)
                if d > max_apothem:
                    max_apothem = d
    return max_apothem * 2.0 / np.sqrt(3.0)

def objective(params):
    """Objective: outer side length with penalty for overlaps."""
    inner_x = params[0:N]
    inner_y = params[N:2*N]
    inner_a = params[2*N:3*N]
    outer_cx = params[3*N]
    outer_cy = params[3*N + 1]
    outer_a = params[3*N + 2]
    
    hexes = []
    for i in range(N):
        hexes.append(hex_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i]))
    
    if not all_hexagons_disjoint(hexes, tol=1e-9):
        return 1e6
    
    return required_outer_side(hexes, (outer_cx, outer_cy), outer_a)

def make_initial_config(type_id):
    """Create diverse initial configurations."""
    np.random.seed(42 + type_id)
    if type_id == 0:
        # Hexagonal lattice: 1 center + 6 ring + 5 outer
        dx = np.sqrt(3) + 0.15
        xs, ys, as_ = [0.0], [0.0], [0.0]
        for k in range(6):
            a = k * np.pi / 3
            xs.append(dx * np.cos(a))
            ys.append(dx * np.sin(a))
            as_.append(0.0)
        # 5 more at second ring positions
        dx2 = 2 * dx + 0.15
        for k in range(5):
            a = k * 2 * np.pi / 5 + np.pi / 5
            xs.append(dx2 * np.cos(a))
            ys.append(dx2 * np.sin(a))
            as_.append(0.0)
    elif type_id == 1:
        # Staggered rows: 4+4+4
        dx = np.sqrt(3) + 0.2
        dy = 1.5 + 0.2
        xs, ys, as_ = [], [], []
        for row in range(3):
            for col in range(4):
                x = col * dx + (row % 2) * dx / 2
                y = row * dy
                xs.append(x)
                ys.append(y)
                as_.append(0.0)
    elif type_id == 2:
        # Two rows of 6
        dx = 2 * np.sqrt(3) + 0.2
        xs, ys, as_ = [], [], []
        for i in range(6):
            xs.append(i * dx)
            ys.append(0.0)
            as_.append(0.0)
        for i in range(6):
            xs.append((i + 0.5) * dx)
            ys.append(2 * np.sqrt(3) + 0.2)
            as_.append(0.0)
    else:
        # Random with hexagonal spacing
        dx = np.sqrt(3) + 0.3
        xs, ys, as_ = [], [], []
        for i in range(N):
            row = i // 4
            col = i % 4
            x = col * dx + (row % 2) * dx / 2 + np.random.randn() * 0.1
            y = row * 1.7 + np.random.randn() * 0.1
            xs.append(x)
            ys.append(y)
            as_.append(np.random.randn() * 0.3)
    
    # Center the configuration
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    xs = [x - cx for x in xs]
    ys = [y - cy for y in ys]
    
    params = np.zeros(3 * N + 3)
    for i in range(N):
        params[3*i] = xs[i]
        params[3*i + 1] = ys[i]
        params[3*i + 2] = as_[i]
    params[3*N] = 0.0
    params[3*N + 1] = 0.0
    params[3*N + 2] = 0.0
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    best_params = None
    best_score = float('inf')
    
    # Phase 1: Differential evolution for global search
    bounds = [(-15, 15)] * N + [(-15, 15)] * N + [(0, 2*np.pi)] * N + [(-5, 5), (-5, 5), (0, 2*np.pi)]
    
    def de_objective(x):
        if time.time() > deadline - 30:
            return 1e6
        return objective(x)
    
    try:
        result_de = differential_evolution(
            de_objective, bounds,
            maxiter=50, popsize=15, seed=42,
            tol=1e-4, atol=1e-6,
            init='latinhypercube'
        )
        if result_de.fun < best_score:
            best_score = result_de.fun
            best_params = result_de.x
    except Exception:
        pass
    
    # Phase 2: Multiple Nelder-Mead refinements from diverse starts
    if time.time() < deadline - 20:
        # Refine from DE result
        if best_params is not None:
            try:
                res = minimize(objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x
            except Exception:
                pass
    
    # Try multiple diverse initial configurations
    for cfg_type in range(4):
        if time.time() > deadline - 15:
            break
        try:
            init_params = make_initial_config(cfg_type)
            res = minimize(objective, init_params, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            continue
    
    # Phase 3: Local refinement with small perturbations
    if best_params is not None and time.time() < deadline - 10:
        np.random.seed(123)
        for _ in range(5):
            if time.time() > deadline - 8:
                break
            try:
                perturbed = best_params + np.random.randn(len(best_params)) * 0.05
                res = minimize(objective, perturbed, method='Nelder-Mead',
                              options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-9})
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x
            except Exception:
                continue
    
    # Extract final solution
    if best_params is None:
        # Fallback: simple valid configuration
        dx = 2 * np.sqrt(3) + 0.5
        inner_result = []
        for i in range(N):
            row = i // 4
            col = i % 4
            x = col * dx - 1.5 * dx
            y = row * 2.0 - 1.0
            inner_result.append([x, y, 0.0])
        outer_center = (0.0, 0.0)
        outer_rot = 0.0
        # Compute required side
        hexes = [hex_vertices(p[0], p[1], 1.0, 0.0) for p in inner_result]
        required_side = required_outer_side(hexes, outer_center, 0.0)
        return inner_result, outer_center, required_side, outer_rot
    
    inner_result = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i + 1]
        ang = best_params[3*i + 2]
        inner_result.append([x, y, ang])
    
    outer_cx = best_params[3*N]
    outer_cy = best_params[3*N + 1]
    outer_rot = best_params[3*N + 2]
    
    outer_center = (outer_cx, outer_cy)
    hexes = [hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner_result]
    required_side = required_outer_side(hexes, outer_center, outer_rot)
    
    # Verify disjointness
    if not all_hexagons_disjoint(hexes, tol=1e-6):
        # Fallback
        dx = 2 * np.sqrt(3) + 0.5
        inner_result = []
        for i in range(N):
            row = i // 4
            col = i % 4
            x = col * dx - 1.5 * dx
            y = row * 2.0 - 1.0
            inner_result.append([x, y, 0.0])
        outer_center = (0.0, 0.0)
        outer_rot = 0.0
        hexes = [hex_vertices(p[0], p[1], 1.0, 0.0) for p in inner_result]
        required_side = required_outer_side(hexes, outer_center, 0.0)
    
    return inner_result, outer_center, required_side, outer_rot
