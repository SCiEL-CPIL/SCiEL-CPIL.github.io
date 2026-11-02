import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 12
SQRT3 = np.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    # Define topologies
    def get_topology_3x4():
        centers = []
        for r in range(3):
            for c in range(4):
                x = c * SQRT3
                y = r * 1.5
                if r % 2 == 1:
                    x += SQRT3 / 2
                centers.append((x, y))
        # Center the cluster
        cx = np.mean([c[0] for c in centers])
        cy = np.mean([c[1] for c in centers])
        return [(x - cx, y - cy) for x, y in centers]

    def get_topology_2x6():
        centers = []
        for r in range(2):
            for c in range(6):
                x = c * SQRT3
                y = r * 1.5
                if r % 2 == 1:
                    x += SQRT3 / 2
                centers.append((x, y))
        cx = np.mean([c[0] for c in centers])
        cy = np.mean([c[1] for c in centers])
        return [(x - cx, y - cy) for x, y in centers]

    def get_topology_ring():
        centers = [(0,0)]
        for i in range(6):
            angle = 2 * np.pi * i / 6
            centers.append((1.75 * np.cos(angle), 1.75 * np.sin(angle)))
        # Add 5 more on the next ring
        for i in range(5):
            angle = 2 * np.pi * (i + 0.5) / 6
            centers.append((3.1 * np.cos(angle), 3.1 * np.sin(angle)))
        cx = np.mean([c[0] for c in centers])
        cy = np.mean([c[1] for c in centers])
        return [(x - cx, y - cy) for x, y in centers]

    topologies = [get_topology_3x4(), get_topology_2x6(), get_topology_ring()]

    def hexagon_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def objective(params, topology):
        theta_inner = params[0]
        x_off = params[1]
        y_off = params[2]
        theta_outer = params[3]
        
        # Rotate and translate topology centers
        cos_t = np.cos(theta_inner)
        sin_t = np.sin(theta_inner)
        rotated_centers = []
        for x, y in topology:
            rx = x * cos_t - y * sin_t + x_off
            ry = x * sin_t + y * cos_t + y_off
            rotated_centers.append((rx, ry))
            
        max_required = 0.0
        # Outer hexagon is centered at (0,0)
        for cx, cy in rotated_centers:
            # Inner hexagon orientation is also theta_inner
            verts = hexagon_vertices(cx, cy, 1.0, theta_inner)
            for v in verts:
                dx = v[0]
                dy = v[1]
                for k in range(6):
                    edge_angle = theta_outer + np.radians(k * 60)
                    nx = np.cos(edge_angle)
                    ny = np.sin(edge_angle)
                    dist = dx * nx + dy * ny
                    if dist > 0:
                        required_S = dist * 2.0 / SQRT3
                        if required_S > max_required:
                            max_required = required_S
        return max_required

    best_result = None
    best_score = np.inf

    for topo in topologies:
        if time.time() > deadline - 20:
            break
            
        # Global search
        bounds = [(-np.pi, np.pi), (-5, 5), (-5, 5), (-np.pi/6, np.pi/6)]
        try:
            res_global = differential_evolution(
                objective, bounds, args=(topo,), 
                seed=42, maxiter=100, tol=1e-6, polish=False
            )
            best_params = res_global.x
            best_obj = res_global.fun
            
            # Local polish
            res_local = minimize(
                objective, best_params, args=(topo,),
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000}
            )
            if res_local.fun < best_obj:
                best_obj = res_local.fun
                best_params = res_local.x
                
            if best_obj < best_score:
                best_score = best_obj
                best_result = (best_params, topo)
                
        except Exception as e:
            pass

    # Extract final parameters
    if best_result is None:
        # Fallback
        return [[i*2, 0, 0] for i in range(12)], (0,0), 10.0, 0.0

    params, topo = best_result
    theta_inner = params[0]
    x_off = params[1]
    y_off = params[2]
    theta_outer = params[3]

    # Construct final inner hexagons
    inner = []
    cos_t = np.cos(theta_inner)
    sin_t = np.sin(theta_inner)
    for x, y in topo:
        rx = x * cos_t - y * sin_t + x_off
        ry = x * sin_t + y * cos_t + y_off
        inner.append([rx, ry, np.degrees(theta_inner)])
        
    # Calculate final side length
    # Re-use objective logic for exact value
    max_required = 0.0
    for cx, cy, _ in inner:
        verts = hexagon_vertices(cx, cy, 1.0, theta_inner)
        for v in verts:
            dx = v[0]
            dy = v[1]
            for k in range(6):
                edge_angle = theta_outer + np.radians(k * 60)
                nx = np.cos(edge_angle)
                ny = np.sin(edge_angle)
                dist = dx * nx + dy * ny
                if dist > 0:
                    required_S = dist * 2.0 / SQRT3
                    if required_S > max_required:
                        max_required = required_S
                        
    return inner, (0.0, 0.0), max_required, np.degrees(theta_outer)
