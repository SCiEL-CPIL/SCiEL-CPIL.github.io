# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns (inner, outer, side).
    """
    deadline = time.time() + 600  # safety margin
    start_time = time.time()

    n = 11

    # Hexagon vertices: for center (cx,cy), side s=1, angle a (deg):
    # v_i = (cx + cos(a + 60i*pi/180), cy + sin(a + 60i*pi/180))
    def hex_vertices(cx, cy, a_deg):
        a = np.radians(a_deg)
        angles = a + 2 * np.pi * np.arange(6) / 6.0
        return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

    def hex_edge_normals(vertices):
        """Return 12 edge normals (6 edges, 2 directions each) and edge midpoint projections."""
        # 6 edges
        normals = []
        for i in range(6):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 6]
            edge = p2 - p1
            # Normal pointing outward
            n = np.array([-edge[1], edge[0]])
            n = n / np.linalg.norm(n)
            normals.append(n)
        normals = np.array(normals)
        return normals

    def separation_check(v1, v2):
        """Check if two hexagons are separated (no overlap).
        Returns True if separated, False if overlapping."""
        n1 = hex_edge_normals(v1)
        n2 = hex_edge_normals(v2)
        all_normals = np.vstack([n1, n2])
        for n in all_normals:
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            # If intervals are apart by more than 1e-6
            if max1 + 1e-6 < min2 or max2 + 1e-6 < min1:
                return True
        return False

    def overlap_penalty(v1, v2):
        """Return a penalty if hexagons overlap (0 if separated)."""
        n1 = hex_edge_normals(v1)
        n2 = hex_edge_normals(v2)
        all_normals = np.vstack([n1, n2])
        min_gap = np.inf
        for n in all_normals:
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            gap = max(0, max1 - min2, max2 - min1)
            min_gap = min(min_gap, gap)
        # If min_gap <= 0, they overlap; return overlap amount
        if min_gap > 1e-6:
            return 0.0
        return max(0.0, -min_gap)

    def outer_hex_vertices(cx, cy, side, a_deg):
        a = np.radians(a_deg)
        angles = a + 2 * np.pi * np.arange(6) / 6.0
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

    def containment_penalty(inner_verts, outer_verts):
        """Check if all inner vertices are inside outer hexagon.
        Returns penalty = max violation (0 if contained)."""
        # Outer hexagon edges: for each edge, check cross product
        penalty = 0.0
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            edge = p2 - p1
            # Normal pointing inward
            n = np.array([-edge[1], edge[0]])
            n = n / np.linalg.norm(n)
            # For containment, inner vertices must be on the same side as the center
            # cross(p2-p1, p-p1) >= -1e-6
            for p in inner_verts:
                cross = np.cross(edge, p - p1)
                if cross < -1e-6:
                    penalty = max(penalty, -cross)
        return penalty

    def compute_objective(params):
        """Compute objective: minimize side + penalties.
        params: [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10, side]
        """
        if time.time() > deadline:
            return None
        
        positions = params[:3 * n].reshape(n, 3)
        side = params[3 * n]
        
        if side < np.sqrt(11):
            side = np.sqrt(11) + 1e-9
        
        # Compute vertices for each inner hexagon
        vertices_list = []
        for i in range(n):
            cx, cy, a_deg = positions[i]
            v = hex_vertices(cx, cy, a_deg)
            vertices_list.append(v)
        
        # Outer hexagon centered at origin
        outer_v = outer_hex_vertices(0, 0, side, 0)
        
        # Containment penalty
        total_inner_verts = np.vstack(vertices_list)
        contain_pen = containment_penalty(total_inner_verts, outer_v)
        
        # Overlap penalties
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                op = overlap_penalty(vertices_list[i], vertices_list[j])
                overlap_pen += op
        
        # Objective: side + large penalty for violations
        obj = side + 1000.0 * contain_pen + 1000.0 * overlap_pen
        return obj

    def compute_objective_grad(params):
        """Numerical gradient."""
        return None  # Let scipy handle it

    # Initial guess: hexagonal grid arrangement
    # Center at origin, 6 around it, 4 more
    init_positions = []
    # Center
    init_positions.append([0, 0, 0])
    # 6 surrounding
    for k in range(6):
        angle = k * 60
        r = 2.0  # distance between centers for touching hexagons
        x = r * np.cos(np.radians(angle))
        y = r * np.sin(np.radians(angle))
        init_positions.append([x, y, 0])
    # 4 more: place them at distance ~3.5
    for k in [30, 90, 150, 210]:
        r = 3.5
        x = r * np.cos(np.radians(k))
        y = r * np.sin(np.radians(k))
        init_positions.append([x, y, 0])
    
    init_positions = np.array(init_positions)
    init_side = 5.0
    
    init_params = np.concatenate([init_positions.flatten(), [init_side]])
    
    best_params = init_params.copy()
    best_obj = compute_objective(init_params)
    
    # Try multiple random restarts
    n_restarts = 5
    np.random.seed(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            params0 = init_params
        else:
            # Random perturbation
            params0 = init_params + np.random.randn(*init_params.shape) * 0.5
            params0[-1] = max(np.sqrt(11), init_side + np.random.randn() * 0.5)
        
        # Use Nelder-Mead for this non-smooth problem
        result = minimize(
            compute_objective,
            params0,
            method='Nelder-Mead',
            options={
                'xatol': 1e-6,
                'fatol': 1e-8,
                'maxiter': 2000,
                'maxfev': 5000
            }
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x.copy()
        
        # Also try L-BFGS-B as a local refinement
        if time.time() > deadline - 120:
            break
        
        result2 = minimize(
            compute_objective,
            result.x,
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'ftol': 1e-10,
                'gtol': 1e-8
            }
        )
        
        if result2.fun < best_obj:
            best_obj = result2.fun
            best_params = result2.x.copy()
    
    # Final refinement with tighter tolerance
    if time.time() < deadline - 30:
        result_final = minimize(
            compute_objective,
            best_params,
            method='Nelder-Mead',
            options={
                'xatol': 1e-8,
                'fatol': 1e-10,
                'maxiter': 3000,
                'maxfev': 10000
            }
        )
        if result_final.fun < best_obj:
            best_obj = result_final.fun
            best_params = result_final.x.copy()
    
    # Extract best solution
    positions = best_params[:3 * n].reshape(n, 3)
    side = best_params[3 * n]
    
    # Ensure side is at least sqrt(11)
    side = max(side, np.sqrt(11))
    
    inner_hex_data = positions
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, float(side)


# EVOLVE-BLOCK-END
