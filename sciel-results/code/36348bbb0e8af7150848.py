import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + d * np.cos(angles),
        cy + d * np.sin(angles)
    ])

def _check_separation(hex1, hex2, tol=1e-6):
    """Check if two hexagons are separated using SAT."""
    v1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
    
    for verts in [v1, v2]:
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i+1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-15)
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            # If separated, there should be a gap
            if max1 + tol < min2 or max2 + tol < min1:
                return True
    
    return False

def _vertex_inside_outer(v, outer_verts):
    """Check if vertex v is inside outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i+1) % 6]
        cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
        if cross < -1e-6:
            return False
    return True

def _validate(inner_data, outer_data, s):
    """Validate the packing."""
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    # Check inner hexagons
    inner_verts_list = []
    for i in range(12):
        verts = _hex_vertices(inner_data[i,0], inner_data[i,1], 1.0, inner_data[i,2])
        inner_verts_list.append(verts)
        
        # Check unit edges
        for j in range(6):
            p1 = verts[j]
            p2 = verts[(j+1) % 6]
            if abs(np.linalg.norm(p2 - p1) - 1.0) > 1e-8:
                return False
        
        # Check inside outer
        for v in verts:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    
    # Check separation
    for i in range(12):
        for j in range(i+1, 12):
            if not _check_separation(inner_data[i], inner_data[j]):
                return False
    
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline
    
    # Start with a reasonable initial guess
    # Honeycomb arrangement
    sqrt3 = np.sqrt(3)
    
    # Initial placement in a hexagonal pattern
    # Center at origin, 12 hexagons in a compact cluster
    # Let's try: 3 rows of 4, or a hexagonal cluster
    
    # A good initial guess: hexagonal cluster
    # Positions for 12 hexagons in a honeycomb:
    # Row structure (offset rows)
    
    # Let me use a symmetric arrangement
    # 1 center + 6 around + 5 more = 12
    
    # Actually, let me try a rectangular-ish honeycomb
    # 4 in a row, 3 rows, with offsets
    
    # Spacing: horizontal = 1.5 (for side-by-side), vertical = sqrt(3)
    # Wait, for hexagons sharing edges:
    # If hexagons have flat tops (angle 0), adjacent horizontally are 3/2 apart
    # Adjacent in the offset direction are (3/4, sqrt(3)/2) away
    
    # Let me use pointy-top hexagons (angle 30) for easier row packing
    # Actually, let me just use angle 0 and figure out positions
    
    # For angle 0 (flat top), a unit hexagon has:
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # Width (x) = 2, Height (y) = sqrt(3)
    
    # Two hexagons sharing a vertical edge: centers are 3/2 apart in x
    # Two hexagons sharing a top/bottom edge: centers are sqrt(3)/2 apart in y, 3/4 in x
    
    # Honeycomb with flat-top hexagons:
    # Column spacing: 3/2
    # Row spacing: sqrt(3)
    # Odd columns offset by sqrt(3)/2 in y
    
    # Let me place 12 hexagons:
    # 4 columns, 3 rows
    # Col 0: rows 0,1,2 at y=0, sqrt(3), 2*sqrt(3)
    # Col 1: rows 0,1,2 at y=sqrt(3)/2, 3*sqrt(3)/2, 5*sqrt(3)/2
    # Col 2: rows 0,1,2 at y=0, sqrt(3), 2*sqrt(3)
    # Col 3: rows 0,1,2 at y=sqrt(3)/2, 3*sqrt(3)/2, 5*sqrt(3)/2
    
    # That's 12 hexagons
    
    col_spacing = 1.5
    row_spacing = sqrt3
    
    positions = []
    for col in range(4):
        for row in range(3):
            x = col * col_spacing
            if col % 2 == 0:
                y = row * row_spacing
            else:
                y = (row + 0.5) * row_spacing
            positions.append([x, y, 0.0])
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    inner_data = np.array(positions)
    
    # Estimate outer hexagon size
    # The bounding hexagon needs to contain all these hexagons
    # Each inner hexagon extends 1 unit from its center in all directions (circumradius = 1)
    # So we need a hexagon that contains all points at distance 1 from the inner centers
    
    # For a flat-top outer hexagon, the apothem must be at least the max distance
    # from center to any point on the inner hexagons
    
    # Let me just start with a large s and optimize
    s = 6.0  # initial guess
    outer_data = np.array([0.0, 0.0, 0.0])
    
    # Now optimize to minimize s
    # Variables: positions of 12 hexagons (x, y for each) + s
    # We'll fix angles to 0 for simplicity (or allow small rotations)
    
    # Let me define the optimization
    def objective(params):
        # params: [x1,y1, x2,y2, ..., x12,y12, s]
        inner_xy = params[:24].reshape(12, 2)
        s_val = params[24]
        
        if s_val < np.sqrt(12):
            return 1e6
        
        # Build inner data
        inner = np.zeros((12, 3))
        inner[:, 0] = inner_xy[:, 0]
        inner[:, 1] = inner_xy[:, 1]
        inner[:, 2] = 0.0  # all angle 0
        
        outer = np.array([0.0, 0.0, 0.0])
        
        # Check validity and compute penalty
        outer_verts = _hex_vertices(0, 0, s_val, 0)
        
        penalty = 0.0
        
        # Check each inner hexagon
        for i in range(12):
            verts = _hex_vertices(inner[i,0], inner[i,1], 1.0, 0.0)
            
            # Check inside outer
            for v in verts:
                for j in range(6):
                    p1 = outer_verts[j]
                    p2 = outer_verts[(j+1) % 6]
                    cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                    if cross < 0:
                        penalty += abs(cross)
            
            # Check separation from other hexagons
            for j in range(i+1, 12):
                verts2 = _hex_vertices(inner[j,0], inner[j,1], 1.0, 0.0)
                
                # SAT check
                separated = False
                for verts in [verts, verts2]:
                    for k in range(6):
                        p1 = verts[k]
                        p2 = verts[(k+1) % 6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-15:
                            normal = normal / norm
                            proj1 = verts @ normal
                            proj2 = verts2 @ normal
                            min1, max1 = proj1.min(), proj1.max()
                            min2, max2 = proj2.min(), proj2.max()
                            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                                separated = True
                                break
                    if separated:
                        break
                
                if not separated:
                    # Compute overlap penalty
                    # Use minimum distance between projections
                    min_gap = np.inf
                    for verts in [verts, verts2]:
                        for k in range(6):
                            p1 = verts[k]
                            p2 = verts[(k+1) % 6]
                            edge = p2 - p1
                            normal = np.array([-edge[1], edge[0]])
                            norm = np.linalg.norm(normal)
                            if norm > 1e-15:
                                normal = normal / norm
                                proj1 = verts @ normal
                                proj2 = verts2 @ normal
                                min1, max1 = proj1.min(), proj1.max()
                                min2, max2 = proj2.min(), proj2.max()
                                gap = min(min2 - max1, min1 - max2)
                                if gap < 0:
                                    min_gap = min(min_gap, -gap)
                    
                    if min_gap < np.inf:
                        penalty += min_gap ** 2
        
        return s_val + 1000 * penalty
    
    # Initial params
    init_params = np.zeros(25)
    init_params[:24] = inner_data[:, :2].flatten()
    init_params[24] = s
    
    # Optimize
    best_result = None
    best_value = 1.0 / s
    
    # Try multiple random restarts
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            params0 = init_params.copy()
        else:
            # Perturb initial guess
            perturbation = 0.5 * np.random.randn(24)
            params0 = init_params.copy()
            params0[:24] += perturbation
            params0[24] = s + 0.5 * np.random.randn()
        
        try:
            result = minimize(
                objective,
                params0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < 100:
                final_params = result.x
                inner_xy_final = final_params[:24].reshape(12, 2)
                s_final = final_params[24]
                
                if s_final > np.sqrt(12):
                    inner_final = np.zeros((12, 3))
                    inner_final[:, 0] = inner_xy_final[:, 0]
                    inner_final[:, 1] = inner_xy_final[:, 1]
                    inner_final[:, 2] = 0.0
                    outer_final = np.array([0.0, 0.0, 0.0])
                    
                    if _validate(inner_final, outer_final, s_final):
                        value = 1.0 / s_final
                        if value > best_value:
                            best_value = value
                            best_result = (inner_final, outer_final, s_final)
        except Exception as e:
            pass
    
    if best_result is not None:
        return best_result
    else:
        # Return the initial valid solution
        return inner_data, outer_data, s


# EVOLVE-BLOCK-END
