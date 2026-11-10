import numpy as np
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600.0  # seconds
    
    # Hexagon vertices relative to center (unit side, angle 0)
    verts_rel = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Edge normals for hexagons (6 unique directions)
    # Normals are perpendicular to edges.
    # Edge 0: (1,0) to (0.5, sqrt(3)/2). Midpoint (0.75, sqrt(3)/4).
    # Normal direction is radial from center to midpoint?
    # For a regular hexagon, the normal to an edge passes through the center and the midpoint of the edge.
    # Midpoint of edge 0 is at angle 30 deg.
    # So normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = np.array([
        [np.cos(np.radians(30)), np.sin(np.radians(30))],
        [np.cos(np.radians(90)), np.sin(np.radians(90))],
        [np.cos(np.radians(150)), np.sin(np.radians(150))],
        [np.cos(np.radians(210)), np.sin(np.radians(210))],
        [np.cos(np.radians(270)), np.sin(np.radians(270))],
        [np.cos(np.radians(330)), np.sin(np.radians(330))]
    ])
    
    def get_positions(x):
        return x.reshape(11, 2)
    
    def check_separation(coords, eps=1e-6):
        """Check if all pairs of hexagons are separated by > eps."""
        for i in range(11):
            for j in range(i+1, 11):
                d = coords[i] - coords[j]
                separated = False
                for n in normals:
                    proj = np.dot(d, n)
                    # Separation condition: |proj| > sqrt(3) + eps
                    if abs(proj) > np.sqrt(3) + eps:
                        separated = True
                        break
                if not separated:
                    return False
        return True

    def compute_min_side(coords):
        """Compute the minimum side length of outer hexagon containing all inner hexagons."""
        # Assume outer hexagon is centered at origin and aligned with inner ones.
        # The max projection of any vertex onto any normal determines the side length.
        # S * sqrt(3)/2 = max_proj
        max_proj = 0
        for c in coords:
            verts = c + verts_rel
            proj = verts @ normals.T
            max_proj = max(max_proj, np.max(np.abs(proj))) # Use abs because normals cover all directions? 
            # Actually, normals are 6 directions. proj can be negative.
            # The "width" of the hexagon in direction n is max(v.n) - min(v.n).
            # For a centered hexagon, max(v.n) = S*sqrt(3)/2.
            # So we need max(v.n) over all v and all n.
            # Since the set of normals covers 360 degrees (every 60 deg), 
            # max_proj = max( max(v.n), max(-v.n) ) ?
            # No, the outer hexagon is fixed orientation.
            # The condition is that all vertices must be inside.
            # A point p is inside the outer hexagon (centered at 0, side S) if
            # |p . n_k| <= S * sqrt(3)/2 for all k=0..5.
            # So we need S * sqrt(3)/2 >= max_{p, k} |p . n_k|.
            # So S = 2/sqrt(3) * max_{p, k} |p . n_k|.
            pass
        
        # Corrected computation
        max_abs_proj = 0
        for c in coords:
            verts = c + verts_rel
            proj = verts @ normals.T
            max_abs_proj = max(max_abs_proj, np.max(np.abs(proj)))
        
        S = max_abs_proj * 2 / np.sqrt(3)
        return S

    def is_valid(coords):
        if not check_separation(coords):
            return False
        S = compute_min_side(coords)
        if S < np.sqrt(11) - 1e-6:
            return False
        return True

    # Initial guess: Scaled honeycomb lattice
    def make_initial(scale=1.01):
        x0 = np.zeros(22)
        # Center
        x0[0:2] = [0, 0]
        # Ring 1: 6 hexagons
        r1 = np.sqrt(3) * scale
        angles1 = [0, 60, 120, 180, 240, 300]
        for k, ang in enumerate(angles1):
            idx = (k+1)*2
            x0[idx] = r1 * np.cos(np.radians(ang))
            x0[idx+1] = r1 * np.sin(np.radians(ang))
        
        # Ring 2: 4 hexagons
        # Place them in the "corner" directions: 0, 60, 120, 180? 
        # Or spread them out?
        # Let's try 0, 60, 120, 180 for now.
        # Distance from center needs to be > r1 + something.
        # In honeycomb, next ring is at distance 2*sqrt(3).
        r2 = 2.0 * np.sqrt(3) * scale
        angles2 = [0, 60, 120, 180]
        for k, ang in enumerate(angles2):
            idx = (k+7)*2
            x0[idx] = r2 * np.cos(np.radians(ang))
            x0[idx+1] = r2 * np.sin(np.radians(ang))
            
        return x0

    x0 = make_initial(1.02) # Scale up to ensure separation
    best_coords = get_positions(x0)
    best_S = compute_min_side(best_coords)
    
    # Optimization
    from scipy.optimize import minimize
    
    def objective(x):
        coords = get_positions(x)
        S = compute_min_side(coords)
        penalty = 0
        for i in range(11):
            for j in range(i+1, 11):
                d = coords[i] - coords[j]
                for n in normals:
                    proj = np.dot(d, n)
                    sep = abs(proj) - (np.sqrt(3) + 1e-6)
                    if sep < 0:
                        penalty += (-sep) ** 2 * 1000
        return S + penalty

    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            x_curr = x0.copy()
        else:
            x_curr = best_coords.flatten() + np.random.randn(22) * 0.5

        res = minimize(objective, x_curr, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        coords = get_positions(res.x)
        if check_separation(coords):
            S = compute_min_side(coords)
            if S < best_S:
                best_S = S
                best_coords = coords.copy()
    
    # Final check
    if not check_separation(best_coords):
        # Fallback to a definitely valid, though large, configuration
        # Scale up the initial guess significantly
        x_safe = make_initial(2.0)
        best_coords = get_positions(x_safe)
        best_S = compute_min_side(best_coords)

    # Construct return values
    inner = np.zeros((11, 3))
    inner[:, :2] = best_coords
    inner[:, 2] = 0
    outer = np.array([0, 0, 0])
    return inner, outer, float(best_S)
