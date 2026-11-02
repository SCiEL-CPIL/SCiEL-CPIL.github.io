"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N unit regular hexagons in a hexagonal lattice arrangement.
    Compute the minimum outer regular hexagon side length analytically.
    """
    
    def axial_to_cartesian(q, r):
        """Convert axial coordinates to Cartesian for unit hex grid."""
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    def generate_hex_positions(n):
        """Generate n positions in a hexagonal cluster pattern."""
        positions = [(0, 0)]
        ring = 1
        while len(positions) < n:
            for q in range(-ring, ring + 1):
                for r in range(-ring, ring + 1):
                    if max(abs(q), abs(r), abs(q + r)) == ring:
                        if len(positions) < n:
                            positions.append((q, r))
            ring += 1
        return positions[:n]
    
    def compute_min_hex_side(caxial):
        """
        Given axial coordinates of unit hexagon centers, compute the minimum
        side length of a regular hexagon that contains all unit hexagons.
        Assumes all unit hexagons are axis-aligned.
        """
        # Convert to Cartesian
        cart = np.array([axial_to_cartesian(q, r) for q, r in caxial])
        
        # For axis-aligned unit hexagons, the support function in direction theta
        # is: for each hexagon center c, the max projection is c·u + inradius_hex
        # where inradius_hex = sqrt(3)/2 for the unit hexagon's inradius in that direction.
        # But actually, the unit hexagon's support function depends on direction.
        # For a regular hexagon with circumradius 1, the support function h(u) = 
        # max over vertices of v·u. For axis-aligned (flat top), vertices are at
        # angles 0, 60, 120, 180, 240, 300 degrees.
        
        # The outer hexagon with side s, center (cx,cy), rotation phi has inradius s*sqrt(3)/2.
        # It contains a unit hexagon at center c (axis-aligned) iff for all 6 directions
        # of the outer hexagon, the projection of (c - center) onto that direction
        # plus the unit hexagon's support in that direction <= s*sqrt(3)/2.
        
        # Unit hexagon (axis-aligned, circumradius 1) support function:
        # h_unit(theta) = max(cos(theta - k*pi/3) for k=0..5)
        
        # Let's do a numerical optimization over the outer hexagon parameters.
        
        centroid = np.mean(cart, axis=0)
        
        # Estimate initial side
        max_dist = np.max(np.linalg.norm(cart - centroid, axis=1))
        init_side = 2 * max_dist / np.sqrt(3) + 2
        
        def support_unit(theta):
            """Support function of axis-aligned unit hexagon in direction theta."""
            vals = [np.cos(theta - k * np.pi / 3.0) for k in range(6)]
            return max(vals)
        
        def check_contains(cx, cy, side, phi, cart_pts):
            inradius = side * np.sqrt(3) / 2.0
            for k in range(6):
                angle = phi + k * np.pi / 3.0
                nx, ny = np.cos(angle), np.sin(angle)
                # Projection of each center onto normal
                projections = (cart_pts[:, 0] - cx) * nx + (cart_pts[:, 1] - cy) * ny
                # Add support of unit hexagon in this direction
                h_vals = np.array([support_unit(angle) for _ in range(len(cart_pts))])
                # Actually support depends on direction, which is the same for all in this loop
                h_val = support_unit(angle)
                max_proj = np.max(projections) + h_val
                if max_proj > inradius + 1e-9:
                    return False, max_proj - inradius
            return True, 0.0
        
        def objective(params):
            cx, cy, side, phi = params
            if side <= 0:
                return 1e10
            ok, viol = check_contains(cx, cy, side, phi, cart)
            if ok:
                return side
            else:
                return side + 100.0 * viol
        
        from scipy.optimize import minimize
        
        best_side = float('inf')
        for phi_init in range(0, 360, 30):
            phi0 = np.radians(phi_init)
            x0 = [centroid[0], centroid[1], init_side, phi0]
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
                if res.fun < best_side and res.fun < init_side + 1:
                    best_side = res.fun
            except Exception:
                pass
        
        return best_side
    
    # Generate hexagonal cluster
    caxial = generate_hex_positions(N)
    side_hex = compute_min_hex_side(caxial)
    
    # Also try linear arrangement
    caxial_linear = [(i, 0) for i in range(N)]
    side_linear = compute_min_hex_side(caxial_linear)
    
    # Try a few more patterns
    # 2 rows
    caxial_2row = []
    for i in range(N):
        row = i // 6
        col = i % 6
        caxial_2row.append((col - row // 2, row))
    side_2row = compute_min_hex_side(caxial_2row)
    
    # Pick best
    candidates = [
        (side_hex, caxial),
        (side_linear, caxial_linear),
        (side_2row, caxial_2row),
    ]
    
    best_side, best_caxial = min(candidates, key=lambda x: x[0])
    
    # Convert to Cartesian for output
    inner = []
    for q, r in best_caxial:
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, 0.0])
    
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END
