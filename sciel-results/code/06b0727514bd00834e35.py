import numpy as np
from scipy.optimize import minimize_scalar

N = 11

def construct():
    # Define the 11 centers
    axial_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, 0), (-2, 0),
        (0, 2), (0, -2)
    ]
    
    centers = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    centers = np.array(centers)
    
    # Precompute vertices of unit hexagon (side 1, circumradius 1)
    hex_angles = np.arange(6) * np.pi / 3.0
    hex_verts = np.stack([np.cos(hex_angles), np.sin(hex_angles)], axis=1)
    
    # Function to compute h_Inner(n)
    # h_Inner(n) = max_i (c_i . n + h_hex(n))
    # Since all inner hexagons have same orientation, h_hex(n) is common.
    # h_hex(n) = max_j (v_j . n)
    
    def h_hex(n):
        return np.max(np.dot(hex_verts, n))
        
    def h_inner(n):
        # n is a unit vector
        proj_centers = np.dot(centers, n)
        return np.max(proj_centers) + h_hex(n)
        
    def h_outer_unit(n):
        # Support function of unit side hexagon (circumradius 1)
        # Vertices at 0, 60...
        # This is the same as h_hex(n) if aligned?
        # Yes, a regular hexagon of side 1 has circumradius 1.
        # So h_outer_unit(n) is exactly h_hex(n) if the outer hexagon is aligned with the inner ones (alpha=0).
        # But the outer hexagon is rotated by alpha.
        # So h_outer(n) = h_hex(n - alpha) ?
        # No, h_outer is the support function of the OUTER hexagon.
        # The outer hexagon has side L.
        # h_outer_L(n) = L * h_unit_hex(n - alpha)
        # where h_unit_hex is the support function of a unit-side hexagon (circumradius 1).
        # Wait, side 1 hexagon has circumradius 1.
        # So h_unit_hex(t) = max_j v_j . (cos(t), sin(t))
        
        # We need L such that h_inner(n) <= L * h_unit_hex(n - alpha) for all n.
        # L >= h_inner(n) / h_unit_hex(n - alpha)
        
        # Note: h_unit_hex(t) is never 0.
        
        return h_hex(n - alpha)
        
    def calc_L(alpha):
        # Sample n in [0, pi/3]
        # The ratio is periodic with pi/3?
        # h_inner is periodic with pi/3 (due to hexagon symmetry).
        # h_outer_unit is periodic with pi/3.
        # So we just need to check n in [0, pi/3].
        
        # Use a fine grid
        n_angles = np.linspace(0, np.pi/3, 360)
        max_ratio = 0.0
        for theta in n_angles:
            n = np.array([np.cos(theta), np.sin(theta)])
            hi = h_inner(n)
            ho = h_hex(theta - alpha) # h_outer_unit(n)
            if ho < 1e-9: continue
            ratio = hi / ho
            if ratio > max_ratio:
                max_ratio = ratio
        return max_ratio

    res = minimize_scalar(calc_L, bounds=(0, np.pi/3), method='bounded')
    best_alpha = res.x
    best_L = res.fun
    
    inner = []
    for x, y in centers:
        inner.append([float(x), float(y), 0.0])
        
    return inner, (0.0, 0.0), float(best_L), float(np.degrees(best_alpha))
