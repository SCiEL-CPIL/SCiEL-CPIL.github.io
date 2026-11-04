"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Uses a honeycomb lattice arrangement and optimizes the container hexagon.
    """
    sqrt3 = np.sqrt(3)
    
    # Generate positions in a hexagonal (axial) coordinate system
    positions = []
    positions.append((0, 0))  # Center
    
    # Add rings around the center
    ring = 1
    while len(positions) < N:
        for i in range(6 * ring):
            if len(positions) >= N:
                break
            if i < ring:
                q = ring
                r = -ring + i
            elif i < 2 * ring:
                q = ring - (i - ring)
                r = i - ring
            elif i < 3 * ring:
                q = -(i - 2 * ring)
                r = ring
            elif i < 4 * ring:
                q = -ring
                r = ring - (i - 3 * ring)
            elif i < 5 * ring:
                q = -ring + (i - 4 * ring)
                r = -(i - 4 * ring)
            else:
                q = i - 5 * ring
                r = -ring
        
        positions.append((q, r))
        ring += 1
    
    positions = positions[:N]
    
    # Convert axial coordinates to Cartesian (flat-top hex grid, unit hexagons)
    cart = []
    for q, r in positions:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        cart.append((x, y))
    
    pts = np.array(cart)
    
    # Find the minimal regular hexagon container
    best_L = float('inf')
    best_params = None
    
    # Try different rotation angles (0 to 60 degrees, step 5)
    for theta_deg in range(0, 60, 5):
        theta = np.radians(theta_deg)
        normals = []
        for k in range(6):
            angle = theta + k * np.pi / 3
            normals.append((np.cos(angle), np.sin(angle)))
        
        def objective(params):
            cx, cy = params
            max_proj = 0.0
            for nx, ny in normals:
                projections = pts[:, 0] * nx + pts[:, 1] * ny - (cx * nx + cy * ny)
                max_abs = np.max(np.abs(projections))
                max_proj = max(max_proj, max_abs)
            return (max_proj + sqrt3 / 2) * 2.0 / sqrt3
        
        cx0, cy0 = np.mean(pts, axis=0)
        res = minimize(objective, [cx0, cy0], method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        L_val = res.fun
        if L_val < best_L:
            best_L = L_val
            best_params = (res.x[0], res.x[1], theta_deg)
    
    cx, cy, theta_deg = best_params
    theta = np.radians(theta_deg)
    
    # All inner hexagons have the same orientation as the container
    inner = [[float(x), float(y), float(theta_deg)] for x, y in cart]
    
    return inner, (float(cx), float(cy)), float(best_L), float(theta_deg)
# EVOLVE-BLOCK-END

