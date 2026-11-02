"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place N unit hexagons inside a regular hexagon, minimizing the outer side length.
    
    Strategy: Place centers on a triangular lattice in a hexagonal arrangement,
    then compute the minimal enclosing regular hexagon.
    """
    d = np.sqrt(3)  # flat-to-flat distance between adjacent unit hexagons
    
    # Axial to pixel conversion for hex lattice
    def axial_to_pixel(q, r):
        x = d * (q + r / 2.0)
        y = d * np.sqrt(3) / 2.0 * r
        return (x, y)
    
    # Generate N points in a hexagonal arrangement
    def hex_cluster_points(N):
        points = []
        points.append(axial_to_pixel(0, 0))
        
        ring = 1
        while len(points) < N:
            for i in range(6):
                if i == 0:
                    for j in range(ring):
                        if len(points) < N:
                            points.append(axial_to_pixel(ring, -j))
                elif i == 1:
                    for j in range(ring):
                        if len(points) < N:
                            points.append(axial_to_pixel(ring - j, -ring))
                elif i == 2:
                    for j in range(ring):
                        if len(points) < N:
                            points.append(axial_to_pixel(-j, -ring + j))
                elif i == 3:
                    for j in range(ring):
                        if len(points) < N:
                            points.append(axial_to_pixel(-ring, j))
                elif i == 4:
                    for j in range(ring):
                        if len(points) < N:
                            points.append(axial_to_pixel(-ring + j, ring))
                elif i == 5:
                    for j in range(ring):
                        if len(points) < N:
                            points.append(axial_to_pixel(j, ring - j))
            ring += 1
            if ring > 50:
                break
        
        return points[:N]
    
    points = hex_cluster_points(N)
    inradius = np.sqrt(3) / 2.0
    
    # For a given hexagon orientation theta and center c, compute required side length
    def required_side(theta, cx, cy):
        """Compute the side length needed for hexagon (cx, cy, theta) to contain all points + inradius clearance."""
        A_max = 0
        for i in range(6):
            angle = theta + np.pi / 3 * i
            nx = np.cos(angle)
            ny = np.sin(angle)
            dots = np.array([np.dot([px - cx, py - cy], [nx, ny]) for px, py in points])
            A_max = max(A_max, np.max(dots) + inradius, -np.min(dots) + inradius)
        return 2 * A_max / np.sqrt(3)
    
    # Optimize over theta and center
    def objective(params):
        theta, cx, cy = params
        return required_side(theta, cx, cy)
    
    # Try multiple starting points
    best_s = float('inf')
    best_params = None
    
    centroid = np.mean([p for p in points], axis=0)
    
    for theta0 in np.linspace(0, np.pi / 3, 7):
        x0 = [theta0, centroid[0], centroid[1]]
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x
    
    theta, cx, cy = best_params
    
    # Build inner list
    inner = [[p[0], p[1], 0.0] for p in points]
    
    return inner, (cx, cy), best_s, np.degrees(theta)
# EVOLVE-BLOCK-END

