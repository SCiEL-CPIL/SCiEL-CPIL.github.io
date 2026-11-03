"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    def hex_vertices(cx, cy, angle_deg, side=1.0):
        """Return 6 vertices of a regular hexagon centered at (cx,cy) with given angle."""
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3.0
            vertices.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
        return np.array(vertices)
    
    def min_outer_side(inner_hexes, outer_angle_deg):
        """Compute minimum outer hexagon side length (centered at origin) to contain all inner hexagons."""
        angle = np.radians(outer_angle_deg)
        apothem_needed = 0.0
        for (cx, cy, ang) in inner_hexes:
            verts = hex_vertices(cx, cy, ang)
            for v in verts:
                for i in range(6):
                    theta = angle + i * np.pi / 3.0
                    proj = np.cos(theta) * v[0] + np.sin(theta) * v[1]
                    if proj > apothem_needed:
                        apothem_needed = proj
        side = 2.0 * apothem_needed / np.sqrt(3.0)
        return side
    
    def objective(x):
        """x = [ix0, iy0, ia0, ..., ix10, iy10, ia10, outer_angle_deg]"""
        inner = [(x[i*3], x[i*3+1], x[i*3+2]) for i in range(N)]
        outer_angle = x[-1]
        return min_outer_side(inner, outer_angle)
    
    # Initial guess: compact cluster
    hex_spacing = np.sqrt(3)
    initial_positions = []
    initial_positions.append((0.0, 0.0, 0.0))
    for i in range(6):
        angle = i * np.pi / 3.0
        initial_positions.append((hex_spacing * np.cos(angle), hex_spacing * np.sin(angle), 0.0))
    for i in range(4):
        angle = i * 2 * np.pi / 6.0 + np.pi/6.0
        initial_positions.append((2 * hex_spacing * np.cos(angle), 2 * hex_spacing * np.sin(angle), 0.0))
    
    x0 = []
    for (cx, cy, ang) in initial_positions:
        x0.extend([cx, cy, ang])
    x0.append(0.0)  # outer angle
    
    best_side = objective(x0)
    best_x = x0.copy()
    
    # Optimize with Nelder-Mead
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 8000, 'xatol': 1e-7, 'fatol': 1e-7})
    
    if result.fun < best_side:
        best_side = result.fun
        best_x = result.x.copy()
    
    # Random restarts
    np.random.seed(42)
    for restart in range(10):
        if time.time() > deadline - 10.0:
            break
        x_rand = []
        for i in range(N):
            cx = np.random.uniform(-4, 4)
            cy = np.random.uniform(-4, 4)
            ang = np.random.uniform(0, 180)
            x_rand.extend([cx, cy, ang])
        x_rand.append(np.random.uniform(0, 60))
        
        res = minimize(objective, x_rand, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-7})
        if res.fun < best_side:
            best_side = res.fun
            best_x = res.x.copy()
    
    # Extract inner hexagons
    inner = []
    for i in range(N):
        inner.append([best_x[i*3], best_x[i*3+1], best_x[i*3+2]])
    
    outer_angle = best_x[-1]
    
    # Final verification and safety margin
    actual_side = min_outer_side(inner, outer_angle)
    if actual_side > best_side * (1 + 1e-5):
        best_side = actual_side
    best_side *= 1.001  # tiny safety margin
    
    return inner, (0.0, 0.0), best_side, outer_angle
# EVOLVE-BLOCK-END

