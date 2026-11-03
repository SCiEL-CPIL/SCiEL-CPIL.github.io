"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Place 12 unit hexagons in a compact hexagonal lattice arrangement
    d = np.sqrt(3)  # center-to-center distance for touching unit hexagons
    
    # Use a 4-4-4 staggered arrangement (3 rows of 4)
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * d + (d / 2 if row % 2 == 1 else 0)
            y = row * d * np.sqrt(3) / 2
            positions.append((x, y))
    
    positions = np.array(positions)
    
    # Center the arrangement
    center = positions.mean(axis=0)
    positions = positions - center
    
    def required_side(positions, container_angle_deg, inner_angle_deg=0.0):
        """Compute minimum side length of regular hexagon with given orientation 
        that contains all unit hexagons at given positions with given inner orientation."""
        phi = np.radians(container_angle_deg)
        theta = np.radians(inner_angle_deg)
        
        # 6 normals for the container hexagon (pointing outward from center)
        normals = []
        for j in range(6):
            angle = phi + j * np.pi / 3
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        normals = np.array(normals)
        
        max_constraint = 0.0
        for p in positions:
            # Vertices of unit hexagon at p with orientation theta
            for k in range(6):
                v_angle = theta + k * np.pi / 3
                v = p + np.array([np.cos(v_angle), np.sin(v_angle)])
                for n in normals:
                    val = np.dot(v, n)
                    if val > max_constraint:
                        max_constraint = val
        
        return max_constraint / (np.sqrt(3) / 2)
    
    # Optimize over container angle
    def objective(params):
        container_angle = params[0]
        return required_side(positions, np.degrees(container_angle))
    
    best_side = float('inf')
    best_angle = 0.0
    for init_angle in np.linspace(0, np.pi/3, 7):
        result = minimize(objective, [init_angle], bounds=[(0, np.pi/3)])
        if result.success and result.fun < best_side:
            best_side = result.fun
            best_angle = np.degrees(result.x[0])
    
    inner = []
    for p in positions:
        inner.append([p[0], p[1], 0.0])
    
    return inner, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
