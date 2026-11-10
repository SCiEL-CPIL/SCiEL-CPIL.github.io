# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _hex_separation(v1, v2, tol=1e-6):
    """Check if two hexagons (given as vertex arrays) are separated."""
    for i in range(6):
        e1 = v1[(i+1)%6] - v1[i]
        n1 = np.array([-e1[1], e1[0]])
        norm = np.linalg.norm(n1)
        if norm > 1e-10:
            n1 = n1 / norm
            proj1 = v1 @ n1
            proj2 = v2 @ n1
            if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                return True
    
    for i in range(6):
        e2 = v2[(i+1)%6] - v2[i]
        n2 = np.array([-e2[1], e2[0]])
        norm = np.linalg.norm(n2)
        if norm > 1e-10:
            n2 = n2 / norm
            proj1 = v1 @ n2
            proj2 = v2 @ n2
            if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                return True
    
    return False


def _outer_hex_side(inner_hex_data, outer_angle_deg=0):
    """Compute minimum side of outer hexagon with given angle to contain all inner hexagons."""
    a = np.radians(outer_angle_deg)
    # For flat-top hexagon with angle a, edge normals are at a + 30 + 60*i
    normals = []
    for i in range(6):
        theta = a + np.radians(30 + 60 * i)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    
    max_proj = 0.0
    for hex_data in inner_hex_data:
        cx, cy, ha = hex_data
        v = _hex_vertices(cx, cy, 1.0, ha)
        for n in normals:
            proj = v @ n
            max_proj = max(max_proj, np.max(proj))
    
    # Apothem = s * sqrt(3)/2, so s = 2 * max_proj / sqrt(3)
    return 2 * max_proj / np.sqrt(3)


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Start with honeycomb arrangement
    sqrt3 = np.sqrt(3)
    dx = 2 * sqrt3
    dy = 3 * sqrt3 / 2
    
    # 11 hexagons: 1 center, 6 ring, 4 outer
    initial = np.array([
        [0, 0, 0],
        [dx, 0, 0],
        [dx/2, dy, 0],
        [-dx/2, dy, 0],
        [-dx, 0, 0],
        [-dx/2, -dy, 0],
        [dx/2, -dy, 0],
        [3*dx/2, 3*dy/2, 0],
        [-3*dx/2, 3*dy/2, 0],
        [3*dx/2, -3*dy/2, 0],
        [-3*dx/2, -3*dy/2, 0],
    ])
    
    # Compute initial side
    best_side = _outer_hex_side(initial, 0)
    best_config = initial.copy()
    
    # Try different outer hexagon angles
    for angle in np.arange(0, 30, 1):
        side = _outer_hex_side(initial, angle)
        if side < best_side:
            best_side = side
            best_angle = angle
    
    # Try small perturbations to improve packing
    n_iter = 0
    while time.time() < deadline:
        n_iter += 1
        # Random perturbation
        perturbed = best_config.copy()
        noise = np.random.randn(11, 2) * 0.1
        perturbed[:, :2] += noise
        perturbed[:, 2] = 0  # Keep angles at 0 for simplicity
        
        # Check validity
        valid = True
        for i in range(11):
            for j in range(i+1, 11):
                v1 = _hex_vertices(perturbed[i,0], perturbed[i,1], 1.0, perturbed[i,2])
                v2 = _hex_vertices(perturbed[j,0], perturbed[j,1], 1.0, perturbed[j,2])
                if not _hex_separation(v1, v2):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            side = _outer_hex_side(perturbed, 0)
            if side < best_side:
                best_side = side
                best_config = perturbed.copy()
    
    outer_hex_data = np.array([0, 0, 0])
    
    return best_config, outer_hex_data, best_side


# EVOLVE-BLOCK-END
