"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize_scalar
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    # Define topologies as lists of (x, y) relative coordinates
    # Topology 1: 3x4 Rectangle
    coords_34 = []
    for i in range(4):
        coords_34.append([i * np.sqrt(3), 0.0])
    for i in range(4):
        coords_34.append([i * np.sqrt(3) + np.sqrt(3)/2, 1.5])
    for i in range(4):
        coords_34.append([i * np.sqrt(3), 3.0])
    coords_34 = np.array(coords_34)

    # Topology 2: 2x6 Rectangle
    coords_26 = []
    for i in range(6):
        coords_26.append([i * np.sqrt(3), 0.0])
    for i in range(6):
        coords_26.append([i * np.sqrt(3) + np.sqrt(3)/2, 1.5])
    coords_26 = np.array(coords_26)

    def optimize_topology(coords):
        # Center the coords
        coords_centered = coords - coords.mean(axis=0)
        
        # Normals of outer hexagon (flat-top)
        # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        normals = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        
        def objective(theta):
            # Rotate
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            rotated = coords_centered @ R.T
            
            # Projections
            projections = rotated @ normals.T
            max_projs = projections.max(axis=0)
            
            # S >= 2 * max_proj / sqrt(3)
            S = np.max(2.0 * max_projs / np.sqrt(3))
            return S
        
        # Optimize theta in [0, pi/6]
        res = minimize_scalar(objective, bounds=(0, np.pi/6), method='bounded', options={'xatol': 1e-8})
        
        theta_opt = res.x
        S_opt = res.fun
        
        # Construct final positions
        c, s = np.cos(theta_opt), np.sin(theta_opt)
        R = np.array([[c, -s], [s, c]])
        final_coords = coords_centered @ R.T
        
        inner = []
        angle_deg = np.degrees(theta_opt)
        for p in final_coords:
            inner.append([p[0], p[1], angle_deg])
            
        return inner, S_opt

    best_S = float('inf')
    best_inner = None

    # Try Topology 1
    if time.time() < deadline:
        inner_34, S_34 = optimize_topology(coords_34)
        if S_34 < best_S:
            best_S = S_34
            best_inner = inner_34

    # Try Topology 2
    if time.time() < deadline:
        inner_26, S_26 = optimize_topology(coords_26)
        if S_26 < best_S:
            best_S = S_26
            best_inner = inner_26

    if best_inner is None:
        # Fallback
        return [[0.0, 0.0, 0.0]] * N, (0.0, 0.0), 10.0, 0.0

    return best_inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END
