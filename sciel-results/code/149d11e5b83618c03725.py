"""Packing 12 unit hexagons in a minimal regular hexagon via Powell + restarts."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # safety margin under 150s wall-clock

# Precompute unit hexagon vertices (flat-top, side=1)
def _unit_hex_vertices():
    verts = []
    for k in range(6):
        a = np.radians(60.0 * k)
        verts.append((np.cos(a), np.sin(a)))
    return np.array(verts)

UNIT_VERTS = _unit_hex_vertices()

def _get_hex_verts(positions):
    """
    positions: (N, 3) array of [x, y, angle_deg]
    returns: (N, 6, 2) array of vertices
    """
    N = len(positions)
    verts = np.zeros((N, 6, 2))
    angles = np.radians(positions[:, 2])
    c = np.cos(angles)
    s = np.sin(angles)
    vx = UNIT_VERTS[:, 0]
    vy = UNIT_VERTS[:, 1]
    new_vx = c[:, np.newaxis] * vx[np.newaxis, :] - s[:, np.newaxis] * vy[np.newaxis, :]
    new_vy = s[:, np.newaxis] * vx[np.newaxis, :] + c[:, np.newaxis] * vy[np.newaxis, :]
    verts[:, :, 0] = new_vx + positions[:, 0:1]
    verts[:, :, 1] = new_vy + positions[:, 1:2]
    return verts

def _sat_overlap_fast(verts_i, verts_j):
    """
    Check overlap between two hexagons (6, 2) and (6, 2).
    Returns True if overlap.
    """
    edges_i = np.roll(verts_i, -1, axis=0) - verts_i
    norms_i = np.column_stack([-edges_i[:, 1], edges_i[:, 0]])
    norms_i /= np.linalg.norm(norms_i, axis=1, keepdims=True)
    
    edges_j = np.roll(verts_j, -1, axis=0) - verts_j
    norms_j = np.column_stack([-edges_j[:, 1], edges_j[:, 0]])
    norms_j /= np.linalg.norm(norms_j, axis=1, keepdims=True)
    
    axes = np.vstack([norms_i, norms_j])
    
    for axis in axes:
        proj_i = np.dot(verts_i, axis)
        proj_j = np.dot(verts_j, axis)
        if np.max(proj_i) < np.min(proj_j) - 1e-9 or np.max(proj_j) < np.min(proj_i) - 1e-9:
            return False
    return True

def _min_outer_side(verts):
    """
    verts: (N, 6, 2)
    Returns (minimal_side, best_angle_deg)
    """
    all_verts = verts.reshape(-1, 2)
    centroid = np.mean(all_verts, axis=0)
    
    best_side = float('inf')
    best_angle = 0.0
    
    # Try orientations 0, 30, 60, 90, 120, 150
    for ang_deg in [0, 30, 60, 90, 120, 150]:
        ang = np.radians(ang_deg)
        base_normals = np.array([
            [np.cos(np.radians(30)), np.sin(np.radians(30))],
            [np.cos(np.radians(90)), np.sin(np.radians(90))],
            [np.cos(np.radians(150)), np.sin(np.radians(150))],
            [np.cos(np.radians(210)), np.sin(np.radians(210))],
            [np.cos(np.radians(270)), np.sin(np.radians(270))],
            [np.cos(np.radians(330)), np.sin(np.radians(330))]
        ])
        
        c = np.cos(ang)
        s = np.sin(ang)
        R = np.array([[c, -s], [s, c]])
        normals = R @ base_normals.T
        
        # Project all vertices onto normals
        projs = np.dot(all_verts, normals.T)
        
        C = centroid
        
        d = np.max(projs - np.dot(C, normals.T), axis=0)
        m = np.min(projs - np.dot(C, normals.T), axis=0)
        
        req = np.maximum(d, -m)
        max_req = np.max(req)
        side = max_req / (np.sqrt(3) / 2)
        
        if side < best_side:
            best_side = side
            best_angle = ang_deg
            
    return best_side, best_angle

def _energy(params):
    """
    params: (N*3,) array
    """
    positions = params.reshape(N, 3)
    verts = _get_hex_verts(positions)
    
    # Outer side
    outer_side, _ = _min_outer_side(verts)
    
    # Overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            c_dist = np.linalg.norm(positions[i, :2] - positions[j, :2])
            if c_dist < 2.1:
                if _sat_overlap_fast(verts[i], verts[j]):
                    penalty += (2.0 - c_dist) ** 2 * 5.0
    
    return outer_side + penalty

def construct():
    start_time = time.time()
    
    # Initial config: 3x4 grid, tightly packed
    positions = []
    for r in range(3):
        for c in range(4):
            x = c * 1.5 + (0.75 if r % 2 == 1 else 0)
            y = r * (np.sqrt(3) / 2)
            positions.append([x, y, 0.0])
            
    params = np.array(positions).flatten()
    
    # Use Powell optimization
    result = minimize(_energy, params, method='Powell', options={'maxiter': 100, 'disp': False})
    
    best_params = result.x
    best_energy = result.fun
    
    # Restart loop
    while time.time() < DEADLINE - 5:
        # Random restart
        noise = np.random.normal(0, 0.5, N * 3)
        new_params = best_params + noise
        
        res = minimize(_energy, new_params, method='Powell', options={'maxiter': 50, 'disp': False})
        
        if res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x
            
    # Extract final positions
    final_positions = best_params.reshape(N, 3)
    
    # Calculate final outer side and center
    verts = _get_hex_verts(final_positions)
    all_verts = verts.reshape(-1, 2)
    centroid = np.mean(all_verts, axis=0)
    
    outer_side, outer_angle = _min_outer_side(verts)
    
    inner = [[p[0], p[1], p[2]] for p in final_positions]
    
    return inner, [centroid[0], centroid[1]], outer_side, outer_angle
