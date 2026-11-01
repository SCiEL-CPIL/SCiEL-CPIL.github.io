import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 130

def _initial_fib():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _get_hull_faces(points):
    hull = ConvexHull(points)
    return hull.simplices

def _covering_radius(points):
    faces = _get_hull_faces(points)
    n_faces = len(faces)
    
    # Extract face vertices: (n_faces, 3, 3)
    A = points[faces[:, 0]]  # (n_faces, 3)
    B = points[faces[:, 1]]
    C = points[faces[:, 2]]
    
    # Compute circumcenters for all faces
    # v = cross(A - B, B - C)
    AB = A - B
    BC = B - C
    V = np.cross(AB, BC)
    V_norm = np.linalg.norm(V, axis=1, keepdims=True)
    V_norm = np.maximum(V_norm, 1e-15)
    centers = V / V_norm  # (n_faces, 3)
    
    # Compute circumradius for each face: arccos(center . A)
    dots = np.sum(centers * A, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    rads = np.arccos(dots)
    
    # Handle obtuse triangles: if circumcenter is outside, use max half-side
    # Compute side lengths (angular)
    AB_dot = np.clip(np.sum(A * B, axis=1), -1.0, 1.0)
    BC_dot = np.clip(np.sum(B * C, axis=1), -1.0, 1.0)
    CA_dot = np.clip(np.sum(C * A, axis=1), -1.0, 1.0)
    side_ab = np.arccos(AB_dot)
    side_bc = np.arccos(BC_dot)
    side_ca = np.arccos(CA_dot)
    
    max_half_side = 0.5 * np.maximum(np.maximum(side_ab, side_bc), side_ca)
    
    # For robustness, use max of circumradius and half-side for obtuse cases
    # But for N=130, triangles are small, so circumradius should be fine
    # Let's use the circumradius directly
    max_rad = np.max(rads)
    worst_idx = np.argmax(rads)
    
    return max_rad, worst_idx, faces

def _optimize(points, deadline, step_size=0.05, n_max_iter=50000):
    best_rad = float('inf')
    best_points = points.copy()
    
    for iter in range(n_max_iter):
        if time.time() > deadline:
            break
        
        rad, worst_idx, faces = _covering_radius(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()
        
        if rad < 1e-8:
            break
        
        face = faces[worst_idx]
        a = points[face[0]].copy()
        b = points[face[1]].copy()
        c = points[face[2]].copy()
        
        # Compute circumcenter
        v = np.cross(a - b, b - c)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            continue
        center = v / v_norm
        
        # Move vertices away from center
        for idx, p in zip(face, [a, b, c]):
            force = p - np.dot(p, center) * center
            force_norm = np.linalg.norm(force)
            if force_norm > 1e-10:
                force = force / force_norm
                points[idx] += step_size * force
                points[idx] /= np.linalg.norm(points[idx])
    
    return best_points, best_rad

def _polish(points, deadline, step_size=0.01, n_max_iter=10000):
    """High-precision polish pass with smaller steps."""
    best_rad = float('inf')
    best_points = points.copy()
    
    for iter in range(n_max_iter):
        if time.time() > deadline:
            break
        
        rad, worst_idx, faces = _covering_radius(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()
        
        if rad < 1e-10:
            break
        
        face = faces[worst_idx]
        a = points[face[0]].copy()
        b = points[face[1]].copy()
        c = points[face[2]].copy()
        
        v = np.cross(a - b, b - c)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            continue
        center = v / v_norm
        
        for idx, p in zip(face, [a, b, c]):
            force = p - np.dot(p, center) * center
            force_norm = np.linalg.norm(force)
            if force_norm > 1e-10:
                force = force / force_norm
                points[idx] += step_size * force
                points[idx] /= np.linalg.norm(points[idx])
    
    return best_points, best_rad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    points = _initial_fib()
    
    # Phase 1: Coarse optimization
    points, rad1 = _optimize(points, deadline - 30.0, step_size=0.05, n_max_iter=30000)
    
    # Phase 2: Fine optimization
    points, rad2 = _optimize(points, deadline - 10.0, step_size=0.02, n_max_iter=20000)
    
    # Phase 3: Polish
    points, rad3 = _polish(points, deadline, step_size=0.005, n_max_iter=10000)
    
    return points
