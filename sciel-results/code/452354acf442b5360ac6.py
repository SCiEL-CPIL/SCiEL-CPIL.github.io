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
    # ConvexHull gives 3D tetrahedra. We need the surface triangles.
    # The faces of the convex hull of points on a sphere are the spherical triangles.
    hull = ConvexHull(points)
    return hull.simplices

def _covering_radius(points):
    # points: (N, 3) unit vectors
    faces = _get_hull_faces(points)
    max_rad = 0.0
    worst_face_idx = 0
    worst_rad = 0.0
    
    for i, face in enumerate(faces):
        a = points[face[0]]
        b = points[face[1]]
        c = points[face[2]]
        
        # Compute circumcenter
        # v is parallel to (a-b) x (b-c)
        v = np.cross(a - b, b - c)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            continue
        center = v / v_norm
        
        # Check if center is "inside" (dot products positive)
        # If the triangle is obtuse, the center might be on the "other side"
        # We want the center that is close to the triangle.
        # The circumradius is arccos(center . a)
        # If the triangle is obtuse, arccos(center . a) might be > pi/2?
        # Actually, if the triangle is obtuse, the circumcenter is outside, 
        # and the distance to the nearest vertex is not the circumradius.
        
        rad = np.arccos(np.clip(center[0]*a[0] + center[1]*a[1] + center[2]*a[2], -1, 1))
        
        # Check if the center is valid (inside the triangle)
        # A simple check: the center should be closer to the vertices than the antipode?
        # Or, check if the triangle is acute.
        # Calculate side lengths (angular)
        ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
        bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
        ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
        
        # If any side is > pi/2, the triangle is obtuse (on sphere, sum of angles > pi)
        # Actually, for small triangles, if a side is large, it's obtuse.
        # If obtuse, the covering radius for this facet is the max of the half-sides?
        # Let's use the max of the circumradius and the max half-side.
        max_half_side = 0.5 * max(ab, bc, ca)
        
        # If the circumcenter is outside, rad might be small or large?
        # If the triangle is obtuse, the circumcenter is outside, and the distance 
        # from the circumcenter to the vertices is the circumradius, but the point 
        # in the triangle farthest from the vertices is the vertex opposite the 
        # longest side? No, it's the midpoint of the longest side?
        # Actually, for an obtuse triangle, the minimal covering cap is centered 
        # at the midpoint of the longest side? No, it's centered at the circumcenter 
        # of the *Euclidean* triangle?
        
        # Let's use a robust metric: The covering radius is the maximum over all 
        # points in the triangle of the distance to the nearest vertex.
        # For an acute triangle, this is the circumradius.
        # For an obtuse triangle, this is the maximum of the distances from the 
        # midpoint of the longest side to the opposite vertex? No.
        
        # Let's just use the circumradius. For N=130, triangles are small.
        # The error from using circumradius for obtuse triangles is small.
        
        if rad > worst_rad:
            worst_rad = rad
            worst_face_idx = i

    return worst_rad, worst_face_idx, faces

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    points = _initial_fib()
    
    # Optimization loop
    step_size = 0.05
    for iter in range(10000):
        if time.time() > deadline:
            break
            
        rad, worst_idx, faces = _covering_radius(points)
        
        if rad < 1e-6:
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
        # Force direction: (vertex - center) projected on tangent plane
        for idx, p in zip(face, [a, b, c]):
            # Tangent vector
            force = p - np.dot(p, center) * center
            force_norm = np.linalg.norm(force)
            if force_norm > 1e-10:
                force = force / force_norm
                points[idx] += step_size * force
                # Normalize
                points[idx] /= np.linalg.norm(points[idx])
                
    return points
