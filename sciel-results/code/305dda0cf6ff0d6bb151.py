import numpy as np
from scipy.optimize import minimize_scalar

N = 11

def construct():
    # Unit regular hexagon vertices (side length 1, centered at origin)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    hex_verts = np.array([[np.cos(np.pi/3 * i), np.sin(np.pi/3 * i)] for i in range(6)])
    
    # Hex grid spacing: distance between centers of adjacent hexagons = sqrt(3)
    s3 = np.sqrt(3.0)
    
    # Compact arrangement of 11 hexagons using axial coordinates
    axial_positions = [
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        (2, 0),
        (1, 1),
        (-1, 2),
        (-2, 1),
    ]
    
    centers = []
    for q, r in axial_positions:
        x = s3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append([x, y])
    centers = np.array(centers)
    
    # Compute all vertices of all hexagons
    all_vertices = []
    for c in centers:
        verts = hex_verts + c
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    def convex_hull(points):
        points = np.array(points)
        points = points[np.lexsort((points[:, 0], points[:, 1]))]
        if len(points) <= 1:
            return points
        
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        upper = []
        for p in points[::-1]:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        
        return np.vstack([lower[:-1], upper[:-1]])
    
    hull = convex_hull(all_vertices)
    
    def min_enclosing_hex_side(rotation):
        # Outward normals at rotation + 30° + k*60°
        normals = []
        for k in range(6):
            angle = rotation + np.pi/6 + k * np.pi / 3
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)
        
        projections = hull @ normals.T
        
        widths = []
        for k in range(6):
            proj = projections[:, k]
            width = np.max(proj) - np.min(proj)
            widths.append(width)
        
        max_width = max(widths)
        s = max_width / np.sqrt(3.0)
        
        midpoints = []
        for k in range(6):
            proj = projections[:, k]
            mid = (np.max(proj) + np.min(proj)) / 2.0
            midpoints.append(mid)
        midpoints = np.array(midpoints)
        
        c, _, _, _ = np.linalg.lstsq(normals, midpoints, rcond=None)
        
        return s, c
    
    def objective(theta):
        s, c = min_enclosing_hex_side(theta)
        return s
    
    result = minimize_scalar(objective, bounds=(0, np.pi / 3), method='bounded', options={'xatol': 1e-10})
    opt_theta = result.x
    opt_s, opt_c = min_enclosing_hex_side(opt_theta)
    
    inner = []
    for c in centers:
        inner.append([float(c[0]), float(c[1]), 0.0])
    
    center = [float(opt_c[0]), float(opt_c[1])]
    side = float(opt_s)
    angle_deg = float(np.degrees(opt_theta))
    
    return inner, center, side, angle_deg
