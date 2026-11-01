import numpy as np
import time

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle + k * np.pi / 3
        verts.append((cx + side * np.cos(a), cy + side * np.sin(a)))
    return np.array(verts)

def min_enclosing_hexagon(points):
    points = np.array(points)
    centroid = points.mean(axis=0)
    centered = points - centroid
    best_s = float('inf')
    best_theta = 0.0
    for theta_deg in np.linspace(0, 60, 121):
        theta = np.radians(theta_deg)
        max_s = 0.0
        for k in range(6):
            a = theta + k * np.pi / 3
            n = np.array([np.cos(a), np.sin(a)])
            projections = centered @ n
            proj_range = projections.max() - projections.min()
            s_needed = proj_range / np.sqrt(3)
            if s_needed > max_s:
                max_s = s_needed
        if max_s < best_s:
            best_s = max_s
            best_theta = theta
    return best_s, best_theta, centroid

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    d = np.sqrt(3)
    def axial_to_cartesian(q, r):
        x = d * (q + r / 2.0)
        y = d * (np.sqrt(3) / 2.0) * r
        return (x, y)
    first_ring = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
    second_ring = [(2,0),(2,-1),(1,-2),(0,-2),(-1,-2),(-2,-1),(-2,0),(-2,1),(-1,2),(0,2),(1,2),(2,1)]
    configs = []
    for sr in [
        [(2,0),(0,2),(-2,0),(0,-2)],
        [(2,0),(1,-2),(-1,-2),(-2,0)],
        [(2,0),(2,-1),(-2,0),(-2,1)],
        [(2,0),(2,1),(-2,0),(-2,-1)],
    ]:
        centers = [(0.0,0.0)] + [axial_to_cartesian(q,r) for q,r in first_ring] + [axial_to_cartesian(q,r) for q,r in sr]
        configs.append(centers)
    best_side = float('inf')
    best_inner = None
    best_center = (0.0,0.0)
    best_angle = 0.0
    for centers in configs:
        if time.time() > deadline - 10.0:
            break
        all_verts = np.vstack([hexagon_vertices(cx,cy,1.0,0.0) for cx,cy in centers])
        side, theta, centroid = min_enclosing_hexagon(all_verts)
        if side < best_side:
            best_side = side
            best_inner = [[cx,cy,0.0] for cx,cy in centers]
            best_center = (centroid[0],centroid[1])
            best_angle = np.degrees(theta)
    if best_inner is not None:
        return best_inner, best_center, best_side, best_angle
    inner = [[4.0*k,0.0,0.0] for k in range(-(N//2), N-N//2)]
    return inner, (0.0,0.0), 40.0, 0.0
