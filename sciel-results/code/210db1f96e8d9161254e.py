import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    d = np.sqrt(3)
    centers = []
    for i in range(4):
        centers.append((i * d, 0.0))
    for i in range(4):
        centers.append((i * d + d/2, 1.5))
    for i in range(3):
        centers.append((i * d, 3.0))
    
    cx_avg = np.mean([c[0] for c in centers])
    cy_avg = np.mean([c[1] for c in centers])
    centers = [(c[0] - cx_avg, c[1] - cy_avg) for c in centers]
    
    angles = np.deg2rad(np.arange(0, 360, 60))
    base_vertices = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    def get_inner_vertices(center, angle_deg):
        theta = np.deg2rad(angle_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        verts = base_vertices @ rot.T + center
        return verts

    def point_in_polygon(point, polygon):
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i+1)%n]
            v1 = p2 - p1
            v2 = point - p1
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            if cross < -1e-9:
                return False
        return True

    def check_inside(S, theta_outer, centers, inner_angles):
        theta_outer_rad = np.deg2rad(theta_outer)
        rot_outer = np.array([[np.cos(theta_outer_rad), -np.sin(theta_outer_rad)], [np.sin(theta_outer_rad), np.cos(theta_outer_rad)]])
        outer_verts = S * base_vertices @ rot_outer.T
        
        for i, center in enumerate(centers):
            verts = get_inner_vertices(center, inner_angles[i])
            for v in verts:
                if not point_in_polygon(v, outer_verts):
                    return False
        return True

    best_S = 100.0
    best_theta = 0.0
    
    for theta_outer in range(0, 30, 5):
        lo = 1.0
        hi = 10.0
        for _ in range(20):
            mid = (lo + hi) / 2
            if check_inside(mid, theta_outer, centers, [0.0]*N):
                hi = mid
            else:
                lo = mid
        if hi < best_S:
            best_S = hi
            best_theta = theta_outer

    inner = [[c[0], c[1], 0.0] for c in centers]
    center = (0.0, 0.0)
    side = best_S
    angle = best_theta
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
