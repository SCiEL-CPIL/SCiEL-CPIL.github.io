import numpy as np
import random
import math

N = 12

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon with side length `side`,
    center (cx,cy), rotation angle in degrees. Vertex 0 at angle `angle_deg`."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def polygons_overlap(v1, v2):
    """Check if two convex polygons overlap using SAT."""
    for verts in (v1, v2):
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            # normal
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            nx, ny = -dy, dx
            # project both polygons
            min1, max1 = float('inf'), float('-inf')
            min2, max2 = float('inf'), float('-inf')
            for p in v1:
                proj = p[0] * nx + p[1] * ny
                min1 = min(min1, proj)
                max1 = max(max1, proj)
            for p in v2:
                proj = p[0] * nx + p[1] * ny
                min2 = min(min2, proj)
                max2 = max(max2, proj)
            if max1 < min2 or max2 < min1:
                return False
    return True

def min_enclosing_hex_radius(verts_all):
    """Given list of all vertices, find min circumradius S and rotation alpha
    of a regular hexagon (with center at centroid) that encloses all points."""
    pts = np.array(verts_all)
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()
    best_S = float('inf')
    best_alpha = 0.0
    for alpha_deg in range(0, 60, 2):
        alpha = math.radians(alpha_deg)
        # side normals
        max_proj = 0.0
        for k in range(6):
            theta = alpha + math.pi / 6 + k * math.pi / 3
            nx = math.cos(theta)
            ny = math.sin(theta)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, projs.max())
        S = max_proj / (math.sqrt(3) / 2)
        if S < best_S:
            best_S = S
            best_alpha = alpha_deg
    return best_S, best_alpha

def construct():
    random.seed(42)
    np.random.seed(42)
    
    # Initial random configuration
    positions = []
    angles = []
    for i in range(N):
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
        a = random.uniform(0, 60)
        positions.append([x, y])
        angles.append(a)
    
    def get_verts(positions, angles):
        all_verts = []
        for i in range(N):
            v = hex_vertices(positions[i][0], positions[i][1], angles[i])
            all_verts.extend(v)
        return all_verts
    
    def objective(positions, angles):
        # Check overlap
        verts_list = []
        for i in range(N):
            verts_list.append(hex_vertices(positions[i][0], positions[i][1], angles[i]))
        for i in range(N):
            for j in range(i + 1, N):
                if polygons_overlap(verts_list[i], verts_list[j]):
                    return float('inf')
        all_verts = []
        for v in verts_list:
            all_verts.extend(v)
        S, _ = min_enclosing_hex_radius(all_verts)
        return S
    
    best_positions = [p[:] for p in positions]
    best_angles = angles[:]
    best_obj = objective(best_positions, best_angles)
    
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    max_iter = 50000
    rng = np.random.RandomState(42)
    
    for it in range(max_iter):
        T = max(T_min, T * cooling)
        # Pick a random hexagon to move
        i = rng.randint(0, N)
        dx = rng.normal(0, 0.5)
        dy = rng.normal(0, 0.5)
        da = rng.normal(0, 5.0)
        
        old_x, old_y = positions[i][0], positions[i][1]
        old_a = angles[i]
        positions[i][0] += dx
        positions[i][1] += dy
        angles[i] = (old_a + da) % 60
        
        new_obj = objective(positions, angles)
        
        if new_obj < best_obj:
            best_positions = [p[:] for p in positions]
            best_angles = angles[:]
            best_obj = new_obj
        else:
            # Accept with probability exp(-(new-old)/T)
            if new_obj == float('inf'):
                delta = 10.0
            else:
                delta = new_obj - best_obj
            if delta < 0 or rng.random() < math.exp(-delta / T):
                pass  # accept
            else:
                positions[i][0] = old_x
                positions[i][1] = old_y
                angles[i] = old_a
    
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
    
    all_verts = []
    for i in range(N):
        v = hex_vertices(best_positions[i][0], best_positions[i][1], best_angles[i])
        all_verts.extend(v)
    
    S, alpha_deg = min_enclosing_hex_radius(all_verts)
    pts = np.array(all_verts)
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()
    
    return inner, (cx, cy), S, alpha_deg
