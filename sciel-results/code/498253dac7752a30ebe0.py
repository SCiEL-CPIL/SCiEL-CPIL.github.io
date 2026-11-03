import math
import time
import numpy as np

N = 11
D = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = 145.0
    
    hex_verts_base = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ])

    def get_minkowski_verts(a1, a2):
        v1 = np.array([[math.cos(a1 + i * math.pi / 3), math.sin(a1 + i * math.pi / 3)] for i in range(6)])
        v2 = np.array([[math.cos(a2 + math.pi + i * math.pi / 3), math.sin(a2 + math.pi + i * math.pi / 3)] for i in range(6)])
        sums = np.array([v1[i] + v2[j] for i in range(6) for j in range(6)])
        
        # Convex hull of 36 points
        min_y_idx = np.argmin(sums[:, 1])
        min_y = sums[min_y_idx, 1]
        candidates = np.where(np.abs(sums[:, 1] - min_y) < 1e-12)[0]
        start_idx = candidates[np.argmin(sums[candidates, 0])]
        start_pt = sums[start_idx]
        
        angles_rel = np.arctan2(sums[:, 1] - start_pt[1], sums[:, 0] - start_pt[0])
        sorted_indices = np.argsort(angles_rel)
        sorted_points = sums[sorted_indices]
        
        unique_pts = []
        for p in sorted_points:
            if not unique_pts or np.linalg.norm(p - unique_pts[-1]) > 1e-12:
                unique_pts.append(p)
        if len(unique_pts) > 1 and np.linalg.norm(unique_pts[0] - unique_pts[-1]) < 1e-12:
            unique_pts.pop()
            
        if len(unique_pts) < 3:
            return np.array(unique_pts)
            
        stack = []
        for p in unique_pts:
            while len(stack) >= 2:
                p1 = stack[-2]
                p2 = stack[-1]
                cross = (p2[0]-p1[0])*(p[1]-p1[1]) - (p2[1]-p1[1])*(p[0]-p1[0])
                if cross <= 1e-12:
                    stack.pop()
                else:
                    break
            stack.append(p)
            
        return np.array(stack)

    def point_in_convex_polygon(point, polygon):
        n = len(polygon)
        if n < 3:
            return False, float('inf')
        
        min_dist = float('inf')
        for i in range(n):
            v1 = polygon[i]
            v2 = polygon[(i+1) % n]
            edge = v2 - v1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-15:
                continue
            
            cross = (edge[0] * (point[1] - v1[1]) - edge[1] * (point[0] - v1[0]))
            
            if cross < -1e-10:
                t = np.dot(point - v1, edge) / (edge_len * edge_len)
                t = max(0, min(1, t))
                closest = v1 + t * edge
                dist = np.linalg.norm(point - closest)
                return False, dist
            
            t = np.dot(point - v1, edge) / (edge_len * edge_len)
            t = max(0, min(1, t))
            closest = v1 + t * edge
            dist = np.linalg.norm(point - closest)
            if dist < min_dist:
                min_dist = dist
                
        return True, min_dist

    def compute_overlap_penalty(centers, angles):
        penalty = 0.0
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                mink_verts = get_minkowski_verts(angles[i], angles[j])
                inside, dist = point_in_convex_polygon(diff, mink_verts)
                if inside:
                    penalty += dist * dist
        return penalty

    def get_all_vertices(centers, angles):
        n = len(centers)
        all_verts = np.zeros((n * 6, 2))
        for i in range(n):
            c = math.cos(angles[i])
            s = math.sin(angles[i])
            R = np.array([[c, -s], [s, c]])
            rotated = hex_verts_base @ R.T
            all_verts[i*6:(i+1)*6] = rotated + centers[i]
        return all_verts

    def objective(x):
        centers = x[:33].reshape(11, 3)[:, :2]
        angles = x[:33].reshape(11, 3)[:, 2]
        
        all_verts = get_all_vertices(centers, angles)
        
        normals = np.array([
            [math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)]
            for k in range(6)
        ])
        
        projs = all_verts @ normals.T
        max_abs_proj = np.max(np.abs(projs))
        S = (2.0 / math.sqrt(3)) * max_abs_proj
        
        penalty = compute_overlap_penalty(centers, angles)
        
        return S + 100.0 * penalty

    init_pts = []
    init_pts.append((0.0, 0.0, 0.0))
    d = D
    for k in range(6):
        a = k * math.pi / 3
        init_pts.append((d * math.cos(a), d * math.sin(a), 0.0))
    for k in range(4):
        a = k * math.pi / 2 + math.pi / 4
        init_pts.append((2 * d * math.cos(a), 2 * d * math.sin(a), 0.0))
        
    x0 = np.array([item for p in init_pts for item in p])
    
    best_S = float('inf')
    best_x = x0.copy()
    
    current_x = x0.copy()
    current_cost = objective(current_x)
    best_cost = current_cost
    
    T = 1.0
    cooling = 0.999
    steps = 5000
    
    for i in range(steps):
        if time.time() > deadline:
            break
            
        step_size = 0.5 * T
        new_x = current_x + np.random.normal(0, step_size, 33)
        new_x[2::3] = (new_x[2::3] + math.pi) % (2 * math.pi) - math.pi
        
        new_cost = objective(new_x)
        
        if new_cost < current_cost:
            current_x = new_x
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_x = new_x.copy()
        else:
            delta = new_cost - current_cost
            if delta < 0:
                accept = True
            else:
                accept = np.random.rand() < math.exp(-delta / T)
            
            if accept:
                current_x = new_x
                current_cost = new_cost
        
        T *= cooling
        if T < 0.01:
            break

    centers = best_x[:33].reshape(11, 3)[:, :2]
    angles = best_x[:33].reshape(11, 3)[:, 2]
    
    all_verts = get_all_vertices(centers, angles)
    normals = np.array([
        [math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)]
        for k in range(6)
    ])
    projs = all_verts @ normals.T
    max_abs_proj = np.max(np.abs(projs))
    final_S = (2.0 / math.sqrt(3)) * max_abs_proj
    
    if final_S > 20 or not np.isfinite(final_S):
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner_rows = []
    for i in range(11):
        inner_rows.append([float(centers[i][0]), float(centers[i][1]), float(math.degrees(angles[i]))])
        
    center = (0.0, 0.0)
    side = float(final_S)
    angle_deg = 0.0
    
    return inner_rows, center, side, angle_deg
