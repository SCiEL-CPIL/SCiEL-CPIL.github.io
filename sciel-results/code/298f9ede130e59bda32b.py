import math
import time
import numpy as np

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin

    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def get_gap(v1, v2):
        max_gap = -float('inf')
        for i in range(3):
            e1 = v1[(i+1)%6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            norm1 = np.linalg.norm(n1)
            if norm1 > 1e-9:
                n1 = n1 / norm1
            
            proj1 = np.dot(v1, n1)
            proj2 = np.dot(v2, n1)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                gap = - (min(max1, max2) - max(min1, min2))
                
            if gap > max_gap:
                max_gap = gap
                
            e2 = v2[(i+1)%6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            norm2 = np.linalg.norm(n2)
            if norm2 > 1e-9:
                n2 = n2 / norm2
                
            proj1_b = np.dot(v1, n2)
            proj2_b = np.dot(v2, n2)
            min1_b, max1_b = np.min(proj1_b), np.max(proj1_b)
            min2_b, max2_b = np.min(proj2_b), np.max(proj2_b)
            
            if max1_b < min2_b:
                gap = min2_b - max1_b
            elif max2_b < min1_b:
                gap = min1_b - max2_b
            else:
                gap = - (min(max1_b, max2_b) - max(min1_b, min2_b))
                
            if gap > max_gap:
                max_gap = gap
                
        return max_gap

    def is_valid(centers, angles):
        verts_list = [hex_vertices(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = get_gap(verts_list[i], verts_list[j])
                if gap < -1e-5:
                    return False
        return True

    def compute_side(centers, angles):
        max_proj = 0.0
        for i in range(N):
            verts = hex_vertices(centers[i][0], centers[i][1], angles[i])
            for k in range(6):
                a = k * math.pi / 3
                n = np.array([math.cos(a), math.sin(a)])
                proj = np.dot(verts, n)
                max_proj = max(max_proj, np.max(proj))
        
        return 2.0 * max_proj / math.sqrt(3)

    # 1. Generate Lattice Candidate
    candidates = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            dist = x*x + y*y
            candidates.append((dist, x, y))
            
    candidates.sort()
    best_candidates = [c for c in candidates[:12]]
    
    centers = [(c[1], c[2]) for c in best_candidates]
    angles = [0.0] * N
    
    # Check validity
    if not is_valid(centers, angles):
        # Fallback: simple grid
        centers = []
        for r in range(4):
            for c in range(3):
                x = c * 2.0
                y = r * math.sqrt(3)
                if r % 2 == 1:
                    x += 1.0
                centers.append((x, y))
        cx_mean = sum(c[0] for c in centers) / N
        cy_mean = sum(c[1] for c in centers) / N
        centers = [(x - cx_mean, y - cy_mean) for x, y in centers]
        angles = [0.0] * N
        if not is_valid(centers, angles):
             return [[0.0, 0.0, 0.0] for _ in range(N)], (0.0, 0.0), 100.0, 0.0

    best_side = compute_side(centers, angles)
    best_centers = centers.copy()
    best_angles = angles.copy()

    # 2. Local Optimization (Simulated Annealing)
    T = 1.0
    T_min = 0.001
    alpha = 0.999
    
    while time.time() < deadline and T > T_min:
        idx = np.random.randint(0, N)
        new_centers = [c for c in centers] # Shallow copy of tuples is fine if we replace
        new_angles = angles.copy()
        
        if np.random.random() < 0.7:
            new_centers[idx] = (centers[idx][0] + np.random.normal(0, 0.1), 
                                centers[idx][1] + np.random.normal(0, 0.1))
        else:
            new_angles[idx] = angles[idx] + np.random.normal(0, 0.1)
            
        if is_valid(new_centers, new_angles):
            new_side = compute_side(new_centers, new_angles)
            delta = new_side - best_side
            if delta < 0 or np.random.random() < math.exp(-delta / T):
                centers = new_centers
                angles = new_angles
                best_side = new_side
                best_centers = new_centers
                best_angles = new_angles
                
        T *= alpha

    inner = []
    for i in range(N):
        inner.append([float(best_centers[i][0]), float(best_centers[i][1]), float(best_angles[i])])
        
    final_side = best_side * 1.01 + 1e-4
    
    return inner, (0.0, 0.0), float(final_side), 0.0
