"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _convex_hull_area(points):
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i+1, n):
            pj = points[j]
            for k in range(j+1, n):
                pk = points[k]
                v1 = pj - pi
                v2 = pk - pi
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def construct():
    start_time = time.time()
    deadline = 110.0
    
    best_score = -1
    best_points = None
    
    tri_indices = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    n_tris = len(tri_indices)
    
    def get_score(pts):
        hull_area = _convex_hull_area(pts)
        if hull_area == 0:
            return 0
        min_area = _min_triangle_area(pts)
        return min_area / hull_area

    def calc_areas_and_signs(points):
        pi = points[tri_indices[:, 0]]
        pj = points[tri_indices[:, 1]]
        pk = points[tri_indices[:, 2]]
        v1 = pj - pi
        v2 = pk - pi
        crosses = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(crosses)
        signs = np.sign(crosses)
        return areas, signs

    # Initializations
    inits = []
    i = np.arange(N)
    r = np.sqrt(i / N)
    theta = i * 2.399963229728653
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    base_init = np.column_stack((x, y))
    base_init -= np.mean(base_init, axis=0)
    
    for rot in range(20):
        angle = rot * 2 * np.pi / 20
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        init = base_init @ R.T
        inits.append(init)
        
    rng = np.random.default_rng(42)
    for _ in range(10):
        init = base_init + rng.normal(0, 0.1, (N, 2))
        inits.append(init)

    for init in inits:
        if time.time() > deadline - 10:
            break
            
        points = init.copy()
        
        step_size = 0.05
        for step in range(1000):
            if time.time() > deadline:
                break
                
            areas, signs = calc_areas_and_signs(points)
            min_idx = np.argmin(areas)
            i, j, k = tri_indices[min_idx]
            
            pi = points[i]
            pj = points[j]
            pk = points[k]
            
            # Gradient of signed area
            # dA/dpi = 0.5 * perp(pj - pk)
            # dA/dpj = 0.5 * perp(pk - pi)
            # dA/dpk = 0.5 * perp(pi - pj)
            
            v_jk = pj - pk
            v_ki = pk - pi
            v_ij = pi - pj
            
            g_i = 0.5 * np.array([-v_jk[1], v_jk[0]])
            g_j = 0.5 * np.array([-v_ki[1], v_ki[0]])
            g_k = 0.5 * np.array([-v_ij[1], v_ij[0]])
            
            s = signs[min_idx]
            
            points[i] += step_size * s * g_i
            points[j] += step_size * s * g_j
            points[k] += step_size * s * g_k
            
            hull_area = _convex_hull_area(points)
            if hull_area > 0:
                scale = 1.0 / np.sqrt(hull_area)
                points *= scale
                
            step_size *= 0.999
            
        score = get_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
    if best_points is not None and time.time() < deadline - 5:
        points = best_points.copy()
        step_size = 0.01
        for step in range(200):
            if time.time() > deadline:
                break
            areas, signs = calc_areas_and_signs(points)
            min_idx = np.argmin(areas)
            i, j, k = tri_indices[min_idx]
            
            pi = points[i]
            pj = points[j]
            pk = points[k]
            
            v_jk = pj - pk
            v_ki = pk - pi
            v_ij = pi - pj
            
            g_i = 0.5 * np.array([-v_jk[1], v_jk[0]])
            g_j = 0.5 * np.array([-v_ki[1], v_ki[0]])
            g_k = 0.5 * np.array([-v_ij[1], v_ij[0]])
            
            s = signs[min_idx]
            
            points[i] += step_size * s * g_i
            points[j] += step_size * s * g_j
            points[k] += step_size * s * g_k
            
            hull_area = _convex_hull_area(points)
            if hull_area > 0:
                scale = 1.0 / np.sqrt(hull_area)
                points *= scale
                
            step_size *= 0.99
            
        score = get_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    return best_points
# EVOLVE-BLOCK-END
