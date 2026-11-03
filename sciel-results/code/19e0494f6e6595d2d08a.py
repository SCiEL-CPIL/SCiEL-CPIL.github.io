import math
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def axial_distance(p1, p2):
        q1, r1 = p1
        q2, r2 = p2
        return 0.5 * (abs(q1 - q2) + abs(r1 - r2) + abs((q1 + r1) - (q2 + r2)))

    def is_valid_axial(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if axial_distance(points[i], points[j]) < 1.0 - 1e-9:
                    return False
        return True

    candidates = []
    
    # Structured
    axial_1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (-1,2), (-2,1), (2,-1)]
    axial_2 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (0,-2)]
    axial_3 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (-1,1), (3,1)]
    axial_4 = [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2)]
    axial_5 = [(0,0), (1,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (0,3), (1,3), (2,3)]
    axial_6 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (0,3), (1,3)]
    axial_7 = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (2,0), (-2,0), (0,2), (0,-2)]
    axial_8 = [(0,0), (1,0), (2,0), (3,0), (4,0), (0,1), (1,1), (2,1), (3,1), (4,1), (5,1)]
    axial_9 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,1), (-2,2), (-1,3), (1,-2)]
    axial_10 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (1,3), (2,3)]
    axial_11 = [(0,0), (1,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (3,2), (1,3)]
    axial_12 = [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (0,2), (1,2), (0,3), (1,3), (0,4)]
    axial_13 = [(0,0), (1,0), (2,0), (1,1), (2,1), (0,2), (1,2), (2,2), (-1,1), (3,1), (0,-1)]
    axial_14 = [(0,0), (1,0), (0,1), (1,1), (2,1), (1,2), (2,2), (0,3), (1,3), (2,3), (3,2)]
    
    for a in [axial_1, axial_2, axial_3, axial_4, axial_5, axial_6, axial_7, axial_8, axial_9, axial_10, axial_11, axial_12, axial_13, axial_14]:
        if is_valid_axial(a):
            candidates.append(a)

    # Random
    def generate_random_candidate(seed):
        np.random.seed(seed)
        points = [(0, 0)]
        attempts = 0
        while len(points) < N and attempts < 2000:
            attempts += 1
            base = points[np.random.randint(len(points))]
            neighbors = [(base[0]+1, base[1]), (base[0]-1, base[1]), 
                         (base[0], base[1]+1), (base[0], base[1]-1),
                         (base[0]+1, base[1]-1), (base[0]-1, base[1]+1)]
            np.random.shuffle(neighbors)
            for n in neighbors:
                valid = True
                for p in points:
                    if axial_distance(n, p) < 1.0 - 1e-9:
                        valid = False
                        break
                if valid:
                    points.append(n)
                    break
        if len(points) == N:
            return points
        return None

    for i in range(100):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        c = generate_random_candidate(i)
        if c:
            candidates.append(c)

    def calc_S_and_center(pts, beta):
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_projs = np.zeros(6)
        min_projs = np.zeros(6)
        
        for k, (nx, ny) in enumerate(normals):
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            max_projs[k] = np.max(projs)
            min_projs[k] = np.min(projs)
        
        widths = max_projs - min_projs
        S = (np.max(widths) + 2 * APOTHEM) / (2 * APOTHEM)
        
        U = np.array(normals)
        M = (max_projs + min_projs) / 2.0
        try:
            C = np.linalg.lstsq(U, M, rcond=None)[0]
        except:
            C = np.array([0.0, 0.0])
            
        return S, C

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        
        def obj(b):
            S, _ = calc_S_and_center(pts, b)
            return S
            
        try:
            res = minimize_scalar(obj, bounds=(0, math.pi / 6), method='bounded', options={'maxiter': 200, 'xatol': 1e-12})
            if res.fun < best_global_S:
                best_global_S = res.fun
                best_global_beta = res.x
                _, best_global_center = calc_S_and_center(pts, best_global_beta)
                best_global_pts = pts
        except Exception:
            pass

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_beta = 0.0
        best_global_S, best_global_center = calc_S_and_center(best_global_pts, best_global_beta)

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
