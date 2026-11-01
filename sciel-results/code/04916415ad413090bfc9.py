import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + angle_rad
    return np.array([[cx + side * np.cos(a), cy + side * np.sin(a)] for a in angles])

def polygons_overlap(v1, v2, eps=1e-9):
    polys = [v1, v2]
    for poly in polys:
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            if proj1.max() < proj2.min() - eps or proj2.max() < proj1.min() - eps:
                return False
    return True

def min_outer_side(vertices_all, outer_angle_deg=0.0):
    angle_rad = np.radians(outer_angle_deg)
    normal_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + angle_rad
    normals = np.array([[np.cos(a), np.sin(a)] for a in normal_angles])
    max_proj = 0.0
    for normal in normals:
        proj = np.dot(vertices_all, normal)
        max_proj = max(max_proj, float(proj.max()))
    return 2.0 * max_proj / np.sqrt(3.0)

def objective(params):
    positions = params[:2*N].reshape(N, 2)
    inner_angles = params[2*N:3*N]
    outer_angle = params[3*N]
    
    all_vertices = []
    hex_verts = []
    for i in range(N):
        v = hex_vertices(positions[i, 0], positions[i, 1], 1.0, inner_angles[i])
        hex_verts.append(v)
        all_vertices.append(v)
    all_vertices = np.vstack(all_vertices)
    
    S = min_outer_side(all_vertices, outer_angle)
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if polygons_overlap(hex_verts[i], hex_verts[j]):
                penalty += 1000.0
    
    return S + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    best_params = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    for trial in range(8):
        if time.time() > deadline:
            break
        
        if trial == 0:
            # 3x4 hexagonal grid
            params = []
            for row in range(4):
                for col in range(3):
                    x = (col - 1) * np.sqrt(3)
                    y = (row - 1.5) * 1.5
                    if row % 2 == 1:
                        x += np.sqrt(3) / 2
                    params.extend([x, y, 0.0])
            params.append(0.0)
            init = np.array(params)
        elif trial == 1:
            # 4x3 hexagonal grid (rotated)
            params = []
            for row in range(3):
                for col in range(4):
                    x = (col - 1.5) * np.sqrt(3)
                    y = (row - 1) * 1.5
                    if row % 2 == 1:
                        x += np.sqrt(3) / 2
                    params.extend([x, y, 0.0])
            params.append(0.0)
            init = np.array(params)
        else:
            # Random perturbation of best
            if best_params is not None:
                init = best_params + np.random.randn(3*N + 1) * 0.5
            else:
                init = np.random.randn(3*N + 1) * 2
        
        try:
            result = minimize(objective, init, method='L-BFGS-B', maxiter=300)
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is None or best_score > 50:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    positions = best_params[:2*N].reshape(N, 2)
    inner_angles = best_params[2*N:3*N]
    outer_angle = best_params[3*N]
    
    inner = [[float(positions[i, 0]), float(positions[i, 1]), float(inner_angles[i])] for i in range(N)]
    
    all_vertices = []
    for i in range(N):
        v = hex_vertices(positions[i, 0], positions[i, 1], 1.0, inner_angles[i])
        all_vertices.append(v)
    all_vertices = np.vstack(all_vertices)
    S = min_outer_side(all_vertices, outer_angle)
    
    return inner, (0.0, 0.0), float(S), float(outer_angle)
