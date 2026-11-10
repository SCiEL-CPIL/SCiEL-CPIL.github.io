import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = 580  # 20s safety margin
    
    def get_hex_vertices(cx, cy, s, angle_deg=0):
        angles = np.radians(angle_deg) + np.radians(60 * np.arange(6))
        return np.column_stack([
            cx + s * np.cos(angles),
            cy + s * np.sin(angles)
        ])
    
    def get_edge_normals(vertices):
        normals = []
        for i in range(6):
            p1 = vertices[i]
            p2 = vertices[(i+1) % 6]
            edge = p2 - p1
            normal = np.array([edge[1], -edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normal = normal / norm
            normals.append(normal)
        return normals
    
    def compute_penalty(positions, S):
        penalty = 0.0
        outer_vertices = get_hex_vertices(0, 0, S, 0)
        
        # Containment
        for i in range(11):
            inner_vertices = get_hex_vertices(positions[i, 0], positions[i, 1], 1, 0)
            for j in range(6):
                p1 = outer_vertices[j]
                p2 = outer_vertices[(j+1) % 6]
                edge = p2 - p1
                for v in inner_vertices:
                    cross = np.cross(edge, v - p1)
                    if cross < 0:
                        penalty += cross * cross
        
        # Separation
        for i in range(11):
            for j in range(i+1, 11):
                v1 = get_hex_vertices(positions[i, 0], positions[i, 1], 1, 0)
                v2 = get_hex_vertices(positions[j, 0], positions[j, 1], 1, 0)
                normals = get_edge_normals(v1) + get_edge_normals(v2)
                min_gap = -np.inf
                for normal in normals:
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    gap = max(proj2.min() - proj1.max(), proj1.min() - proj2.max())
                    min_gap = max(min_gap, gap)
                if min_gap < 1e-6:
                    penalty += (1e-6 - min_gap) ** 2
        
        return penalty
    
    def is_valid(positions, S):
        if not np.all(np.isfinite(positions)) or not np.isfinite(S):
            return False
        if S < np.sqrt(11):
            return False
        outer_vertices = get_hex_vertices(0, 0, S, 0)
        
        for i in range(11):
            inner_vertices = get_hex_vertices(positions[i, 0], positions[i, 1], 1, 0)
            for j in range(6):
                p1 = outer_vertices[j]
                p2 = outer_vertices[(j+1) % 6]
                edge = p2 - p1
                for v in inner_vertices:
                    cross = np.cross(edge, v - p1)
                    if cross < -1e-6:
                        return False
        
        for i in range(11):
            for j in range(i+1, 11):
                v1 = get_hex_vertices(positions[i, 0], positions[i, 1], 1, 0)
                v2 = get_hex_vertices(positions[j, 0], positions[j, 1], 1, 0)
                normals = get_edge_normals(v1) + get_edge_normals(v2)
                separated = False
                for normal in normals:
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    if proj1.max() < proj2.min() - 1e-6 or proj2.max() < proj1.min() - 1e-6:
                        separated = True
                        break
                if not separated:
                    return False
        
        return True
    
    def objective(vars):
        positions = vars[:22].reshape(11, 2)
        S = vars[22]
        if S < 1:
            return 1e10
        return S + 1000 * compute_penalty(positions, S)
    
    best_S = 8.0
    best_positions = np.array([
        [0, 0],
        [-2.5, 0], [2.5, 0],
        [-1.25, 2.17], [1.25, 2.17],
        [-1.25, -2.17], [1.25, -2.17],
        [-3.75, 2.17], [3.75, 2.17],
        [-3.75, -2.17], [3.75, -2.17],
    ])
    
    sqrt3 = np.sqrt(3)
    
    # Generate diverse initial configurations
    configs = [best_positions.copy()]
    
    # Hexagonal cluster
    hex7 = np.array([
        [0, 0],
        [sqrt3, 0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
    ])
    extra = np.array([
        [2*sqrt3, 0],
        [sqrt3, 3],
        [-sqrt3, 3],
        [-2*sqrt3, 0],
    ])
    configs.append(np.vstack([hex7, extra]))
    
    # Random configs
    np.random.seed(42)
    for _ in range(30):
        pos = np.random.randn(11, 2) * 3
        configs.append(pos)
    
    for config in configs:
        if time.time() - start_time > deadline:
            break
        
        init_vars = np.zeros(23)
        for i in range(11):
            init_vars[2*i] = config[i, 0]
            init_vars[2*i+1] = config[i, 1]
        init_vars[22] = 6.0
        
        result = minimize(
            objective,
            init_vars,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7}
        )
        
        positions = result.x[:22].reshape(11, 2)
        S = result.x[22]
        
        if is_valid(positions, S) and S < best_S:
            best_S = S
            best_positions = positions.copy()
    
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [best_positions[i, 0], best_positions[i, 1], 0]
    
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, best_S
