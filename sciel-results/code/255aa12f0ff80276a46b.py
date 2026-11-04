import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    # 1. Define the graph of 12 hexagons (3x4 grid in axial coordinates)
    nodes = []
    for r in range(3):
        for q in range(4):
            nodes.append((q, r))
    
    n_nodes = len(nodes)
    A = np.zeros((n_nodes, n_nodes))
    for i, (q1, r1) in enumerate(nodes):
        for j, (q2, r2) in enumerate(nodes):
            if i == j: continue
            dq = q2 - q1
            dr = r2 - r1
            if (dq, dr) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
                A[i, j] = 1.0
                A[j, i] = 1.0
                
    # 2. Spectral Embedding
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    vals, vecs = np.linalg.eigh(L)
    
    v1 = vecs[:, 1]
    v2 = vecs[:, 2]
    P_init = np.column_stack((v1, v2))
    
    # Scale to edge length sqrt(3)
    dists = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if A[i, j] > 0:
                dists.append(np.linalg.norm(P_init[i] - P_init[j]))
    if len(dists) > 0 and np.mean(dists) > 0:
        scale_factor = np.sqrt(3) / np.mean(dists)
        P_init = P_init * scale_factor
    
    P_init = P_init - np.mean(P_init, axis=0)
    
    # 3. Optimization
    def get_hex_vertices(center, angle_deg, side=1.0):
        angle_rad = np.deg2rad(angle_deg)
        vertices = []
        for k in range(6):
            a = angle_rad + k * np.pi / 3
            vertices.append([center[0] + side * np.cos(a), center[1] + side * np.sin(a)])
        return np.array(vertices)

    def objective(params):
        Xc, Yc, Phi, Theta = params
        Phi_rad = np.deg2rad(Phi)
        Theta_rad = np.deg2rad(Theta)
        
        R_theta = np.array([[np.cos(Theta_rad), -np.sin(Theta_rad)],
                            [np.sin(Theta_rad), np.cos(Theta_rad)]])
        P_rot = P_init @ R_theta.T
        
        normals = []
        for k in range(6):
            a = Phi_rad + np.pi/6 + k * np.pi / 3
            normals.append([np.cos(a), np.sin(a)])
        normals = np.array(normals)
        
        max_proj = 0.0
        for i in range(n_nodes):
            center = P_rot[i]
            vertices = get_hex_vertices(center, 0.0)
            for v in vertices:
                for n in normals:
                    proj = np.dot(v - np.array([Xc, Yc]), n)
                    if proj > max_proj:
                        max_proj = proj
                    if -proj > max_proj:
                        max_proj = -proj
                        
        S = 2.0 * max_proj / np.sqrt(3)
        return S

    X0 = [0.0, 0.0, 0.0, 0.0]
    try:
        res = minimize(objective, X0, method='Nelder-Mead', options={'maxfevs': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
        Xc, Yc, Phi, Theta = res.x
        S = res.fun
    except Exception:
        # Fallback
        Xc, Yc, Phi, Theta = 0.0, 0.0, 0.0, 0.0
        S = 5.0

    Theta_rad = np.deg2rad(Theta)
    R_theta = np.array([[np.cos(Theta_rad), -np.sin(Theta_rad)],
                        [np.sin(Theta_rad), np.cos(Theta_rad)]])
    P_final = P_init @ R_theta.T
    
    inner = []
    for i in range(n_nodes):
        inner.append([float(P_final[i][0]), float(P_final[i][1]), 0.0])
        
    return inner, (float(Xc), float(Yc)), float(S), float(Phi)

# EVOLVE-BLOCK-END
