import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hex_support_6(alpha, phis):
    """Support of unit hexagon with orientation alpha in directions phis."""
    angles = alpha + np.arange(6) * np.pi / 3
    diff = phis[:, None] - angles[None, :]
    return np.max(np.cos(diff), axis=1)

def enclosing_side_fast(centers, orientations, cx, cy, beta):
    base_dirs = np.arange(6) * np.pi / 3
    phis = np.concatenate([base_dirs, base_dirs + np.pi/6])
    outer_sup = hex_support_6(beta, phis)
    max_s = 0.0
    for i in range(len(centers)):
        inner_sup = hex_support_6(orientations[i], phis)
        dx = centers[i][0] - cx
        dy = centers[i][1] - cy
        proj = dx * np.cos(phis) + dy * np.sin(phis)
        s_candidate = np.max((inner_sup + proj) / outer_sup)
        max_s = max(max_s, s_candidate)
    return max_s

def overlap_penalty(centers, orientations, n_samples=60):
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            diff = centers[j] - centers[i]
            dist = np.linalg.norm(diff)
            if dist < 3.5:
                phis = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
                s1 = hex_support_6(orientations[i], phis)
                s2 = hex_support_6(orientations[j], -phis)
                proj = diff[0]*np.cos(phis) + diff[1]*np.sin(phis)
                max_ov = np.max(s1 + s2 - proj)
                if max_ov > 0:
                    penalty += max_ov * 100
    return penalty

def objective(params):
    cx, cy, beta = params[0], params[1], params[2]
    centers = params[3:3+2*N].reshape(N, 2)
    orientations = params[3+2*N:3+3*N]
    s = enclosing_side_fast(centers, orientations, cx, cy, beta)
    pen = overlap_penalty(centers, orientations)
    return s + pen

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    best_score = float('inf')
    best_x = None
    
    # Generate diverse initial configs
    configs = []
    
    # Hexagonal cluster
    c = np.array([[0.0, 0.0]])
    for k in range(6):
        a = np.pi/3 * k
        c = np.vstack([c, [[2*np.cos(a), 2*np.sin(a)]]])
    for k in range(4):
        a = np.pi/3 * k + np.pi/6
        c = np.vstack([c, [[3.5*np.cos(a), 3.5*np.sin(a)]]])
    configs.append((c, np.zeros(N)))
    
    # Rotated hexagonal
    for rot in [np.pi/6, np.pi/3, np.pi/12]:
        R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        configs.append((c @ R.T, np.full(N, rot)))
    
    # Linear
    cl = np.array([[2.0*k, 0.0] for k in range(N)])
    configs.append((cl, np.zeros(N)))
    
    # 2x6 grid-like
    cg = []
    for i in range(6):
        cg.append([2.0*i, 0.0])
        cg.append([2.0*i + 1.0, np.sqrt(3)])
    cg = np.array(cg[:N])
    configs.append((cg, np.zeros(N)))
    
    for centers_init, orientations_init in configs:
        if time.time() > deadline - 20:
            break
        x0 = np.concatenate([[0.0, 0.0, 0.0], centers_init.flatten(), orientations_init])
        
        for attempt in range(4):
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 400, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
                if res.fun < best_score:
                    best_score = res.fun
                    best_x = res.x.copy()
                x0 = res.x + np.random.normal(0, 0.05, size=res.x.shape)
            except:
                break
    
    if best_x is None:
        inner = [[2.0*k, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), 20.0, 0.0
    
    cx, cy, beta = best_x[0], best_x[1], best_x[2]
    centers = best_x[3:3+2*N].reshape(N, 2)
    orientations = best_x[3+2*N:3+3*N]
    s = enclosing_side_fast(centers, orientations, cx, cy, beta)
    
    inner = [[float(centers[i][0]), float(centers[i][1]), float(np.degrees(orientations[i]))] for i in range(N)]
    return inner, (float(cx), float(cy)), float(s), float(np.degrees(beta))
