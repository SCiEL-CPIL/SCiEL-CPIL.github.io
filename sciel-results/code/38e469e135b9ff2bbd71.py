import numpy as np
from scipy.optimize import minimize
import time

N = 12
APOTHEM = np.sqrt(3) / 2.0
MIN_DIST_SQ = 3.0 # (sqrt(3))^2

def get_hex_vertices(center, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    angles = angle_rad + np.deg2rad(np.arange(0, 360, 60))
    verts = np.column_stack((np.cos(angles), np.sin(angles)))
    return verts + center

def get_outer_normals(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    angles = angle_rad + np.deg2rad(np.arange(0, 360, 60))
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    return normals

def objective(vars):
    centers = vars[:2*N].reshape(N, 2)
    C = vars[2*N:2*N+2]
    beta = vars[2*N+2]
    
    normals = get_outer_normals(beta)
    
    max_proj = 0.0
    for i in range(N):
        verts = get_hex_vertices(centers[i], 0.0)
        diffs = verts - C
        projs = diffs @ normals.T
        current_max = np.max(projs)
        if current_max > max_proj:
            max_proj = current_max
            
    return max_proj / APOTHEM

def constraints(vars):
    centers = vars[:2*N].reshape(N, 2)
    cons = []
    for i in range(N):
        for j in range(i+1, N):
            d2 = np.sum((centers[i] - centers[j])**2)
            cons.append(d2 - MIN_DIST_SQ)
    return cons

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_S = float('inf')
    best_vars = None
    
    guesses = []
    
    d = np.sqrt(3)
    e1 = np.array([d, 0])
    e2 = np.array([d/2, 1.5])
    
    # Grid 3x4
    centers = []
    for i in range(3):
        for j in range(4):
            centers.append(i*e1 + j*e2)
    centers = np.array(centers)
    centers -= np.mean(centers, axis=0)
    guesses.append(np.concatenate([centers.flatten(), [0.0, 0.0, 0.0]]))
    
    # Grid 2x6
    centers = []
    for i in range(2):
        for j in range(6):
            centers.append(i*e1 + j*e2)
    centers = np.array(centers)
    centers -= np.mean(centers, axis=0)
    guesses.append(np.concatenate([centers.flatten(), [0.0, 0.0, 0.0]]))
    
    # Hex Cluster 7 + 5
    c0 = np.array([0,0])
    neighbors = []
    for k in range(6):
        angle = np.deg2rad(k*60)
        neighbors.append(d * np.array([np.cos(angle), np.sin(angle)]))
    second_ring = []
    for k in range(6):
        angle = np.deg2rad(k*60 + 30)
        second_ring.append(2*d * np.array([np.cos(angle), np.sin(angle)]))
    centers = [c0] + neighbors + second_ring[:5]
    centers = np.array(centers)
    centers -= np.mean(centers, axis=0)
    guesses.append(np.concatenate([centers.flatten(), [0.0, 0.0, 0.0]]))
    
    # Random Restarts
    np.random.seed(42)
    for _ in range(5):
        centers = np.random.rand(N, 2) * 10 - 5
        for _ in range(100):
            for i in range(N):
                for j in range(i+1, N):
                    diff = centers[i] - centers[j]
                    dist = np.linalg.norm(diff)
                    if dist < d and dist > 1e-6:
                        force = (d - dist) / dist * 0.5
                        centers[i] += force * diff
                        centers[j] -= force * diff
        centers -= np.mean(centers, axis=0)
        guesses.append(np.concatenate([centers.flatten(), [0.0, 0.0, 0.0]]))

    for guess in guesses:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, guess, method='SLSQP', 
                           constraints={'type': 'ineq', 'fun': constraints},
                           options={'maxiter': 500, 'ftol': 1e-8})
            
            if res.success:
                S = res.fun
                if S < best_S:
                    best_S = S
                    best_vars = res.x
        except Exception:
            pass

    if best_vars is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    centers = best_vars[:2*N].reshape(N, 2)
    C = best_vars[2*N:2*N+2]
    beta = best_vars[2*N+2]
    
    inner = [[centers[i][0], centers[i][1], 0.0] for i in range(N)]
    
    return inner, (C[0], C[1]), best_S, beta
