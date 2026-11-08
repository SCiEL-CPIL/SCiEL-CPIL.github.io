import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14
TRIS = list(itertools.combinations(range(N), 3))
TRI_INDICES = np.array(TRIS)

def calc_score(pts):
    # pts: (N, 2)
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return -1e10
    except Exception:
        return -1e10
        
    p_i = pts[TRI_INDICES[:, 0]]
    p_j = pts[TRI_INDICES[:, 1]]
    p_k = pts[TRI_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def objective(pts_flat):
    pts = pts_flat.reshape(N, 2)
    # Normalize to unit norm to handle scale invariance
    norm = np.linalg.norm(pts)
    if norm < 1e-10:
        return 1e10
    pts = pts / norm
    
    s = calc_score(pts)
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = -1e10
    best_pts = None
    
    inits = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Regular 13-gon + Center
    angles_13 = np.linspace(0, 2*np.pi, 13, endpoint=False)
    p13 = np.column_stack([np.cos(angles_13), np.sin(angles_13)])
    p13 = np.vstack([p13, [[0, 0]]])
    inits.append(p13)
    
    # 3. Regular 12-gon + 2 points
    angles_12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    p12 = np.column_stack([np.cos(angles_12), np.sin(angles_12)])
    p_extra = np.array([[0.5, 0], [-0.5, 0]])
    p12_extra = np.vstack([p12, p_extra])
    inits.append(p12_extra)
    
    # 4. Random
    rng = np.random.RandomState(42)
    inits.append(rng.rand(N, 2) - 0.5)
    
    # 5. Another Random
    rng = np.random.RandomState(123)
    inits.append(rng.rand(N, 2) - 0.5)
    
    # 6. Two concentric 7-gons
    angles_7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    p7_outer = np.column_stack([np.cos(angles_7), np.sin(angles_7)])
    p7_inner = 0.5 * np.column_stack([np.cos(angles_7 + np.pi/7), np.sin(angles_7 + np.pi/7)])
    inits.append(np.vstack([p7_outer, p7_inner]))

    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        x0 = init.flatten()
        
        try:
            # Use Powell for robustness against non-smoothness
            res = minimize(objective, x0, method='Powell', options={'maxiter': 200, 'ftol': 1e-6})
            
            if time.time() > deadline:
                break
                
            score = -res.fun
            if score > best_score:
                best_score = score
                best_pts = res.x.reshape(N, 2)
                
        except Exception:
            pass

    if best_pts is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_pts
