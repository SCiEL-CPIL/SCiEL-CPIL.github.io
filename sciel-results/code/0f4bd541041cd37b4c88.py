import numpy as np
import math
from scipy.optimize import minimize

N = 12

def construct():
    candidates = []
    
    # Packing 1: 3x4 Grid
    u = np.array([np.sqrt(3), 0])
    v = np.array([np.sqrt(3)/2, 1.5])
    centers = []
    for i in range(3):
        for j in range(4):
            centers.append(i*u + j*v)
    candidates.append(np.array(centers))
    
    # Packing 2: Compact shape
    shape_2 = [
        (0,0), (1,0),
        (0,1), (1,1), (2,1),
        (0,2), (1,2), (2,2),
        (0,3), (1,3), (2,3),
        (1,4)
    ]
    centers = []
    for i,j in shape_2:
        x = (i + j/2.0) * np.sqrt(3)
        y = j * 1.5
        centers.append(np.array([x,y]))
    candidates.append(np.array(centers))
    
    best_S = float('inf')
    best_config = None
    
    # Precompute vertices for inner hexagon
    verts = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    for centers in candidates:
        centroid = np.mean(centers, axis=0)
        centers = centers - centroid
        
        # Scan phi
        for phi_deg in np.linspace(0, 30, 30):
            phi = np.radians(phi_deg)
            
            # Objective function to minimize S
            # Variables: [Tx, Ty, S]
            def obj(vars):
                Tx, Ty, S = vars
                T = np.array([Tx, Ty])
                # Sample alphas
                alphas = np.linspace(0, 2*np.pi, 180, endpoint=False)
                n_vectors = np.stack([np.cos(alphas), np.sin(alphas)], axis=1)
                
                # A(alpha) = max_k (C_k.n) + h_inner
                projections = centers @ n_vectors.T
                max_proj = np.max(projections, axis=0)
                
                dot_verts = verts @ n_vectors.T
                h_inner = np.max(dot_verts, axis=0)
                
                A = max_proj + h_inner
                
                # h_outer(alpha + phi)
                c_phi = np.cos(phi)
                s_phi = np.sin(phi)
                R = np.array([[c_phi, -s_phi], [s_phi, c_phi]])
                verts_rot = verts @ R.T
                dot_verts_rot = verts_rot @ n_vectors.T
                h_outer = np.max(dot_verts_rot, axis=0)
                
                # Constraint: A + T.n <= S * h_outer
                # Violation = max(0, (A + T.n) - S * h_outer)
                val = A + T @ n_vectors.T.T
                diff = val - S * h_outer
                return np.max(diff)
            
            # Initial guess
            # S should be at least max extent
            # T should be 0
            x0 = [0, 0, 5.0]
            
            # Minimize S such that obj <= 0
            # This is a feasibility problem.
            # We can minimize S + penalty * max(0, obj)
            # Or just minimize S subject to obj <= 0.
            
            def obj_minimize(vars):
                Tx, Ty, S = vars
                T = np.array([Tx, Ty])
                alphas = np.linspace(0, 2*np.pi, 180, endpoint=False)
                n_vectors = np.stack([np.cos(alphas), np.sin(alphas)], axis=1)
                
                projections = centers @ n_vectors.T
                max_proj = np.max(projections, axis=0)
                
                dot_verts = verts @ n_vectors.T
                h_inner = np.max(dot_verts, axis=0)
                
                A = max_proj + h_inner
                
                c_phi = np.cos(phi)
                s_phi = np.sin(phi)
                R = np.array([[c_phi, -s_phi], [s_phi, c_phi]])
                verts_rot = verts @ R.T
                dot_verts_rot = verts_rot @ n_vectors.T
                h_outer = np.max(dot_verts_rot, axis=0)
                
                val = A + T @ n_vectors.T.T
                diff = val - S * h_outer
                # Penalize violations
                penalty = 100.0
                return S + penalty * np.sum(np.maximum(0, diff))
            
            res = minimize(obj_minimize, x0, method='Nelder-Mead')
            
            if res.fun < best_S:
                # Check validity
                Tx, Ty, S = res.x
                # Verify
                alphas = np.linspace(0, 2*np.pi, 360, endpoint=False)
                n_vectors = np.stack([np.cos(alphas), np.sin(alphas)], axis=1)
                T = np.array([Tx, Ty])
                projections = centers @ n_vectors.T
                max_proj = np.max(projections, axis=0)
                dot_verts = verts @ n_vectors.T
                h_inner = np.max(dot_verts, axis=0)
                A = max_proj + h_inner
                c_phi = np.cos(phi)
                s_phi = np.sin(phi)
                R = np.array([[c_phi, -s_phi], [s_phi, c_phi]])
                verts_rot = verts @ R.T
                dot_verts_rot = verts_rot @ n_vectors.T
                h_outer = np.max(dot_verts_rot, axis=0)
                val = A + T @ n_vectors.T.T
                if np.max(val - S * h_outer) < 1e-5:
                    best_S = S
                    best_T = T
                    best_phi = phi
                    best_centers = centers
    
    if best_S < float('inf'):
        inner = []
        for c in best_centers + best_T:
            inner.append([c[0], c[1], 0.0])
        return inner, (0.0, 0.0), best_S, best_phi * 180.0 / np.pi
    else:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
