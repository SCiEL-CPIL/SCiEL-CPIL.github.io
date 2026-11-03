import numpy as np
from scipy.optimize import minimize
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Base vertices of a unit regular hexagon centered at (0,0)
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    base_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Normals for the outer hexagon (angle 0)
    n_angles = np.linspace(np.pi/6, 2*np.pi + np.pi/6, 7)[:-1]
    normals = np.column_stack((np.cos(n_angles), np.sin(n_angles)))
    
    def get_vertices(pos, ang):
        c = np.cos(ang)
        s = np.sin(ang)
        R = np.zeros((N, 2, 2))
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c
        
        v_rot = np.einsum('nrc,kc->nkr', R, base_verts).transpose(0, 2, 1)
        return pos[:, np.newaxis, :] + v_rot

    def objective(params):
        pos = params[:2*N].reshape(N, 2)
        ang = params[2*N:3*N]
        
        verts = get_vertices(pos, ang)
        
        dots = np.einsum('nij,jk->nik', verts, normals.T)
        max_dots = np.max(dots, axis=2)
        max_per_hex = np.max(max_dots, axis=1)
        global_max = np.max(max_per_hex)
        
        S_req = (2.0 / np.sqrt(3.0)) * global_max
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        d_min = np.sqrt(3.0)
        violations = np.maximum(d_min - dists, 0.0)
        penalty = 100.0 * np.sum(violations**2)
        
        return S_req + penalty

    inits = []
    np.random.seed(42)
    for _ in range(5):
        # Random points in a hexagon of side 3
        # Sample in square and reject? Or use polar?
        # Just sample in a circle of radius 3
        r = np.sqrt(np.random.rand(N)) * 3.0
        theta = np.random.rand(N) * 2 * np.pi
        xs = r * np.cos(theta)
        ys = r * np.sin(theta)
        
        init = []
        for i in range(N):
            init.append(xs[i])
            init.append(ys[i])
            init.append(0.0)
        inits.append(np.array(init))
        
    best_result = None
    best_score = float('inf')
    
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, init, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            final_score = objective(res.x)
            
            pos = res.x[:2*N].reshape(N, 2)
            ang = res.x[2*N:3*N]
            verts = get_vertices(pos, ang)
            
            dots = np.einsum('nij,jk->nik', verts, normals.T)
            max_dots = np.max(dots, axis=2)
            max_per_hex = np.max(max_dots, axis=1)
            global_max = np.max(max_per_hex)
            S_final = (2.0 / np.sqrt(3.0)) * global_max
            
            # Check validity
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            if np.min(dists) > np.sqrt(3.0) - 1e-4:
                if S_final < best_score:
                    best_score = S_final
                    best_result = res.x
        except Exception:
            pass

    if best_result is None:
        # Fallback
        best_result = inits[0]
        pos = best_result[:2*N].reshape(N, 2)
        ang = best_result[2*N:3*N]
        verts = get_vertices(pos, ang)
        dots = np.einsum('nij,jk->nik', verts, normals.T)
        max_dots = np.max(dots, axis=2)
        max_per_hex = np.max(max_dots, axis=1)
        global_max = np.max(max_per_hex)
        best_score = (2.0 / np.sqrt(3.0)) * global_max

    inner = []
    for i in range(N):
        x = best_result[2*i]
        y = best_result[2*i+1]
        a = best_result[2*N+i]
        a_deg = np.degrees(a) % 360
        inner.append([x, y, a_deg])
        
    center = [0.0, 0.0]
    side = best_score
    angle = 0.0
    
    return inner, center, side, angle
