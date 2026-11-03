"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    def get_cluster_coords(topology):
        if topology == "3x4":
            coords = []
            for q in range(-1, 2):
                for r in range(-1, 3):
                    x = np.sqrt(3) * (q + r/2.0)
                    y = 1.5 * r
                    coords.append([x, y])
            return np.array(coords)
        elif topology == "blob":
            coords = [[0,0]]
            for i in range(6):
                ang = i * np.pi / 3
                x = np.sqrt(3) * np.cos(ang)
                y = np.sqrt(3) * np.sin(ang)
                coords.append([x, y])
            for i in range(5):
                ang = i * np.pi / 3
                x = 2 * np.sqrt(3) * np.cos(ang)
                y = 2 * np.sqrt(3) * np.sin(ang)
                coords.append([x, y])
            return np.array(coords)
        elif topology == "2x6":
            coords = []
            for q in range(0, 2):
                for r in range(0, 6):
                    x = np.sqrt(3) * (q + r/2.0)
                    y = 1.5 * r
                    coords.append([x, y])
            return np.array(coords)
        else:
            return np.array([[0,0]]*N)

    def compute_min_side(cluster_coords, alpha, tx, ty):
        v0 = np.array([(np.cos(i * np.pi / 3), np.sin(i * np.pi / 3)) for i in range(6)])
        c_a = np.cos(alpha)
        s_a = np.sin(alpha)
        R = np.array([[c_a, -s_a], [s_a, c_a]])
        v_rot = v0 @ R.T
        all_verts = cluster_coords[:, np.newaxis, :] + v_rot[np.newaxis, :, :]
        all_verts -= np.array([tx, ty])
        normals = np.array([(np.cos(i * np.pi / 3), np.sin(i * np.pi / 3)) for i in range(6)])
        proj = np.einsum('ijk,kl->ijl', all_verts, normals)
        max_proj = np.max(np.abs(proj), axis=(0, 1))
        a = np.max(max_proj)
        S = 2 * a / np.sqrt(3)
        return S

    def optimize_cluster(topology_name):
        coords = get_cluster_coords(topology_name)
        coords -= np.mean(coords, axis=0)
        best_S = float('inf')
        best_params = None
        for alpha in np.linspace(0, np.pi/3, 10):
            def objective(params):
                tx, ty = params
                return compute_min_side(coords, alpha, tx, ty)
            res = minimize(objective, [0, 0], method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 100})
            if res.fun < best_S:
                best_S = res.fun
                best_params = (alpha, res.x[0], res.x[1])
        return best_S, best_params, coords

    start_time = time.time()
    deadline = 110.0
    
    topologies = ["3x4", "blob", "2x6"]
    
    best_overall_S = float('inf')
    best_overall_params = None
    best_overall_coords = None
    
    for top in topologies:
        if time.time() - start_time > deadline:
            break
        S, params, coords = optimize_cluster(top)
        if S < best_overall_S:
            best_overall_S = S
            best_overall_params = params
            best_overall_coords = coords
            
    if best_overall_S == float('inf'):
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    alpha, tx, ty = best_overall_params
    coords = best_overall_coords
    S = best_overall_S
    
    inner = []
    for i in range(N):
        cx = coords[i, 0] + tx
        cy = coords[i, 1] + ty
        inner.append([cx, cy, alpha])
        
    center = [0.0, 0.0]
    side = S
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
