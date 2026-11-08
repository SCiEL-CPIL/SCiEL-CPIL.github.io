import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import time
import itertools

N = 14
TRI_INDICES = np.array(list(itertools.combinations(range(N), 3)))

def construct():
            start_time = time.time()
            timeout = 140
            
            # Pre-calculate indices
            i_idx, j_idx, k_idx = TRI_INDICES[:,0], TRI_INDICES[:,1], TRI_INDICES[:,2]
            
            def score_cartesian(pts):
                # pts: (N, 2)
                if np.any(~np.isfinite(pts)):
                    return -1e10
                
                # Hull
                try:
                    hull = ConvexHull(pts)
                    h_area = hull.volume
                except:
                    return -1e10
                
                if h_area < 1e-6:
                    return -1e10
                
                # Areas
                pi = pts[i_idx]
                pj = pts[j_idx]
                pk = pts[k_idx]
                v1 = pj - pi
                v2 = pk - pi
                cross = v1[:,0] * v2[:,1] - v1[:,1] * v2[:,0]
                min_a = np.min(np.abs(cross)) * 0.5
                
                return min_a / h_area

            def objective_de(params):
                r = params[:N]
                w = params[N:]
                
                # Normalize weights
                w_sum = np.sum(w)
                if w_sum < 1e-6:
                    return 1e10
                theta = np.cumsum(2 * np.pi * w / w_sum)
                theta = theta - theta[0] # Fix rotation
                
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                pts = np.column_stack([x, y])
                
                s = score_cartesian(pts)
                return -s

            # Bounds
            bounds = [(0.5, 1.5)] * N + [(0.5, 1.5)] * N
            
            # Initial guess
            x0 = np.ones(N)
            w0 = np.ones(N)
            init = np.concatenate([x0, w0])
            
            best_result = None
            best_score = -1e10
            
            # Run DE
            if time.time() - start_time < timeout - 20:
                try:
                    res = differential_evolution(
                        objective_de, 
                        bounds, 
                        maxiter=200, 
                        popsize=20, 
                        tol=1e-8,
                        seed=42,
                        init='latinhypercube'
                    )
                    if res.fun < best_score:
                        best_score = res.fun
                        best_result = res.x
                except Exception as e:
                    pass
            
            # Convert best_result to Cartesian
            if best_result is not None:
                r = best_result[:N]
                w = best_result[N:]
                w_sum = np.sum(w)
                theta = np.cumsum(2 * np.pi * w / w_sum)
                theta = theta - theta[0]
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                current_pts = np.column_stack([x, y])
            else:
                # Fallback to regular polygon
                angles = np.linspace(0, 2*np.pi, N, endpoint=False)
                current_pts = np.column_stack([np.cos(angles), np.sin(angles)])
            
            # Polish with Nelder-Mead
            if time.time() - start_time < timeout - 5:
                def obj_nm(pts_flat):
                    pts = pts_flat.reshape(N, 2)
                    s = score_cartesian(pts)
                    return -s
                
                try:
                    res_nm = minimize(
                        obj_nm,
                        current_pts.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    if res_nm.fun < best_score:
                        best_score = res_nm.fun
                        current_pts = res_nm.x.reshape(N, 2)
                except Exception:
                    pass
            
            return current_pts
