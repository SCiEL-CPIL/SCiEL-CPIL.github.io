import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

def construct():
    # Outer hexagon fixed at center (0,0) with rotation 0
    # Normals to edges are at 30, 90, ... degrees
    normals = np.array([[np.cos(np.pi/6 + k*np.pi/3), np.sin(np.pi/6 + k*np.pi/3)] for k in range(6)])
    
    def get_all_vertices(params):
        # params: (12, 3) -> (x, y, theta)
        # Returns (12, 6, 2)
        x = params[:, 0]
        y = params[:, 1]
        th = params[:, 2]
        
        # Angles for vertices
        # (12, 6)
        angles = th[:, None] + np.arange(6)[None, :] * np.pi / 3
        
        cx = x[:, None]
        cy = y[:, None]
        
        vx = cx + np.cos(angles)
        vy = cy + np.sin(angles)
        
        return np.stack([vx, vy], axis=-1)

    def objective(params_flat):
        if np.any(np.isnan(params_flat)):
            return 1e10
            
        params = params_flat.reshape(N, 3)
        verts = get_all_vertices(params) # (12, 6, 2)
        
        # 1. Calculate S_req
        # Project all vertices onto normals
        # verts: (12, 6, 2)
        # normals: (6, 2)
        # We want max dot product.
        # dot: (12, 6, 6)
        projs = np.einsum('ijk,kl->ijl', verts, normals)
        max_proj = np.max(projs)
        S_req = max_proj * 2.0 / np.sqrt(3)
        
        # 2. Overlap Penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                vi = verts[i] # (6, 2)
                vj = verts[j] # (6, 2)
                
                # Check vi inside vj
                # For each point in vi, check against vj
                # vj edges: vj[1:]-vj[:-1] and vj[0]-vj[-1]
                ej = np.roll(vj, -1, axis=0) - vj
                
                # Vector from vj[k] to vi[m]
                # vi: (6, 1, 2), vj: (1, 6, 2)
                # diff: (6, 6, 2)
                diff = vi[:, None, :] - vj[None, :, :]
                
                # Cross product
                # ej: (6, 2) -> (1, 6, 2)
                # diff: (6, 6, 2)
                # cross = ej[0]*diff[1] - ej[1]*diff[0]
                cross = np.einsum('kl,mkl->ml', ej, diff) 
                # Wait, ej is (6,2). diff is (6,6,2).
                # We need cross of ej[k] and diff[m, k].
                # Result should be (6, 6) where row m is point, col k is edge.
                # If any cross[m, k] < 0, point m is outside (assuming CCW).
                
                # Actually, let's just loop the 6 points of vi. It's small.
                # Vectorized approach above might have index confusion.
                # Let's stick to simple loops for clarity and correctness, 
                # but optimize the inner part if needed.
                
                for m in range(6):
                    p = vi[m]
                    # Check if p inside vj
                    # diff = p - vj (6, 2)
                    dp = p[None, :] - vj
                    # cross = ej x dp
                    # ej: (6,2), dp: (6,2)
                    cross_vals = ej[:, 0] * dp[:, 1] - ej[:, 1] * dp[:, 0]
                    if np.any(cross_vals < 0):
                        # Outside, no penalty for this point
                        continue
                    else:
                        # Inside!
                        # Calculate distance to nearest edge
                        # dist = min norm(dp[:, k] - proj)
                        # Simpler: since it's inside, distance is negative.
                        # Penalty += 100
                        penalty += 100.0
                        # We can break early if we just want a count, 
                        # but quadratic is better.
                        # Let's just add a fixed penalty for now for speed.
                        # Or calculate min dist.
                        
                        min_dist = float('inf')
                        for k in range(6):
                            # dist from p to edge vj[k]-vj[k+1]
                            # v1 = vj[k], v2 = vj[(k+1)%6]
                            v1 = vj[k]
                            v2 = vj[(k+1)%6]
                            edge = v2 - v1
                            vec = p - v1
                            edge_len_sq = np.dot(edge, edge)
                            if edge_len_sq == 0: continue
                            t = np.dot(vec, edge) / edge_len_sq
                            t = np.clip(t, 0, 1)
                            proj = v1 + t * edge
                            dist = np.linalg.norm(p - proj)
                            if dist < min_dist:
                                min_dist = dist
                        penalty += min_dist * min_dist * 100
                        
                # Check vj inside vi
                ei = np.roll(vi, -1, axis=0) - vi
                for m in range(6):
                    p = vj[m]
                    dp = p[None, :] - vi
                    cross_vals = ei[:, 0] * dp[:, 1] - ei[:, 1] * dp[:, 0]
                    if np.any(cross_vals < 0):
                        continue
                    else:
                        penalty += 100.0
                        min_dist = float('inf')
                        for k in range(6):
                            v1 = vi[k]
                            v2 = vi[(k+1)%6]
                            edge = v2 - v1
                            vec = p - v1
                            edge_len_sq = np.dot(edge, edge)
                            if edge_len_sq == 0: continue
                            t = np.dot(vec, edge) / edge_len_sq
                            t = np.clip(t, 0, 1)
                            proj = v1 + t * edge
                            dist = np.linalg.norm(p - proj)
                            if dist < min_dist:
                                min_dist = dist
                        penalty += min_dist * min_dist * 100

        return S_req + penalty

    bounds = []
    for i in range(N):
        bounds.append((-5, 5))
        bounds.append((-5, 5))
        bounds.append((0, 2*np.pi))

    # Initial guess
    # Place in a circle
    x0 = []
    for i in range(N):
        angle = i * 2 * np.pi / N
        x0.append([2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0])
    x0 = np.array(x0).flatten()

    result = differential_evolution(objective, bounds, seed=42, maxiter=500, tol=1e-6, polish=False, init='sobol')
    result2 = minimize(objective, result.x, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_params = result2.x.reshape(N, 3)
    
    inner = []
    for i in range(N):
        x = best_params[i, 0]
        y = best_params[i, 1]
        th_rad = best_params[i, 2]
        th_deg = np.degrees(th_rad) % 360
        inner.append([x, y, th_deg])
        
    # Recalculate final S
    verts = get_all_vertices(best_params)
    projs = np.einsum('ijk,kl->ijl', verts, normals)
    max_proj = np.max(projs)
    final_S = max_proj * 2.0 / np.sqrt(3)
    
    return inner, (0.0, 0.0), final_S, 0.0

# EVOLVE-BLOCK-END
