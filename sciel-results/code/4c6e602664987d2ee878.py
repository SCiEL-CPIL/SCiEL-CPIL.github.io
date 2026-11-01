import numpy as np
from scipy.optimize import differential_evolution
import time

N = 14
D = 3

def construct():
    start_time = time.time()
    time_limit = 140.0
    
    def unpack(params):
        P = np.zeros((N, D))
        P[0] = [0, 0, 0]
        P[1] = [1, 0, 0]
        P[2:] = params.reshape((N-2, D))
        return P

    def objective(params):
        P = unpack(params)
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, 1e10)
        d_min = np.min(d2)
        d_max = np.max(d2)
        if d_min < 1e-9:
            return 1e10
        return d_max / d_min

    bounds = [(-5, 5)] * (N-2)*D
    best_result = None
    best_val = 1e10

    # Run 1: Random init
    try:
        res1 = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=10, 
            tol=0.01, 
            polish=True, 
            seed=1
        )
        if res1.fun < best_val:
            best_val = res1.fun
            best_result = res1.x
    except Exception:
        pass

    if time.time() - start_time > time_limit - 20:
        if best_result is not None:
            return unpack(best_result)
        else:
            # Fallback
            side = int(np.ceil(N ** (1 / D)))
            grid = list(np.meshgrid(*([range(side)]*D)))
            grid = np.array(grid).T.reshape(-1, D)[:N]
            return grid.astype(float)

    # Run 2: Icosahedron init
    try:
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
            [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        # Add 2 points: let's try (0,0,0) is P0, so we need 13 more.
        # We have 12 ico points. Add 2 more.
        # Let's add (0, 0, 2*phi) and (0, 0, -2*phi)? No, too far.
        # Let's just use the 12 ico points and 2 random points near center.
        pts = np.vstack([ico, np.array([[0,0,0.1], [0,0,-0.1]])])
        pts = pts - np.mean(pts, axis=0)
        
        # Map to reduced coords
        p0_idx = 0
        p0 = pts[p0_idx]
        pts_shifted = pts - p0
        
        p1_idx = 1
        p1 = pts_shifted[p1_idx]
        norm_p1 = np.linalg.norm(p1)
        if norm_p1 < 1e-9:
             # Handle degenerate case
             p1_idx = 2
             p1 = pts_shifted[p1_idx]
             norm_p1 = np.linalg.norm(p1)
        
        p1_norm = p1 / norm_p1
        v = p1_norm
        u = np.array([1, 0, 0])
        c = np.dot(v, u)
        s = np.linalg.norm(np.cross(v, u))
        if s < 1e-9:
            R = np.eye(3) if c > 0 else -np.eye(3)
        else:
            K = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R = np.eye(3) + K + K @ K * (1-c)/(s*s)
            
        pts_rot = pts_shifted @ R.T
        pts_rot = pts_rot / norm_p1
        
        # Ensure P0 is 0 and P1 is 1,0,0
        # P0 is pts_rot[p0_idx] which is 0.
        # P1 is pts_rot[p1_idx] which is 1,0,0.
        # We need to reorder so that index 0 is P0 and index 1 is P1.
        # Current indices: p0_idx, p1_idx, others.
        # Let's just extract all points and reorder.
        # Actually, unpack expects P[0] and P[1] to be fixed.
        # So we need to construct the parameter vector from pts_rot[2:]?
        # No, pts_rot contains all 14 points.
        # We need to identify which point is P0 and P1 in the final array.
        # We defined P0 = pts[p0_idx] and P1 = pts[p1_idx].
        # So in the final array, P[0] should be the point that was P0, and P[1] should be P1.
        # The other 12 points are the parameters.
        
        # Reorder pts_rot so that p0_idx is 0 and p1_idx is 1.
        order = [p0_idx, p1_idx] + [i for i in range(N) if i not in [p0_idx, p1_idx]]
        pts_final = pts_rot[order]
        
        # Now pts_final[0] is 0,0,0 and pts_final[1] is 1,0,0.
        params_ico = pts_final[2:].flatten()
        
        res2 = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=0.01, 
            polish=True, 
            seed=2,
            x0=params_ico
        )
        
        if res2.fun < best_val:
            best_val = res2.fun
            best_result = res2.x
    except Exception:
        pass

    if best_result is not None:
        return unpack(best_result)
    else:
        side = int(np.ceil(N ** (1 / D)))
        grid = list(np.meshgrid(*([range(side)]*D)))
        grid = np.array(grid).T.reshape(-1, D)[:N]
        return grid.astype(float)

# EVOLVE-BLOCK-END
