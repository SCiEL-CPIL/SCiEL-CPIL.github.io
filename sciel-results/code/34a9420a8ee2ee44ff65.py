import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin for safety

    def score_points(pts):
        """Compute (max_dist / min_dist)^2 for a set of points."""
        if len(pts) != N:
            return 1e10
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    # Step 1: Generate hexagonal lattice
    R = 6
    hex_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append((x, y))
    hex_pts = np.array(hex_pts)
    num_hex = len(hex_pts)
    
    # Step 2: Find good initial subset via greedy swap
    dists_from_origin = np.sum(hex_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    
    def score_subset(indices):
        sub_pts = hex_pts[indices]
        return score_points(sub_pts)
    
    current_indices = sorted_indices[:N].copy()
    current_score = score_subset(current_indices)
    
    pool_size = 120
    candidate_indices = sorted_indices[:pool_size]
    in_subset = set(current_indices)
    outside_candidates = [i for i in candidate_indices if i not in in_subset]
    
    # Greedy swap search
    improved = True
    while improved and time.time() < deadline - 60:
        improved = False
        for i_idx in range(N):
            if time.time() > deadline - 60:
                break
            for j in outside_candidates:
                if time.time() > deadline - 60:
                    break
                new_indices = current_indices.copy()
                new_indices[i_idx] = j
                new_score = score_subset(new_indices)
                if new_score < current_score:
                    current_score = new_score
                    current_indices = new_indices
                    in_subset = set(current_indices)
                    outside_candidates = [k for k in candidate_indices if k not in in_subset]
                    improved = True
                    break
            if improved:
                break
    
    best_pts = hex_pts[current_indices]
    best_score = current_score

    # Step 3: Normalize - fix two closest points to (0,0) and (1,0)
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, 1)
    d = dists[iu]
    min_idx = np.argmin(d)
    pair = np.unravel_index(min_idx, (N, N))
    i0, i1 = pair[0], pair[1]
    if i0 > i1:
        i0, i1 = i1, i0
    
    p0 = best_pts[i0]
    p1 = best_pts[i1]
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        return best_pts
    
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    transformed = (best_pts - p0) @ R_mat.T / norm_v
    
    perm = list(range(N))
    perm[0] = i0
    perm[1] = i1
    remaining = [i for i in range(N) if i != i0 and i != i1]
    perm[2:] = remaining
    transformed = transformed[perm]
    
    # Base positions for the sinusoidal deformation
    base_pts = transformed.copy()
    base_pts[0] = [0.0, 0.0]
    base_pts[1] = [1.0, 0.0]
    
    # Step 4: Sinusoidal deformation parameterization
    # Each point's position = base + sum of sinusoidal modes
    # Parameters: amplitudes and phases for a few global modes
    # This keeps the configuration close to the good lattice structure while allowing smooth deformation
    
    def sinusoidal_deform(params):
        """
        params: array of shape (num_params,)
        Structure:
          - First 4: global amplitude and phase for mode 1 (low frequency)
          - Next 4: global amplitude and phase for mode 2 (medium frequency)
          - Next 4: global amplitude and phase for mode 3 (higher frequency)
          - Remaining 28: per-point offsets (2 coords each for 14 free points)
        """
        pts = base_pts.copy()
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        
        # Global sinusoidal modes
        # Mode 1: low frequency
        amp1_x, amp1_y, phase1_x, phase1_y = params[0:4]
        # Mode 2: medium frequency
        amp2_x, amp2_y, phase2_x, phase2_y = params[4:8]
        # Mode 3: higher frequency
        amp3_x, amp3_y, phase3_x, phase3_y = params[8:12]
        
        # Per-point offsets for points 2..15
        offsets = params[12:].reshape(-1, 2)
        
        for i in range(2, N):
            # Base position
            bx, by = base_pts[i]
            
            # Sinusoidal deformations based on base position
            # Use normalized coordinates
            nx = bx / 3.0  # approximate scale
            ny = by / 3.0
            
            # Mode contributions
            dx1 = amp1_x * np.sin(2*np.pi*nx + phase1_x)
            dy1 = amp1_y * np.sin(2*np.pi*ny + phase1_y)
            dx2 = amp2_x * np.sin(4*np.pi*nx + phase2_x)
            dy2 = amp2_y * np.sin(4*np.pi*ny + phase2_y)
            dx3 = amp3_x * np.sin(6*np.pi*nx + phase3_x)
            dy3 = amp3_y * np.sin(6*np.pi*ny + phase3_y)
            
            pts[i] = [bx + dx1 + dx2 + dx3 + offsets[i-2, 0],
                      by + dy1 + dy2 + dy3 + offsets[i-2, 1]]
        
        return pts

    def objective_sinusoidal(params):
        if time.time() > deadline - 10:
            return 1e10
        pts = sinusoidal_deform(params)
        return score_points(pts)

    # Initial parameters: zero amplitudes, zero offsets
    num_params = 12 + 2*(N-2)  # 12 global + 28 per-point
    init_params = np.zeros(num_params)
    
    # Bounds: keep amplitudes small to prevent collapse
    bnds = []
    for _ in range(12):  # global params
        bnds.append((-0.5, 0.5))
    for _ in range(2*(N-2)):  # per-point offsets
        bnds.append((-0.3, 0.3))
    
    best_params = init_params.copy()
    best_opt_score = best_score
    
    # Step 5: Optimize the sinusoidal deformation
    try:
        res = minimize(
            objective_sinusoidal,
            init_params,
            method='L-BFGS-B',
            bounds=bnds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.fun < best_opt_score:
            best_opt_score = res.fun
            best_params = res.x
            best_pts = sinusoidal_deform(res.x)
    except Exception:
        pass

    # Step 6: Polish with direct optimization on the deformed configuration
    if time.time() < deadline - 15:
        try:
            # Extract current best points
            current_pts = best_pts.copy()
            
            # Re-normalize
            diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            iu = np.triu_indices(N, 1)
            d = dists[iu]
            min_idx = np.argmin(d)
            pair = np.unravel_index(min_idx, (N, N))
            i0, i1 = pair[0], pair[1]
            if i0 > i1:
                i0, i1 = i1, i0
            
            p0 = current_pts[i0]
            p1 = current_pts[i1]
            v = p1 - p0
            norm_v = np.linalg.norm(v)
            
            if norm_v > 1e-8:
                cos_t = v[0] / norm_v
                sin_t = v[1] / norm_v
                R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
                
                transformed = (current_pts - p0) @ R_mat.T / norm_v
                
                perm = list(range(N))
                perm[0] = i0
                perm[1] = i1
                remaining = [i for i in range(N) if i != i0 and i != i1]
                perm[2:] = remaining
                transformed = transformed[perm]
                
                init_params2 = transformed[2:].flatten()
                bnds2 = [(-10, 10)] * (2*(N-2))
                
                def objective_direct(params):
                    if time.time() > deadline - 5:
                        return 1e10
                    pts = np.zeros((N, 2))
                    pts[0] = [0.0, 0.0]
                    pts[1] = [1.0, 0.0]
                    pts[2:] = params.reshape(-1, 2)
                    return score_points(pts)
                
                res2 = minimize(
                    objective_direct,
                    init_params2,
                    method='L-BFGS-B',
                    bounds=bnds2,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res2.fun < best_opt_score:
                    final_pts = np.zeros((N, 2))
                    final_pts[0] = [0.0, 0.0]
                    final_pts[1] = [1.0, 0.0]
                    final_pts[2:] = res2.x.reshape(-1, 2)
                    
                    # Check validity
                    diff_f = final_pts[:, np.newaxis, :] - final_pts[np.newaxis, :, :]
                    dists_f = np.sqrt(np.sum(diff_f**2, axis=2))
                    iu_f = np.triu_indices(N, 1)
                    d_f = dists_f[iu_f]
                    
                    if np.min(d_f) > 1e-6:
                        best_opt_score = res2.fun
                        best_pts = final_pts
        except Exception:
            pass

    return best_pts
