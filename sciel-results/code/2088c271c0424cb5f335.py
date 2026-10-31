import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Generate hexagonal lattice points
    R = 8
    pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            pts.append((x, y))
    
    pts = np.array(pts)
    num_pts = len(pts)
    
    # Get initial subset: 16 closest to origin
    dists_from_origin = np.sum(pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_subset = sorted_indices[:N]
    
    # Candidate pool for swaps
    pool_size = 200
    candidate_indices = sorted_indices[:pool_size]
    
    def score_subset(indices):
        sub_pts = pts[indices]
        diff = sub_pts[:, np.newaxis, :] - sub_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2
    
    current_indices = initial_subset.copy()
    current_score = score_subset(current_indices)
    
    in_subset = set(current_indices)
    outside_candidates = [i for i in candidate_indices if i not in in_subset]
    
    # Local search: swap points to improve
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        for i_idx in range(N):
            if time.time() > deadline - 30:
                break
            for j in outside_candidates:
                if time.time() > deadline - 30:
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
    
    best_pts = pts[current_indices]
    
    # Normalize: fix closest pair to (0,0) and (1,0)
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, 1)
    d = dists[iu]
    min_idx = np.argmin(d)
    pair = np.unravel_index(min_idx, (N, N))
    i0, i1 = pair
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
    
    # Reorder so i0 is at index 0, i1 at index 1
    perm = list(range(N))
    perm[0] = i0
    perm[1] = i1
    remaining = [i for i in range(N) if i != i0 and i != i1]
    perm[2:] = remaining
    transformed = transformed[perm]
    
    init_params = transformed[2:].flatten()
    
    def exact_objective(params):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-10:
            return 1e10
        return max_d_sq / min_d_sq
    
    # Smooth surrogate using log-sum-exp
    def soft_objective(params, beta=30.0):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Soft max and min using log-sum-exp
        max_d_sq_soft = np.log(np.sum(np.exp(beta * d_sq))) / beta
        min_d_sq_soft = -np.log(np.sum(np.exp(-beta * d_sq))) / beta
        
        if min_d_sq_soft < 1e-10:
            return 1e10
        
        return max_d_sq_soft / min_d_sq_soft
    
    bnds = [(-15, 15)] * (2 * (N - 2))
    
    best_params = init_params
    best_score = exact_objective(init_params)
    
    # Phase 1: Smooth surrogate optimization
    if time.time() < deadline - 40:
        try:
            res = minimize(soft_objective, init_params, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < 1e9:
                exact_score = exact_objective(res.x)
                if exact_score < best_score:
                    best_params = res.x
                    best_score = exact_score
        except Exception:
            pass
    
    # Phase 2: Exact objective with L-BFGS-B
    if time.time() < deadline - 20:
        try:
            res = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 10:
        try:
            res = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12})
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass
    
    # Try multiple random restarts if time permits
    if time.time() < deadline - 15:
        rng = np.random.default_rng(42)
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            # Perturb the best solution
            perturbation = rng.normal(0, 0.1, size=init_params.shape)
            restart_params = best_params + perturbation
            restart_score = exact_objective(restart_params)
            
            if restart_score < 1e9:
                try:
                    res = minimize(exact_objective, restart_params, method='L-BFGS-B', bounds=bnds,
                                   options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
                    if res.fun < best_score:
                        best_params = res.x
                        best_score = res.fun
                except Exception:
                    pass
    
    # NEW PHASE 4: Polynomial critical points optimization
    # Represent points as critical points of P(z) = prod(z - z_k)
    # Optimize polynomial coefficients to minimize distance ratio
    
    if time.time() < deadline - 5:
        try:
            def poly_critical_objective(coeffs):
                # coeffs are complex coefficients of P(z) = z^16 + c15*z^15 + ... + c0
                # P has 16 roots, P' has 15 roots (critical points)
                # But we need 16 points, so use P of degree 17, P' has 16 roots
                # Let P(z) = z^17 + a16*z^16 + ... + a0
                # coeffs[0:16] are a16...a1 (real and imag interleaved)
                
                # Build polynomial from coefficients
                n_poly = 17
                # coeffs has 2*16 = 32 elements (16 complex coefficients)
                complex_coeffs = coeffs[0::2] + 1j * coeffs[1::2]
                
                # P(z) = z^17 + c[0]*z^16 + c[1]*z^15 + ... + c[15]
                poly_coeffs = np.ones(1)
                for c in complex_coeffs:
                    poly_coeffs = np.convolve(poly_coeffs, np.array([1, c]))
                
                # Find roots of P' (derivative)
                # P'(z) = 17*z^16 + 16*c[0]*z^15 + ... + c[15]
                deriv_coeffs = np.array([17.0 + 0j])
                for i in range(16):
                    deriv_coeffs = np.convolve(deriv_coeffs, np.array([1, complex_coeffs[i]]))
                # Actually, let's compute derivative properly
                # If P(z) = sum_{k=0}^{17} a_k z^k where a_17=1
                # P'(z) = sum_{k=1}^{17} k*a_k z^{k-1}
                
                # Rebuild: a[17] = 1, a[16] = c[0], ..., a[1] = c[15], a[0] = 0 (or we can set a[0])
                a = np.zeros(18, dtype=complex)
                a[17] = 1.0 + 0j
                for i in range(16):
                    a[16-i] = complex_coeffs[i]
                # a[0] is free, set to 0 for simplicity
                
                # Derivative coefficients: d[k] = (k+1)*a[k+1] for k=0..16
                d = np.array([(k+1)*a[k+1] for k in range(17)], dtype=complex)
                
                # Find roots of derivative polynomial
                try:
                    critical_pts = np.roots(d)
                except Exception:
                    return 1e10
                
                # Convert to real 2D points
                points_2d = np.column_stack([np.real(critical_pts), np.imag(critical_pts)])
                
                # Check for distinct points
                if len(np.unique(points_2d, axis=0)) < N:
                    return 1e10
                
                # Compute distance ratio
                diff = points_2d[:, np.newaxis, :] - points_2d[np.newaxis, :, :]
                dists_sq = np.sum(diff**2, axis=2)
                iu = np.triu_indices(N, 1)
                d_sq = dists_sq[iu]
                
                min_d_sq = np.min(d_sq)
                max_d_sq = np.max(d_sq)
                
                if min_d_sq < 1e-10:
                    return 1e10
                
                return max_d_sq / min_d_sq
            
            # Initialize with small random coefficients
            rng2 = np.random.default_rng(123)
            init_poly_coeffs = rng2.normal(0, 0.5, size=32)
            
            # Optimize
            poly_bnds = [(-5, 5)] * 32
            res_poly = minimize(poly_critical_objective, init_poly_coeffs, method='L-BFGS-B', 
                               bounds=poly_bnds, options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res_poly.fun < best_score and res_poly.fun < 1e9:
                # Extract the 16 points from the optimal polynomial
                complex_coeffs = res_poly.x[0::2] + 1j * res_poly.x[1::2]
                a = np.zeros(18, dtype=complex)
                a[17] = 1.0 + 0j
                for i in range(16):
                    a[16-i] = complex_coeffs[i]
                d = np.array([(k+1)*a[k+1] for k in range(17)], dtype=complex)
                critical_pts = np.roots(d)
                points_2d = np.column_stack([np.real(critical_pts), np.imag(critical_pts)])
                
                # Normalize: fix closest pair to (0,0) and (1,0)
                diff = points_2d[:, np.newaxis, :] - points_2d[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                iu = np.triu_indices(N, 1)
                d = dists[iu]
                min_idx = np.argmin(d)
                pair = np.unravel_index(min_idx, (N, N))
                i0, i1 = pair
                if i0 > i1:
                    i0, i1 = i1, i0
                
                p0 = points_2d[i0]
                p1 = points_2d[i1]
                v = p1 - p0
                norm_v = np.linalg.norm(v)
                
                if norm_v > 1e-8:
                    cos_t = v[0] / norm_v
                    sin_t = v[1] / norm_v
                    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
                    transformed_poly = (points_2d - p0) @ R_mat.T / norm_v
                    
                    perm = list(range(N))
                    perm[0] = i0
                    perm[1] = i1
                    remaining = [i for i in range(N) if i != i0 and i != i1]
                    perm[2:] = remaining
                    transformed_poly = transformed_poly[perm]
                    
                    poly_params = transformed_poly[2:].flatten()
                    poly_score = exact_objective(poly_params)
                    
                    if poly_score < best_score:
                        best_params = poly_params
                        best_score = poly_score
                        
        except Exception:
            pass
    
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    return final_pts

# EVOLVE-BLOCK-END
