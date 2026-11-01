"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    def compute_exact_score(pts):
        """Compute the exact packing radius for given centers."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        
        # Radial distances
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0
        
        # Pairwise distances
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        
        min_pair_dist = np.min(pair_dists)
        min_radial_dist = 1.0 - np.max(dists)
        
        r = min(min_pair_dist / 2.0, min_radial_dist)
        return r

    def generate_hexagonal_initialization():
        """Generate a hexagonal lattice clipped to the unit circle."""
        best_pts = None
        best_score = -1.0
        
        # Try a few spacings to ensure we get enough points
        for scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
            spacing = 2.0 / (np.sqrt(N) * scale)
            
            # Generate grid points
            nx = int(2.0 / spacing) + 3
            ny = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 3
            
            # Center the grid
            x_coords = (np.arange(-nx//2, nx//2 + 1) + 0.5) * spacing
            y_coords = (np.arange(-ny//2, ny//2 + 1)) * spacing * np.sqrt(3.0) / 2.0
            
            pts_list = []
            for i, y in enumerate(y_coords):
                offset = 0.0 if i % 2 == 0 else spacing / 2.0
                for x in x_coords:
                    px = x + offset
                    py = y
                    if np.hypot(px, py) <= 1.0 - spacing / 2.0:
                        pts_list.append([px, py])
            
            if len(pts_list) < N:
                continue
                
            pts = np.array(pts_list)
            norms = np.linalg.norm(pts, axis=1)
            idx = np.argsort(norms)
            pts = pts[idx[:N]]
            
            score = compute_exact_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts
        
        if best_pts is None:
            best_pts = np.random.rand(N, 2) * 2 - 1
            norms = np.linalg.norm(best_pts, axis=1)
            best_pts = best_pts * (0.9 / norms)[:, None]
            best_score = compute_exact_score(best_pts)
            
        return best_pts, best_score

    def generate_ring_initialization():
        """Generate points on concentric rings."""
        best_pts = None
        best_score = -1.0
        
        # Try different numbers of rings
        for n_rings in range(5, 15):
            pts_list = []
            for ring_idx in range(n_rings):
                r = (ring_idx + 1) / n_rings * 0.95
                n_on_ring = max(1, int(2 * np.pi * r * n_rings / 3))
                angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False)
                for a in angles:
                    pts_list.append([r * np.cos(a), r * np.sin(a)])
            
            if len(pts_list) < N:
                continue
                
            pts = np.array(pts_list)
            norms = np.linalg.norm(pts, axis=1)
            idx = np.argsort(norms)
            pts = pts[idx[:N]]
            
            score = compute_exact_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts
        
        if best_pts is None:
            best_pts = np.random.rand(N, 2) * 2 - 1
            norms = np.linalg.norm(best_pts, axis=1)
            best_pts = best_pts * (0.9 / norms)[:, None]
            best_score = compute_exact_score(best_pts)
            
        return best_pts, best_score

    def soft_min_2(x, y, beta):
        """
        Smooth approximation of min(x, y) using log-sum-exp.
        """
        m = min(x, y)
        return m - np.logaddexp(-beta * (x - m), -beta * (y - m)) / beta

    def create_objective(pts_init, beta):
        """
        Create the objective function for L-BFGS-B.
        Uses auxiliary variable t to handle min more smoothly.
        """
        def objective(params):
            if time.time() > deadline:
                return 1e10
            
            pts = params.reshape(-1, 2)
            
            if not np.all(np.isfinite(pts)):
                return 1e10
            
            dists = np.linalg.norm(pts, axis=1)
            if np.any(dists > 1.0):
                return 1e10 + np.sum(np.maximum(dists - 1.0, 0.0))
            
            i_idx, j_idx = np.triu_indices(N, k=1)
            diffs = pts[i_idx] - pts[j_idx]
            pair_dists = np.linalg.norm(diffs, axis=1)
            
            min_pair_dist = np.min(pair_dists)
            min_radial_dist = 1.0 - np.max(dists)
            
            x = min_pair_dist / 2.0
            y = min_radial_dist
            
            val = soft_min_2(x, y, beta)
            
            return -val
        
        return objective

    def create_objective_with_t(pts_init, beta):
        """
        Objective with auxiliary variable t: minimize -t + penalty for t > x or t > y.
        params = [x1, y1, x2, y2, ..., t]
        """
        def objective(params):
            if time.time() > deadline:
                return 1e10
            
            t = params[-1]
            pts = params[:-1].reshape(-1, 2)
            
            if not np.all(np.isfinite(pts)):
                return 1e10
            
            dists = np.linalg.norm(pts, axis=1)
            if np.any(dists > 1.0):
                return 1e10 + np.sum(np.maximum(dists - 1.0, 0.0))
            
            i_idx, j_idx = np.triu_indices(N, k=1)
            diffs = pts[i_idx] - pts[j_idx]
            pair_dists = np.linalg.norm(diffs, axis=1)
            
            min_pair_dist = np.min(pair_dists)
            min_radial_dist = 1.0 - np.max(dists)
            
            x = min_pair_dist / 2.0
            y = min_radial_dist
            
            # Penalty if t exceeds either x or y
            penalty = np.maximum(0, t - x) + np.maximum(0, t - y)
            
            return -t + beta * penalty
        
        return objective

    # 1. Generate initial points from multiple strategies
    hex_pts, hex_score = generate_hexagonal_initialization()
    ring_pts, ring_score = generate_ring_initialization()
    
    # Pick the better initialization
    if ring_score > hex_score:
        pts, best_score = ring_pts, ring_score
    else:
        pts, best_score = hex_pts, hex_score
    
    # 2. Multi-stage optimization with increasing beta
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    current_pts = pts
    current_score = best_score
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        objective = create_objective(current_pts, beta)
        
        try:
            res = minimize(objective, current_pts.flatten(), method='L-BFGS-B',
                           bounds=[(-1.0, 1.0)] * (2 * N),
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.success or res.fun < 0:
                new_pts = res.x.reshape(-1, 2)
                new_score = compute_exact_score(new_pts)
                
                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
        except Exception:
            pass

    # 3. Try auxiliary variable formulation
    if time.time() < deadline - 15:
        t_init = current_score
        init_params = np.append(current_pts.flatten(), t_init)
        objective_t = create_objective_with_t(current_pts, 100.0)
        
        try:
            res = minimize(objective_t, init_params, method='L-BFGS-B',
                           bounds=[(-1.0, 1.0)] * (2 * N) + [(0.0, 1.0)],
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.success or res.fun < 0:
                new_pts = res.x[:-1].reshape(-1, 2)
                new_score = compute_exact_score(new_pts)
                
                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
        except Exception:
            pass

    # 4. Local polish with high beta and tight tolerances
    if time.time() < deadline - 5:
        objective = create_objective(current_pts, 5000.0)
        try:
            res = minimize(objective, current_pts.flatten(), method='L-BFGS-B',
                           bounds=[(-1.0, 1.0)] * (2 * N),
                           options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
            
            if res.success or res.fun < 0:
                new_pts = res.x.reshape(-1, 2)
                new_score = compute_exact_score(new_pts)
                
                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
        except Exception:
            pass

    # Final validation
    current_pts = current_pts.reshape(-1, 2)
    norms = np.linalg.norm(current_pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        current_pts[mask] *= (1.0 / norms[mask])[:, np.newaxis]
        
    return current_pts
# EVOLVE-BLOCK-END
