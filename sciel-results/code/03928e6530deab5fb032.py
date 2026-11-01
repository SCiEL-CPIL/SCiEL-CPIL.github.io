"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    def compute_radius(pts):
        """Compute the actual radius (min of half min-pair-distance and min boundary distance)."""
        # Pairwise distances
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d)
        
        # Boundary distances
        boundary = 0.5 - np.abs(pts)
        min_boundary = np.min(boundary)
        
        return min(min_pair / 2.0, min_boundary)
    
    def logsumexp_min(pts, beta):
        """
        Compute soft-min approximation of the radius using LogSumExp.
        Returns (soft_min_value, gradient) where gradient is for ASCENT (maximization).
        """
        # Pairwise distances
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        
        # Boundary distances
        boundary = 0.5 - np.abs(pts)
        
        # Combine all constraints: we want min of {d_ij/2 for all i<j, boundary_ik for all i,k}
        # For LogSumExp soft-min: -1/beta * log(sum(exp(-beta * x))) approximates min(x)
        # We maximize this, so we minimize 1/beta * log(sum(exp(-beta * x)))
        
        # Pairwise terms: x = d/2
        d_half = d / 2.0
        np.fill_diagonal(d_half, np.inf)
        
        # Only use upper triangle to avoid double counting
        iu = np.triu_indices(N, k=1)
        d_upper = d_half[iu]
        
        # Boundary terms
        b = boundary.ravel()
        
        # Combine all terms
        all_terms = np.concatenate([d_upper, b])
        
        # Soft min via LogSumExp: soft_min ≈ -1/beta * log(sum(exp(-beta * x)))
        # To avoid overflow, use log-sum-exp trick
        m = np.min(all_terms)
        exp_terms = np.exp(-beta * (all_terms - m))
        log_sum = np.log(np.sum(exp_terms))
        soft_min_val = -1.0 / beta * log_sum + m
        
        # Gradient of soft_min with respect to pts (for maximization)
        # d(soft_min)/d(pts) = -1/beta * sum(exp(-beta*(x-m)) * (-beta * dx/dpts)) / sum(exp(-beta*(x-m)))
        #                     = sum(exp(-beta*(x-m)) * dx/dpts) / sum(exp(-beta*(x-m)))
        
        weights = exp_terms / np.sum(exp_terms)
        
        grad = np.zeros_like(pts)
        
        # Pairwise contribution: d(d_ij/2)/d(pts_i) = (pts_i - pts_j) / (2 * d_ij) for i != j
        # But we need to be careful: d_ij = ||pts_i - pts_j||, so d(d_ij/2)/d(pts_i) = (pts_i - pts_j) / (2*d_ij)
        
        # Process pairwise terms
        w_pairs = weights[:len(d_upper)]
        for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                grad[i] += w_pairs[idx] * diff / (2.0 * dist)
                grad[j] -= w_pairs[idx] * diff / (2.0 * dist)
        
        # Boundary contribution: d(0.5 - |pts_ik|)/d(pts_ik) = -sign(pts_ik)
        w_bound = weights[len(d_upper):]
        for idx in range(N * 3):
            i = idx // 3
            k = idx % 3
            s = np.sign(pts[i, k])
            if s == 0:
                s = 1.0
            grad[i, k] += w_bound[idx] * (-s)  # Negative because we're maximizing -|x|
        
        return soft_min_val, grad
    
    def simulated_annealing(pts0, T0=0.1, T_end=0.001, cooling=0.995, max_iters=50000):
        """Simulated annealing with LogSumExp objective."""
        pts = pts0.copy()
        current_val = -np.inf
        
        # Compute initial value
        try:
            current_val, _ = logsumexp_min(pts, beta=50.0)
        except:
            current_val = -1.0
        
        best_val = current_val
        best_pts = pts.copy()
        
        T = T0
        beta = 20.0  # Start with moderate sharpness
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Increase beta over time (sharper approximation)
            beta = min(beta * 1.0005, 200.0)
            
            # Compute gradient
            try:
                val, grad = logsumexp_min(pts, beta)
            except:
                val = -1.0
                grad = np.zeros_like(pts)
            
            if val > best_val:
                best_val = val
                best_pts = pts.copy()
            
            # Propose move: gradient ascent + random perturbation
            step_scale = 0.01 * (T / T0) ** 0.5
            move = grad * step_scale + np.random.normal(0, 0.005 * (T / T0) ** 0.5, size=(N, 3))
            
            new_pts = pts + move
            new_pts = np.clip(new_pts, -0.4999, 0.4999)
            
            try:
                new_val, _ = logsumexp_min(new_pts, beta)
            except:
                new_val = -1.0
            
            # Accept if better or with probability exp((new_val - val) / T)
            if new_val > val:
                pts = new_pts
                val = new_val
            else:
                delta = new_val - val
                if np.random.random() < np.exp(delta / max(T, 1e-10)):
                    pts = new_pts
                    val = new_val
            
            # Cool down
            T *= cooling
        
        return best_pts, best_val
    
    def polish(pts, max_iters=5000):
        """High-beta LogSumExp polish for final refinement."""
        pts = pts.copy()
        best_val = -np.inf
        best_pts = pts.copy()
        
        beta = 100.0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            try:
                val, grad = logsumexp_min(pts, beta)
            except:
                break
            
            if val > best_val:
                best_val = val
                best_pts = pts.copy()
            
            # Small step ascent
            step = 0.001
            new_pts = pts + step * grad
            new_pts = np.clip(new_pts, -0.4999, 0.4999)
            
            try:
                new_val, _ = logsumexp_min(new_pts, beta)
            except:
                break
            
            if new_val > val:
                pts = new_pts
                val = new_val
            else:
                # Try smaller step
                step *= 0.5
                if step < 1e-8:
                    break
                new_pts = pts + step * grad
                new_pts = np.clip(new_pts, -0.4999, 0.4999)
                try:
                    new_val, _ = logsumexp_min(new_pts, beta)
                except:
                    break
                if new_val > val:
                    pts = new_pts
                    val = new_val
                else:
                    break
        
        return best_pts, best_val
    
    def init_grid(n1, n2, n3):
        tx = (np.arange(n1) + 0.5) / n1 - 0.5
        ty = (np.arange(n2) + 0.5) / n2 - 0.5
        tz = (np.arange(n3) + 0.5) / n3 - 0.5
        grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
        return grid[:N]
    
    def init_fcc():
        """FCC lattice initialization."""
        pts_list = []
        n_cells = 5
        spacing = 0.9 / n_cells
        
        for ix in range(n_cells):
            for iy in range(n_cells):
                for iz in range(n_cells):
                    base_x = -0.45 + ix * spacing
                    base_y = -0.45 + iy * spacing
                    base_z = -0.45 + iz * spacing
                    pts_list.append([base_x, base_y, base_z])
                    pts_list.append([base_x + spacing/2, base_y + spacing/2, base_z])
                    pts_list.append([base_x + spacing/2, base_y, base_z + spacing/2])
                    pts_list.append([base_x, base_y + spacing/2, base_z + spacing/2])
        
        pts = np.array(pts_list[:N])
        max_abs = np.max(np.abs(pts))
        if max_abs > 0.45:
            pts = pts * (0.45 / max_abs)
        
        return pts
    
    # Multiple initializations
    np.random.seed(42)
    starts = [
        init_grid(5, 5, 4),
        init_grid(4, 5, 5),
        init_grid(5, 4, 5),
        init_fcc(),
        init_grid(5, 5, 4) + np.random.normal(0, 0.03, size=(N, 3)),
        init_fcc() + np.random.normal(0, 0.03, size=(N, 3)),
    ]
    
    # Clip all starts
    for i in range(len(starts)):
        starts[i] = np.clip(starts[i], -0.49, 0.49)
    
    best_radius = -1.0
    best_points = None
    
    # Run SA from each start
    for start_idx, x0 in enumerate(starts):
        if time.time() > deadline - 25:
            break
        
        try:
            sa_pts, sa_val = simulated_annealing(x0, T0=0.1, T_end=0.001, cooling=0.995, max_iters=30000)
            
            # Polish the result
            pol_pts, pol_val = polish(sa_pts, max_iters=3000)
            
            # Compute actual radius
            actual_r = compute_radius(pol_pts)
            
            if actual_r > best_radius:
                best_radius = actual_r
                best_points = pol_pts.copy()
        except Exception as e:
            continue
    
    # If we have a good result, do additional refinement
    if best_points is not None and time.time() < deadline - 15:
        # Try perturbation and re-optimize
        for perturb in range(5):
            if time.time() > deadline - 8:
                break
            try:
                perturbed = best_points + np.random.normal(0, 0.005, size=(N, 3))
                perturbed = np.clip(perturbed, -0.49, 0.49)
                
                sa_pts, sa_val = simulated_annealing(perturbed, T0=0.05, T_end=0.0005, cooling=0.998, max_iters=10000)
                pol_pts, pol_val = polish(sa_pts, max_iters=2000)
                actual_r = compute_radius(pol_pts)
                
                if actual_r > best_radius:
                    best_radius = actual_r
                    best_points = pol_pts.copy()
            except:
                continue
    
    # Final polish
    if best_points is not None and time.time() < deadline - 5:
        try:
            pol_pts, pol_val = polish(best_points, max_iters=5000)
            actual_r = compute_radius(pol_pts)
            if actual_r > best_radius:
                best_radius = actual_r
                best_points = pol_pts.copy()
        except:
            pass
    
    if best_points is None:
        best_points = init_grid(5, 5, 4)
    
    # Ensure valid output
    best_points = np.clip(best_points, -0.5, 0.5)
    return best_points
# EVOLVE-BLOCK-END
