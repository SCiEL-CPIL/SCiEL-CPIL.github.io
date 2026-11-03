"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def construct():
    """
    Place N=200 equal non-overlapping circles in a unit square [-0.5, 0.5]^2
    to maximize the minimum radius.
    
    Strategy:
    1. Start from a good initial configuration (hexagonal packing approximation).
    2. Use local optimization with gradient-based methods to maximize the minimum distance.
    3. Use simulated annealing or random restarts if needed.
    
    The objective is to maximize r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|))
    """
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Best known radius for reference
    best_known_r = 0.036612798904
    
    # Generate initial hexagonal packing
    def hex_packing_init(n):
        """Generate a hexagonal packing pattern for n circles in unit square."""
        # Hexagonal packing: rows offset by half spacing
        # Try different densities
        best_config = None
        best_min_dist = 0
        
        for trial in range(20):
            # Random perturbation of hexagonal grid
            # Estimate rows and columns
            # For hexagonal packing, aspect ratio is sqrt(3)/2
            # Try various grid configurations
            
            # Method: fill a hexagonal lattice, then select n points
            # Hexagonal lattice spacing s, rows offset
            # For n points, roughly sqrt(n) x sqrt(n) grid
            
            # Try different spacings
            base_sp = 1.0 / np.sqrt(n)
            
            # Create a grid slightly larger than needed
            n_extra = 10
            m = int(np.ceil(np.sqrt(n))) + n_extra
            
            # Hexagonal lattice points
            pts = []
            for row in range(m):
                for col in range(m):
                    x = col * base_sp * 1.0 + (0.5 * base_sp if row % 2 else 0)
                    y = row * base_sp * np.sqrt(3) / 2
                    pts.append([x, y])
            
            pts = np.array(pts)
            
            # Center the points
            center = pts.mean(axis=0)
            pts = pts - center
            
            # Scale to fit in [-0.5, 0.5]^2
            max_extent = np.max(np.abs(pts))
            if max_extent > 0:
                pts = pts / max_extent * 0.45  # Leave some margin
            
            # Add random jitter
            jitter = np.random.randn(len(pts), 2) * 0.01
            pts = pts + jitter
            
            # Select n points that are well-distributed
            # Use farthest point sampling
            selected = [pts[0]]
            min_dists = np.full(len(pts), np.inf)
            min_dists[0] = 0
            
            while len(selected) < n and len(selected) < len(pts):
                # Find point with maximum minimum distance to selected
                new_min_dists = np.min(np.linalg.norm(pts[:, np.newaxis] - np.array(selected)[np.newaxis, :], axis=1), axis=1)
                # Weight by both distance to selected and distance to boundary
                boundary_dist = np.min(np.minimum(0.5 - np.abs(pts), 0.5 - np.abs(pts)), axis=1)
                score = 0.7 * new_min_dists + 0.3 * boundary_dist
                # Avoid already selected
                for s in selected:
                    score[np.argmin(np.linalg.norm(pts - s, axis=1))] = -1
                next_idx = np.argmax(score)
                selected.append(pts[next_idx])
            
            config = np.array(selected[:n])
            
            # Compute min distance
            min_d = np.inf
            for i in range(len(config)):
                for j in range(i + 1, len(config)):
                    d = np.linalg.norm(config[i] - config[j])
                    min_d = min(min_d, d)
                # Boundary
                bd = np.min(0.5 - np.abs(config[i]))
                min_d = min(min_d, bd)
            
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_config = config
        
        return best_config, best_min_dist / 2
    
    # Get initial configuration
    config, initial_r = hex_packing_init(N)
    
    # Local optimization using scipy
    def objective(x):
        """Negative of the minimum radius (to minimize)."""
        pts = x.reshape(-1, 2)
        # Check bounds
        if np.any(np.abs(pts) > 0.5):
            return 1000.0
        
        # Compute all pairwise distances
        min_d = 0.5  # Start with boundary
        for i in range(N):
            # Boundary distance
            bd = np.min(0.5 - np.abs(pts[i]))
            min_d = min(min_d, bd)
        
        # Pairwise distances
        # Use vectorized computation for speed
        for i in range(N):
            dists = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
            min_d = min(min_d, dists.min() if len(dists) > 0 else 0.5)
        
        return -min_d
    
    def objective_grad(x):
        """Gradient of the negative minimum radius."""
        pts = x.reshape(-1, 2)
        grad = np.zeros_like(x)
        
        # Find the minimum distance pair
        min_d = 0.5
        min_i, min_j = 0, 0
        min_type = 'boundary'  # 'pair' or 'boundary'
        
        # Check boundaries
        for i in range(N):
            bd = np.min(0.5 - np.abs(pts[i]))
            if bd < min_d:
                min_d = bd
                min_i = i
                min_j = i
                min_type = 'boundary'
        
        # Check pairs
        for i in range(N):
            dists = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
            for j_idx in range(len(dists)):
                j = i + 1 + j_idx
                if dists[j_idx] < min_d:
                    min_d = dists[j_idx]
                    min_i = i
                    min_j = j
                    min_type = 'pair'
        
        # Compute gradient
        if min_type == 'pair':
            diff = pts[min_i] - pts[min_j]
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                unit = diff / norm
                grad[min_i * 2: min_i * 2 + 2] = -unit
                grad[min_j * 2: min_j * 2 + 2] = unit
        else:
            # Boundary
            for k in range(2):
                if pts[min_i, k] > 0:
                    grad[min_i * 2 + k] = -1.0
                else:
                    grad[min_i * 2 + k] = 1.0
        
        return grad
    
    # Optimize
    best_r = initial_r
    best_config = config.copy()
    
    # Multiple optimization passes with different methods
    methods = ['L-BFGS-B', 'Powell', 'Nelder-Mead']
    
    for method in methods:
        if time.time() > deadline - 20:
            break
        
        x0 = best_config.flatten()
        
        try:
            if method == 'L-BFGS-B':
                res = minimize(objective, x0, method=method, jac=objective_grad,
                              bounds=[(-0.5, 0.5)] * (2 * N),
                              options={'maxiter': 100, 'gtol': 1e-8})
            elif method == 'Powell':
                res = minimize(objective, x0, method=method,
                              bounds=[(-0.5, 0.5)] * (2 * N),
                              options={'maxiter': 50})
            else:
                res = minimize(objective, x0, method=method,
                              bounds=[(-0.5, 0.5)] * (2 * N),
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.success or res.fun < objective(x0):
                new_config = res.x.reshape(-1, 2)
                # Clip to bounds
                new_config = np.clip(new_config, -0.5, 0.5)
                new_r = -res.fun
                if new_r > best_r:
                    best_r = new_r
                    best_config = new_config
        except Exception:
            pass
        
        if time.time() > deadline - 15:
            break
    
    # Additional local search: random perturbations and re-optimization
    if time.time() < deadline - 10:
        n_restarts = 10
        for _ in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Perturb current best
            perturbation = np.random.randn(N, 2) * 0.005
            perturbed = best_config + perturbation
            perturbed = np.clip(perturbed, -0.5, 0.5)
            
            x0 = perturbed.flatten()
            try:
                res = minimize(objective, x0, method='L-BFGS-B', jac=objective_grad,
                              bounds=[(-0.5, 0.5)] * (2 * N),
                              options={'maxiter': 50, 'gtol': 1e-8})
                
                if res.fun < -best_r:
                    new_config = res.x.reshape(-1, 2)
                    new_config = np.clip(new_config, -0.5, 0.5)
                    new_r = -res.fun
                    if new_r > best_r:
                        best_r = new_r
                        best_config = new_config
            except Exception:
                pass
    
    # Final validation
    best_config = np.clip(best_config, -0.5, 0.5)
    
    return best_config
# EVOLVE-BLOCK-END
