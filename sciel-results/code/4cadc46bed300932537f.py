import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time

def construct():
    N = 26
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    # Initial configuration: Hexagonal-like packing
    # For 26 circles, use a 6x5 grid and select 26 most central positions
    cols, rows = 6, 5
    xs = np.linspace(0.5/cols, 1 - 0.5/cols, cols)
    ys = np.linspace(0.5/rows, 1 - 0.5/rows, rows)
    
    centers = []
    for x in xs:
        for y in ys:
            centers.append([x, y])
    centers = np.array(centers)
    
    # Select 26 most central positions
    d2c = np.sum((centers - 0.5)**2, axis=1)
    idx = np.argsort(d2c)[:N]
    pos = centers[idx].copy()
    
    # Initial radii: small to ensure validity
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            min_dist = min(min_dist, d)
        min_dist = min(min_dist, pos[i][0], 1-pos[i][0], pos[i][1], 1-pos[i][1])
    
    radii = np.full(N, min_dist * 0.9)
    
    def is_valid(pos, radii):
        for i in range(N):
            if pos[i][0] - radii[i] < -1e-8 or pos[i][0] + radii[i] > 1 + 1e-8:
                return False
            if pos[i][1] - radii[i] < -1e-8 or pos[i][1] + radii[i] > 1 + 1e-8:
                return False
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < radii[i] + radii[j] - 1e-8:
                    return False
        return True
    
    def get_score(pos, radii):
        if not is_valid(pos, radii):
            return -np.inf
        return np.sum(radii)
    
    def get_delaunay_edges(pos):
        """Get edges from Delaunay triangulation of centers"""
        try:
            tri = Delaunay(pos)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        edges.add((min(a, b), max(a, b)))
            return list(edges)
        except Exception:
            # Fallback: all pairs
            return [(i, j) for i in range(N) for j in range(i+1, N)]
    
    def objective(x):
        """Penalty-based objective: minimize penalty + maximize radii (via negative sum)"""
        pos = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        
        # Penalty for violations
        penalty = 0.0
        for i in range(N):
            # Boundary violations
            if pos[i][0] < radii[i]:
                penalty += (radii[i] - pos[i][0])**2
            if pos[i][0] > 1 - radii[i]:
                penalty += (pos[i][0] + radii[i] - 1)**2
            if pos[i][1] < radii[i]:
                penalty += (radii[i] - pos[i][1])**2
            if pos[i][1] > 1 - radii[i]:
                penalty += (pos[i][1] + radii[i] - 1)**2
        
        # Overlap penalties (use Delaunay edges for efficiency)
        edges = get_delaunay_edges(pos)
        for i, j in edges:
            d = np.linalg.norm(pos[i] - pos[j])
            overlap = radii[i] + radii[j] - d
            if overlap > 0:
                penalty += overlap**2
        
        # Also check all pairs for safety (but weight less)
        for i in range(N):
            for j in range(i+1, N):
                if (i, j) in [(min(a,b), max(a,b)) for a, b in edges]:
                    continue
                d = np.linalg.norm(pos[i] - pos[j])
                overlap = radii[i] + radii[j] - d
                if overlap > 0:
                    penalty += 0.1 * overlap**2
        
        # Negative sum of radii to maximize
        return penalty - 0.5 * np.sum(radii)
    
    def gradient(x):
        """Compute gradient of objective"""
        pos = x[:2*N].reshape(N, 2).copy()
        radii = x[2*N:].copy()
        
        grad_pos = np.zeros_like(pos)
        grad_radii = np.zeros_like(radii)
        
        # Boundary gradients
        for i in range(N):
            if pos[i][0] < radii[i]:
                grad_pos[i][0] += 2 * (radii[i] - pos[i][0])
                grad_radii[i] += 2 * (radii[i] - pos[i][0])
            if pos[i][0] > 1 - radii[i]:
                grad_pos[i][0] -= 2 * (pos[i][0] + radii[i] - 1)
                grad_radii[i] -= 2 * (pos[i][0] + radii[i] - 1)
            if pos[i][1] < radii[i]:
                grad_pos[i][1] += 2 * (radii[i] - pos[i][1])
                grad_radii[i] += 2 * (radii[i] - pos[i][1])
            if pos[i][1] > 1 - radii[i]:
                grad_pos[i][1] -= 2 * (pos[i][1] + radii[i] - 1)
                grad_radii[i] -= 2 * (pos[i][1] + radii[i] - 1)
        
        # Overlap gradients
        edges = get_delaunay_edges(pos)
        for i, j in edges:
            d_vec = pos[i] - pos[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-10:
                continue
            overlap = radii[i] + radii[j] - d
            if overlap > 0:
                force = 2 * overlap / d
                grad_pos[i] += force * d_vec
                grad_pos[j] -= force * d_vec
                grad_radii[i] += 2 * overlap
                grad_radii[j] += 2 * overlap
        
        # Check all pairs
        for i in range(N):
            for j in range(i+1, N):
                key = (min(i,j), max(i,j))
                if key in [(min(a,b), max(a,b)) for a, b in edges]:
                    continue
                d_vec = pos[i] - pos[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-10:
                    continue
                overlap = radii[i] + radii[j] - d
                if overlap > 0:
                    force = 0.2 * overlap / d
                    grad_pos[i] += force * d_vec
                    grad_pos[j] -= force * d_vec
                    grad_radii[i] += 0.2 * overlap
                    grad_radii[j] += 0.2 * overlap
        
        # Gradient of -0.5 * sum(radii)
        grad_radii -= 0.5
        
        return np.concatenate([grad_pos.flatten(), grad_radii])
    
    current_score = get_score(pos, radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = current_score
    
    # Optimization loop
    while time.time() < deadline - 10:
        # Flatten for optimization
        x0 = np.concatenate([pos.flatten(), radii])
        
        # Use L-BFGS-B with bounds
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.49))  # x_i in [0.01, 0.49]
            bounds.append((0.01, 0.49))  # y_i in [0.01, 0.49]
        for i in range(N):
            bounds.append((0.005, 0.49))  # radii in [0.005, 0.49]
        
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'gtol': 1e-4, 'maxls': 20}
            )
            
            if result.success or result.fun < 0:  # Improved
                x_opt = result.x
                pos_new = x_opt[:2*N].reshape(N, 2)
                radii_new = x_opt[2*N:]
                
                # Ensure validity
                if is_valid(pos_new, radii_new):
                    score_new = get_score(pos_new, radii_new)
                    if score_new > best_score:
                        best_score = score_new
                        best_pos = pos_new.copy()
                        best_radii = radii_new.copy()
                        pos = pos_new.copy()
                        radii = radii_new.copy()
                else:
                    # Try to fix validity by shrinking radii slightly
                    for scale in [0.95, 0.9, 0.85, 0.8]:
                        radii_try = radii_new * scale
                        if is_valid(pos_new, radii_try):
                            score_try = get_score(pos_new, radii_try)
                            if score_try > best_score:
                                best_score = score_try
                                best_pos = pos_new.copy()
                                best_radii = radii_try.copy()
                                pos = pos_new.copy()
                                radii = radii_try.copy()
                            break
        except Exception:
            pass
        
        # Perturbation to escape local minima
        if time.time() < deadline - 20:
            rng = np.random.default_rng(42)
            pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
            pos_pert = np.clip(pos_pert, 0.01, 0.99)
            radii_pert = np.maximum(radii * 0.95, 0.005)
            
            if is_valid(pos_pert, radii_pert):
                score_pert = get_score(pos_pert, radii_pert)
                if score_pert > current_score * 0.9:  # Accept even slightly worse to escape
                    pos = pos_pert.copy()
                    radii = radii_pert.copy()
                    current_score = score_pert
    
    # Final validation
    if not is_valid(best_pos, best_radii):
        # Fallback: shrink radii until valid
        scale = 1.0
        while scale > 0.01 and not is_valid(best_pos, best_radii * scale):
            scale *= 0.9
        best_radii = best_radii * scale
    
    return np.column_stack((best_pos, best_radii))
