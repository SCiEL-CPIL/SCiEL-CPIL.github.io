"""Pack 50 equal circles in a unit circle using L-BFGS-B on overlap energy."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

def _initial_positions(n=50):
    """Generate a good initial hexagonal packing inside the unit disk."""
    # Use a hexagonal grid and pick the best n points
    spacing = 0.22
    candidates = []
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0 - spacing / 2.0:
                candidates.append((x, y, r))
    
    candidates.sort(key=lambda t: t[2])
    points = np.array([(c[0], c[1]) for c in candidates[:n]])
    return points

def _energy_and_radius(pos):
    """Compute overlap energy and current radius for a configuration."""
    n = pos.shape[0]
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Boundary distances
    boundary = 1.0 - np.sqrt(np.sum(pos ** 2, axis=1))
    
    # Current radius is min of half-pairwise-dist and boundary
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    min_boundary = np.min(boundary)
    r = min(min_pair, min_boundary)
    
    # Energy: sum of squared violations (positive when overlapping)
    # For pairs: violation = r - dist/2 if dist < 2r
    # For boundary: violation = r - boundary if boundary < r
    # We use a soft penalty: sum of (max(0, 2r - dist))^2 + sum of (max(0, r - boundary))^2
    # But we want to maximize r, so we minimize negative r plus penalties
    
    # Actually, let's use a different approach:
    # Minimize: sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, r - b_i)^2 - r
    # But r is a variable... Let's use a fixed-r approach with iterative r update.
    
    # Simpler: just compute the "badness" = sum of overlaps assuming r is the current min
    pair_viol = np.maximum(0, 2 * r - dist)
    np.fill_diagonal(pair_viol, 0)
    pair_energy = np.sum(pair_viol ** 2)
    
    boundary_viol = np.maximum(0, r - boundary)
    boundary_energy = np.sum(boundary_viol ** 2)
    
    return pair_energy + boundary_energy, r

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Initial position
    pos = _initial_positions(N)
    
    # Compute initial radius
    _, best_r = _energy_and_radius(pos)
    best_pos = pos.copy()
    
    # Iterative optimization: increase r gradually, minimize overlap energy
    # Start with a small r and increase
    r_current = best_r * 0.95
    r_target = 0.126  # slightly above known best
    
    while time.time() < deadline:
        # Define objective function for L-BFGS-B
        def objective(x):
            p = x.reshape(N, 2)
            # Pairwise distances
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            
            # Boundary
            boundary = 1.0 - np.sqrt(np.sum(p ** 2, axis=1))
            
            # Penalties for violations at current r
            pair_viol = np.maximum(0, 2 * r_current - dist)
            np.fill_diagonal(pair_viol, 0)
            pair_energy = np.sum(pair_viol ** 2)
            
            boundary_viol = np.maximum(0, r_current - boundary)
            boundary_energy = np.sum(boundary_viol ** 2)
            
            # Small regularization to keep points spread
            return pair_energy + boundary_energy
        
        # Gradient (approximate for L-BFGS-B)
        def gradient(x):
            p = x.reshape(N, 2)
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2)) + 1e-10
            np.fill_diagonal(dist, np.inf)
            
            boundary = 1.0 - np.sqrt(np.sum(p ** 2, axis=1))
            grad = np.zeros_like(x)
            
            # Pair gradients
            for i in range(N):
                for j in range(i + 1, N):
                    d = dist[i, j]
                    if d < 2 * r_current:
                        factor = 2 * (2 * r_current - d) / d
                        grad[i * 2:i * 2 + 2] += factor * (p[i] - p[j])
                        grad[j * 2:j * 2 + 2] -= factor * (p[i] - p[j])
            
            # Boundary gradients
            for i in range(N):
                if boundary[i] < r_current:
                    factor = 2 * (r_current - boundary[i])
                    norm = np.linalg.norm(p[i]) + 1e-10
                    grad[i * 2:i * 2 + 2] += factor * p[i] / norm
            
            return grad
        
        # Optimize
        result = minimize(
            objective,
            pos.ravel(),
            method='L-BFGS-B',
            jac=gradient,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
        )
        
        if result.success or result.fun < 1e-8:
            pos = result.x.reshape(N, 2)
            _, r_new = _energy_and_radius(pos)
            
            if r_new > best_r:
                best_r = r_new
                best_pos = pos.copy()
            
            # Increase r
            r_current = r_new * 1.01
            if r_current > r_target:
                r_current = r_target
        
        # Check time
        if time.time() > deadline - 10.0:
            break
        
        # If energy is very low, try to push r higher
        _, r_check = _energy_and_radius(pos)
        if r_check > best_r:
            best_r = r_check
            best_pos = pos.copy()
    
    # Final polish: use the best position found
    # Do one more optimization at the best radius
    if time.time() < deadline - 5.0:
        def objective_final(x):
            p = x.reshape(N, 2)
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            boundary = 1.0 - np.sqrt(np.sum(p ** 2, axis=1))
            
            pair_viol = np.maximum(0, 2 * best_r - dist)
            np.fill_diagonal(pair_viol, 0)
            pair_energy = np.sum(pair_viol ** 2)
            
            boundary_viol = np.maximum(0, best_r - boundary)
            boundary_energy = np.sum(boundary_viol ** 2)
            
            return pair_energy + boundary_energy
        
        def gradient_final(x):
            p = x.reshape(N, 2)
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2)) + 1e-10
            np.fill_diagonal(dist, np.inf)
            
            boundary = 1.0 - np.sqrt(np.sum(p ** 2, axis=1))
            grad = np.zeros_like(x)
            
            for i in range(N):
                for j in range(i + 1, N):
                    d = dist[i, j]
                    if d < 2 * best_r:
                        factor = 2 * (2 * best_r - d) / d
                        grad[i * 2:i * 2 + 2] += factor * (p[i] - p[j])
                        grad[j * 2:j * 2 + 2] -= factor * (p[i] - p[j])
            
            for i in range(N):
                if boundary[i] < best_r:
                    factor = 2 * (best_r - boundary[i])
                    norm = np.linalg.norm(p[i]) + 1e-10
                    grad[i * 2:i * 2 + 2] += factor * p[i] / norm
            
            return grad
        
        result = minimize(
            objective_final,
            best_pos.ravel(),
            method='L-BFGS-B',
            jac=gradient_final,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 8000}
        )
        
        if result.fun < 1e-10:
            pos = result.x.reshape(N, 2)
            _, r_final = _energy_and_radius(pos)
            if r_final > best_r:
                best_r = r_final
                best_pos = pos.copy()
    
    # Verify validity
    pos = best_pos
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pos ** 2, axis=1))
    
    r_min = min(np.min(dist) / 2.0, np.min(boundary))
    
    # Ensure all constraints are satisfied
    if r_min <= 0:
        # Fallback to initial
        return _initial_positions(N)
    
    return pos

# EVOLVE-BLOCK-END
