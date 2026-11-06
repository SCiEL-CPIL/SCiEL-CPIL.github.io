"""Packing 50 equal circles in a unit square: hexagonal lattice init + force-directed relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s kill

def _score(centers):
    """Compute the minimum radius: min of (min pair dist / 2, min boundary clearance)."""
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary clearance
    # Square is [-0.5, 0.5]^2
    clearance_x = np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5)
    clearance_y = np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    min_boundary = np.min(np.minimum(clearance_x, clearance_y))
    
    return min(min_pair, min_boundary)

def _hexagonal_init():
    """Create a hexagonal lattice packing with variable row lengths summing to 50."""
    # Known good structure for 50 circles: rows of 7,6,7,6,7,6,7,4
    row_counts = [7, 6, 7, 6, 7, 6, 7, 4]
    n_rows = len(row_counts)
    
    # Compute spacing
    # In hexagonal packing, horizontal spacing = 2r, vertical spacing = sqrt(3)*r
    # We want to fit within [-0.5, 0.5] in both x and y
    
    # Let's parameterize by radius r and find best fit
    # For a row with k circles, the x-extent is (k-1)*2r + 2r = k*2r (from center of first to center of last is (k-1)*2r, plus r on each side)
    # Actually, the centers span from x_min to x_max where x_max - x_min = (k-1)*2r
    # The full extent including radius is (k-1)*2r + 2r = 2r*k
    
    # Let's just place them and then optimize
    # Start with a reasonable estimate
    r = 0.07
    
    centers = []
    for i, k in enumerate(row_counts):
        y = -0.5 + r + i * np.sqrt(3) * r
        # Center the row horizontally
        x_start = -(k - 1) * r
        for j in range(k):
            x = x_start + j * 2 * r
            centers.append([x, y])
    
    centers = np.array(centers)
    
    # Check if it fits, if not scale
    max_x = np.max(np.abs(centers[:, 0])) + r
    max_y = np.max(np.abs(centers[:, 1])) + r
    scale = min(0.5 / max_x, 0.5 / max_y) if max_x > 0.5 or max_y > 0.5 else 1.0
    centers *= scale
    
    return centers

def _force_relaxation(centers, max_iter=5000, lr=0.001):
    """Simple force-directed relaxation to maximize minimum distance."""
    centers = centers.copy()
    n = len(centers)
    
    for it in range(max_iter):
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion (soft sphere model)
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.sqrt(np.sum(diff ** 2))
                if dist < 1e-10:
                    diff = np.random.randn(2) * 1e-6
                    dist = np.sqrt(np.sum(diff ** 2))
                # Repulsive force, stronger when closer
                force_mag = 1.0 / (dist ** 2 + 1e-6)
                forces[i] += force_mag * diff / dist
                forces[j] -= force_mag * diff / dist
        
        # Boundary forces (push away from walls)
        for i in range(n):
            for dim in range(2):
                if centers[i, dim] < -0.45:
                    forces[i, dim] += 5.0 * (-0.45 - centers[i, dim])
                if centers[i, dim] > 0.45:
                    forces[i, dim] += 5.0 * (0.45 - centers[i, dim])
        
        # Update positions
        centers += lr * forces
        
        # Clamp to square
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # Decay learning rate
        lr *= 0.999
    
    return centers

def _gradient_ascent(centers, max_iter=2000, lr=0.0005):
    """Gradient ascent on the score function using numerical gradients."""
    centers = centers.copy()
    n = len(centers)
    best_score = _score(centers)
    best_centers = centers.copy()
    
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        # Compute gradient numerically
        grad = np.zeros_like(centers)
        eps = 1e-5
        base_score = best_score
        
        for i in range(n):
            for d in range(2):
                old = centers[i, d]
                centers[i, d] = old + eps
                s_plus = _score(centers)
                centers[i, d] = old - eps
                s_minus = _score(centers)
                centers[i, d] = old
                grad[i, d] = (s_plus - s_minus) / (2 * eps)
        
        # Normalize gradient
        grad_norm = np.sqrt(np.sum(grad ** 2))
        if grad_norm < 1e-12:
            break
        grad /= grad_norm
        
        # Update
        centers += lr * grad
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        new_score = _score(centers)
        if new_score > best_score:
            best_score = new_score
            best_centers = centers.copy()
        
        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            lr *= 0.9
    
    return best_centers

def _scipy_optimize(centers, max_iter=500):
    """Use scipy minimize on the negative score."""
    x0 = centers.flatten()
    n = len(centers)
    
    def neg_score(x):
        c = x.reshape(n, 2)
        c[:, 0] = np.clip(c[:, 0], -0.5, 0.5)
        c[:, 1] = np.clip(c[:, 1], -0.5, 0.5)
        return -_score(c)
    
    result = minimize(neg_score, x0, method='L-BFGS-B', 
                      bounds=[(-0.5, 0.5)] * (2 * n),
                      options={'maxiter': max_iter, 'gtol': 1e-8})
    
    if result.success or result.fun < neg_score(x0):
        c = result.x.reshape(n, 2)
        c[:, 0] = np.clip(c[:, 0], -0.5, 0.5)
        c[:, 1] = np.clip(c[:, 1], -0.5, 0.5)
        return c
    
    return centers

def construct():
    start_time = time.time()
    
    # Initial hexagonal configuration
    centers = _hexagonal_init()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Force-directed relaxation
    centers = _force_relaxation(centers, max_iter=3000, lr=0.0005)
    s = _score(centers)
    if s > best_score:
        best_score = s
        best_centers = centers.copy()
    
    if time.time() > DEADLINE - 20:
        return best_centers
    
    # Gradient ascent
    centers = _gradient_ascent(best_centers, max_iter=500, lr=0.0003)
    s = _score(centers)
    if s > best_score:
        best_score = s
        best_centers = centers.copy()
    
    if time.time() > DEADLINE - 15:
        return best_centers
    
    # Scipy optimization
    centers = _scipy_optimize(best_centers, max_iter=300)
    s = _score(centers)
    if s > best_score:
        best_score = s
        best_centers = centers.copy()
    
    # Final validation
    best_centers[:, 0] = np.clip(best_centers[:, 0], -0.5, 0.5)
    best_centers[:, 1] = np.clip(best_centers[:, 1], -0.5, 0.5)
    
    return best_centers
