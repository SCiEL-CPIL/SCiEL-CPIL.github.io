"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random
import time


def _compute_inradii(centers, width, height):
    """
    Compute the inradius (max radius) for each circle center within the rectangle,
    using Voronoi tessellation logic.
    """
    n = len(centers)
    if n == 0:
        return np.array([])
    
    radii = np.zeros(n)
    
    for i in range(n):
        cx, cy = centers[i]
        # Distance to rectangle boundaries
        dist_to_boundary = min(cx, width - cx, cy, height - cy)
        
        # Distance to other centers: half the minimum distance to any other center
        min_dist_to_other = float('inf')
        for j in range(n):
            if i == j:
                continue
            dx = cx - centers[j][0]
            dy = cy - centers[j][1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < min_dist_to_other:
                min_dist_to_other = d
        
        # The inradius is the minimum of distance to boundary and half distance to nearest neighbor
        r = dist_to_boundary
        for j in range(n):
            if i == j:
                continue
            dx = cx - centers[j][0]
            dy = cy - centers[j][1]
            d = np.sqrt(dx * dx + dy * dy)
            half_d = d / 2.0
            if half_d < r:
                r = half_d
        
        radii[i] = r
    
    return radii


def _objective(x, width, height):
    """
    Objective function to minimize: negative sum of inradii.
    x is a flattened array of 2*N center coordinates.
    """
    centers = x.reshape(-1, 2)
    
    # Clamp to stay inside the rectangle with a small margin
    centers[:, 0] = np.clip(centers[:, 0], 0.001, width - 0.001)
    centers[:, 1] = np.clip(centers[:, 1], 0.001, height - 0.001)
    
    radii = _compute_inradii(centers, width, height)
    return -np.sum(radii)


def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4 (so w + h = 2).
    Maximize the sum of radii.
    """
    random.seed(42)
    np.random.seed(42)
    
    # Perimeter is 4, so 2*(w + h) = 4, thus w + h = 2
    # We'll try different aspect ratios
    
    best_sum = 0
    best_centers = None
    best_width = 1.0
    best_height = 1.0
    
    # Try several aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    
    for ar in aspect_ratios:
        # w + h = 2, w/h = ar => h(1 + ar) = 2 => h = 2/(1+ar), w = 2*ar/(1+ar)
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Try multiple initializations
        for init_iter in range(3):
            # Grid initialization with some randomness
            cols = 5
            rows = 5
            centers = []
            count = 0
            for i in range(cols):
                for j in range(rows):
                    if count >= 21:
                        break
                    cx = (i + 0.5) * w / cols + random.uniform(-0.05, 0.05)
                    cy = (j + 0.5) * h / rows + random.uniform(-0.05, 0.05)
                    # Clamp to stay inside
                    cx = max(0.01, min(w - 0.01, cx))
                    cy = max(0.01, min(h - 0.01, cy))
                    centers.append([cx, cy])
                    count += 1
                if count >= 21:
                    break
            
            if len(centers) < 21:
                continue
            
            centers = np.array(centers[:21])
            x0 = centers.flatten()
            
            # Use Nelder-Mead optimizer
            result = minimize(
                _objective,
                x0,
                args=(w, h),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            opt_centers = result.x.reshape(-1, 2)
            # Clamp to stay inside
            opt_centers[:, 0] = np.clip(opt_centers[:, 0], 0.001, w - 0.001)
            opt_centers[:, 1] = np.clip(opt_centers[:, 1], 0.001, h - 0.001)
            
            # Compute the sum of inradii
            radii = _compute_inradii(opt_centers, w, h)
            sum_radii = np.sum(radii)
            
            if sum_radii > best_sum:
                best_sum = sum_radii
                best_centers = opt_centers.copy()
                best_width = w
                best_height = h
    
    # Return the best packing
    # Format: (n, 3) array with [x, y, r]
    radii = _compute_inradii(best_centers, best_width, best_height)
    
    # Ensure circles don't overlap by slightly reducing radii if needed
    for i in range(N):
        for j in range(i + 1, N):
            dx = best_centers[i][0] - best_centers[j][0]
            dy = best_centers[i][1] - best_centers[j][1]
            d = np.sqrt(dx * dx + dy * dy)
            min_r = d / 2.0
            if radii[i] > min_r:
                radii[i] = min_r
            if radii[j] > min_r:
                radii[j] = min_r
    
    # Also ensure circles don't go outside the rectangle
    for i in range(N):
        cx, cy = best_centers[i]
        max_r = min(cx, best_width - cx, cy, best_height - cy)
        if radii[i] > max_r:
            radii[i] = max_r
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i][0] = best_centers[i][0]
        result[i][1] = best_centers[i][1]
        result[i][2] = radii[i]
    
    return result
# EVOLVE-BLOCK-END
