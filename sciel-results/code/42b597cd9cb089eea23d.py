"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4, maximizing sum of radii.
    
    Strategy:
    - Fix aspect ratio w/h = r (try several).
    - For a given aspect ratio, the rectangle has width w = 2*r/(1+r), height h = 2/(1+r).
    - Use a smooth penalty-based optimization with L-BFGS-B to maximize sum of radii
      subject to non-overlap and boundary constraints.
    - The objective is: maximize sum(r_i) - penalty * (overlap_violations + boundary_violations)
    - Use log-sum-exp style smooth penalties for better gradient behavior.
    - Multi-start with different random initializations.
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Try several aspect ratios
    aspect_ratios = [1.0, 1.5, 2.0, 2.5, 3.0, 0.75, 0.66, 1.25, 1.8, 2.2]
    
    best_sum = 0.0
    best_result = None
    
    # Initial guess: grid-like arrangement
    def initial_grid(w, h, n_circles):
        """Create a grid-like initial arrangement."""
        cols = int(np.ceil(np.sqrt(n_circles * w / h)))
        rows = int(np.ceil(n_circles / cols))
        dx = w / cols
        dy = h / rows
        centers = []
        radii = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= n_circles:
                    break
                cx = (i + 0.5) * dx
                cy = (j + 0.5) * dy
                centers.append([cx, cy])
                # Initial radius estimate
                r_est = min(dx, dy) * 0.45
                radii.append(r_est)
                count += 1
        return np.array(centers), np.array(radii)
    
    def objective_and_grad(x, w, h):
        """
        x = [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
        Maximize sum(r_i) subject to:
        - Circles don't overlap: dist(ci, cj) >= ri + rj
        - Circles in boundary: ri <= xi <= w-ri, ri <= yi <= h-ri
        
        We minimize: -sum(r_i) + penalty * violation
        """
        n = N
        centers = x[0:2*n].reshape(n, 2)
        radii = x[2*n:3*n]
        
        # Ensure radii are positive
        radii = np.maximum(radii, 1e-10)
        
        # Sum of radii (we want to maximize, so minimize negative)
        sum_r = np.sum(radii)
        
        # Penalty for overlaps
        # For each pair (i,j), violation = max(0, ri + rj - dist(i,j))
        # Use smooth approximation: softplus(ri + rj - dist)
        penalty = 0.0
        grad_penalty = np.zeros_like(x)
        
        # Boundary violations
        for i in range(n):
            xi, yi = centers[i]
            ri = radii[i]
            # Left/right boundary
            viol_lr = max(0, ri - xi) + max(0, ri - (w - xi))
            # Top/bottom boundary
            viol_tb = max(0, ri - yi) + max(0, ri - (h - yi))
            total_viol = viol_lr + viol_tb
            if total_viol > 0:
                penalty += total_viol ** 2
                # Gradient
                dxi = 0
                dyi = 0
                dri = 0
                if ri > xi:
                    dxi -= 1
                    dri += 1
                if ri > w - xi:
                    dxi += 1
                    dri += 1
                if ri > yi:
                    dyi -= 1
                    dri += 1
                if ri > h - yi:
                    dyi += 1
                    dri += 1
                grad_penalty[2*i] += 2 * total_viol * dxi
                grad_penalty[2*i+1] += 2 * total_viol * dyi
                grad_penalty[2*n+i] += 2 * total_viol * dri
        
        # Pairwise overlaps
        for i in range(n):
            for j in range(i+1, n):
                di = centers[i] - centers[j]
                dist = np.linalg.norm(di)
                if dist < 1e-10:
                    dist = 1e-10
                    di = np.array([1e-10, 0])
                ri = radii[i]
                rj = radii[j]
                overlap = ri + rj - dist
                if overlap > 0:
                    penalty += overlap ** 2
                    # Gradient
                    # d/dxi of (ri + rj - dist)^2 = 2*(ri+rj-dist) * (-(xi-xj)/dist)
                    factor = 2 * overlap / dist
                    grad_penalty[2*i] += factor * (xi - xj)
                    grad_penalty[2*i+1] += factor * (yi - yj)
                    grad_penalty[2*j] -= factor * (xi - xj)
                    grad_penalty[2*j+1] -= factor * (yi - yj)
                    grad_penalty[2*n+i] += 2 * overlap
                    grad_penalty[2*n+j] += 2 * overlap
        
        # Objective: -sum_r + penalty_weight * penalty
        penalty_weight = 100.0
        obj = -sum_r + penalty_weight * penalty
        
        # Gradient
        grad = np.zeros_like(x)
        grad[2*n:3*n] = -1.0  # d(-sum_r)/dr_i = -1
        grad += penalty_weight * grad_penalty
        
        return obj, grad
    
    def is_valid(result, w, h, tol=1e-6):
        """Check if the result is valid (no overlaps, within bounds)."""
        centers = result[0:2*N].reshape(N, 2)
        radii = result[2*N:3*N]
        
        if np.any(radii < -tol):
            return False
        
        # Check boundaries
        for i in range(N):
            xi, yi = centers[i]
            ri = radii[i]
            if ri < -tol or xi < ri - tol or xi > w - ri + tol or yi < ri - tol or yi > h - ri + tol:
                return False
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j] - tol:
                    return False
        
        return True
    
    # Try each aspect ratio with multiple restarts
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        # Multiple restarts
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Initial guess
            centers0, radii0 = initial_grid(w, h, N)
            
            # Add some randomness for multi-start
            if restart > 0:
                np.random.seed(restart * 12345 + int(ar * 100))
                centers0 = centers0 + np.random.randn(N, 2) * 0.05 * min(w, h)
                centers0[:, 0] = np.clip(centers0[:, 0], 0.05 * min(radii0), w - 0.05 * min(radii0))
                centers0[:, 1] = np.clip(centers0[:, 1], 0.05 * min(radii0), h - 0.05 * min(radii0))
            
            x0 = np.concatenate([centers0.flatten(), radii0])
            
            # Bounds
            bounds = []
            for i in range(N):
                bounds.append((1e-10, w))  # xi
                bounds.append((1e-10, h))  # yi
            for i in range(N):
                bounds.append((1e-10, min(w, h) / 2))  # ri
            
            # Optimize
            try:
                res = minimize(
                    lambda x: objective_and_grad(x, w, h)[0],
                    x0,
                    jac=lambda x: objective_and_grad(x, w, h)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < 1e6:
                    result = res.x
                    sum_r = np.sum(result[2*N:3*N])
                    
                    if sum_r > best_sum:
                        # Verify validity
                        if is_valid(result, w, h):
                            best_sum = sum_r
                            best_result = result.copy()
                            best_w, best_h = w, h
                        else:
                            # Try to repair: clip radii to fit
                            repaired = result.copy()
                            centers_r = repaired[0:2*N].reshape(N, 2)
                            radii_r = repaired[2*N:3*N]
                            
                            # Clip radii to fit in bounds
                            for i in range(N):
                                xi, yi = centers_r[i]
                                ri = radii_r[i]
                                max_r = min(xi, w - xi, yi, h - yi)
                                if ri > max_r:
                                    radii_r[i] = max_r
                            
                            # Check if still valid
                            if is_valid(repaired, w, h):
                                sum_r_repaired = np.sum(radii_r)
                                if sum_r_repaired > best_sum:
                                    best_sum = sum_r_repaired
                                    best_result = repaired.copy()
                                    best_w, best_h = w, h
            except Exception as e:
                pass
    
    # If we didn't find anything valid, fall back to a simple grid
    if best_result is None:
        # Simple 5x4 grid (20 circles) + 1 extra, or 4x5
        # Use a 5x5 grid but only 21 circles
        cols, rows = 5, 5
        w = 2.0  # square for simplicity, perimeter = 4 => w=h=1
        # Actually perimeter 4 => w+h=2, so if square, w=h=1
        w = 1.0
        h = 1.0
        dx = w / cols
        dy = h / rows
        cells = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                cx = (i + 0.5) * dx
                cy = (j + 0.5) * dy
                r = min(dx, dy) * 0.48
                cells.append([cx, cy, r])
                count += 1
        return np.array(cells)
    
    # Extract result
    centers = best_result[0:2*N].reshape(N, 2)
    radii = best_result[2*N:3*N]
    result = np.column_stack([centers, radii])
    return result

# EVOLVE-BLOCK-END
