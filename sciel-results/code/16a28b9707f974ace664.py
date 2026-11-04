"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Initial parameters
    w = 1.0
    h = 2.0 - w
    
    # Initialize positions on a grid
    # 5x5 grid = 25 points. We need 21.
    # Use a 5x5 grid, remove 4 corners to get 21? Or just use first 21.
    # A hexagonal layout is better.
    # Let's use a simple grid for initialization and let the optimizer work.
    xs = np.linspace(0.2, 0.8, 5)
    ys = np.linspace(0.2, 0.8, 5)
    X = np.array([(x, y) for x in xs for y in ys][:N])
    
    # Initial radii
    R = np.full(N, 0.05)
    
    # Learning rates and momentum
    lr_X = 0.01
    lr_R = 0.005
    lr_w = 0.001
    momentum = 0.9
    
    vel_X = np.zeros_like(X)
    vel_R = np.zeros_like(R)
    vel_w = 0.0
    
    # Penalty weight
    lambda_pen = 50.0
    
    while time.time() < deadline:
        h = 2.0 - w
        
        # Compute pairwise distances
        # X is (N, 2)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        D = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        D_safe = np.where(D > 1e-6, D, 1e-6)
        
        # Overlaps: R_i + R_j - D_ij
        R_sum = R[:, np.newaxis] + R[np.newaxis, :]
        overlaps = np.maximum(0, R_sum - D)
        
        # Boundary violations
        # Left: R_i - X_i[:, 0]
        # Right: X_i[:, 0] + R_i - w
        # Bottom: R_i - X_i[:, 1]
        # Top: X_i[:, 1] + R_i - h  =>  X_i[:, 1] + R_i - (2-w)
        
        viol_left = np.maximum(0, R - X[:, 0])
        viol_right = np.maximum(0, X[:, 0] + R - w)
        viol_bottom = np.maximum(0, R - X[:, 1])
        viol_top = np.maximum(0, X[:, 1] + R - h)
        
        # Gradients for X
        grad_X = np.zeros_like(X)
        
        # From overlaps
        # d/dX_i sum_j max(0, R_i+R_j-D_ij)^2
        # = 2 * sum_j overlap_ij * (X_i - X_j) / D_ij
        # We can vectorize this.
        # overlap_grad_factor[i, j] = 2 * overlaps[i, j] / D_safe[i, j]
        # Note: overlaps is 0 if no overlap.
        
        # Create a matrix of forces
        # Force on i from j is proportional to (X_i - X_j)
        # Factor is 2 * overlaps[i, j] / D_safe[i, j]
        
        # Set diagonal to 0
        np.fill_diagonal(overlaps, 0)
        np.fill_diagonal(D_safe, 1.0) # Avoid div by 0 on diag, though overlaps is 0
        
        force_factors = 2.0 * overlaps / D_safe
        # Force on i is sum_j force_factors[i, j] * (X_i - X_j)
        # X_i is repeated N times.
        # Sum_j (X_i - X_j) * factor[i,j] = X_i * sum_j factor[i,j] - sum_j X_j * factor[i,j]
        
        sum_factors = np.sum(force_factors, axis=1)
        # (N, 1) * (N, 1) - (N, 2) @ (N, 2).T ? No.
        # sum_j X_j * factor[i, j] is (X.T @ force_factors)[i, :]
        
        grad_X_overlaps = X * sum_factors[:, np.newaxis] - (X.T @ force_factors).T
        grad_X += lambda_pen * grad_X_overlaps
        
        # From boundaries
        # Left: d/dx (R-x)^2 = -2(R-x)
        grad_X[:, 0] -= lambda_pen * 2.0 * viol_left
        # Right: d/dx (x+R-w)^2 = 2(x+R-w)
        grad_X[:, 0] += lambda_pen * 2.0 * viol_right
        # Bottom: d/dy (R-y)^2 = -2(R-y)
        grad_X[:, 1] -= lambda_pen * 2.0 * viol_bottom
        # Top: d/dy (y+R-h)^2 = 2(y+R-h)
        grad_X[:, 1] += lambda_pen * 2.0 * viol_top
        
        # Gradients for R
        grad_R = np.zeros_like(R)
        
        # Objective: -sum(R) -> -1
        grad_R -= 1.0
        
        # From overlaps
        # d/dR_i sum_j max(0, R_i+R_j-D_ij)^2
        # = 2 * sum_j overlaps[i, j]
        # (Since d/dR_i (R_i+R_j-D)^2 = 2(R_i+R_j-D))
        grad_R += lambda_pen * 2.0 * np.sum(overlaps, axis=1)
        
        # From boundaries
        grad_R += lambda_pen * 2.0 * (viol_left + viol_right + viol_bottom + viol_top)
        
        # Gradients for w
        grad_w = 0.0
        # Only right and top boundaries depend on w.
        # Right: d/dw (x+R-w)^2 = -2(x+R-w)
        grad_w -= lambda_pen * 2.0 * np.sum(viol_right)
        # Top: h = 2-w. Viol = y+R-h = y+R-(2-w) = y+R-2+w.
        # d/dw (y+R-2+w)^2 = 2(y+R-2+w)
        grad_w += lambda_pen * 2.0 * np.sum(viol_top)
        
        # Update velocities
        vel_X = momentum * vel_X - lr_X * grad_X
        vel_R = momentum * vel_R - lr_R * grad_R
        vel_w = momentum * vel_w - lr_w * grad_w
        
        # Update positions
        X += vel_X
        R += vel_R
        w += vel_w
        
        # Constraints
        # R must be positive
        R = np.maximum(R, 0.01)
        
        # w must be in [0.5, 1.5]
        w = np.clip(w, 0.5, 1.5)
        h = 2.0 - w
        
        # X must be in [R, w-R] x [R, h-R]
        X[:, 0] = np.clip(X[:, 0], R, w - R)
        X[:, 1] = np.clip(X[:, 1], R, h - R)
        
        # Adaptive learning rate?
        # If overlaps are large, maybe reduce lr?
        # For now, keep it simple.
        
        # Check for convergence or stagnation?
        # Just run until deadline.

    # Final Polish
    # Ensure no overlaps by shrinking radii if necessary
    # This is a safety measure.
    for _ in range(10):
        h = 2.0 - w
        # Compute max possible radii
        # This is a heuristic: just check overlaps and shrink if any.
        any_overlap = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(X[i] - X[j])
                req = R[i] + R[j]
                if d < req:
                    # Shrink both
                    excess = req - d
                    R[i] -= excess / 2
                    R[j] -= excess / 2
                    any_overlap = True
            # Boundaries
            if X[i][0] < R[i]:
                R[i] = X[i][0]
                any_overlap = True
            if X[i][0] + R[i] > w:
                R[i] = w - X[i][0]
                any_overlap = True
            if X[i][1] < R[i]:
                R[i] = X[i][1]
                any_overlap = True
            if X[i][1] + R[i] > h:
                R[i] = h - X[i][1]
                any_overlap = True
        R = np.maximum(R, 0.0)
        if not any_overlap:
            break

    return np.column_stack([X[:, 0], X[:, 1], R])

# EVOLVE-BLOCK-END
