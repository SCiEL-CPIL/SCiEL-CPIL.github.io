"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Adaptive Coordinate Descent with PCA-based rotation.
    Designed for Sharp Ridge (f13) where the landscape has a narrow valley.
    Standard methods struggle to align with the ridge. This method:
    1. Uses coordinate descent which is efficient along axes.
    2. Dynamically rotates the coordinate system using PCA of recent evaluations
       to align one axis with the ridge direction and another with the normal.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        # History for PCA: list of (x, f)
        # We store points to compute covariance of positions weighted or just positions
        # For ridge alignment, we want the principal direction of the "cloud" of points
        # that are close to the optimum? Or just the recent trajectory?
        # Let's use the last K points.
        history = []
        K = 20  # Window size for PCA
        
        def count_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return 1e10
            if time.time() > deadline:
                return 1e10
            calls += 1
            xc = np.clip(x, -5, 5)
            f_val = func(xc)
            if f_val < best_f:
                best_f = f_val
                best_x = xc.copy()
            history.append((xc, f_val))
            if len(history) > K:
                history.pop(0)
            return f_val

        # Initial point
        # For f13, the minimum is often near the center or shifted.
        # Start at 0.
        x = np.zeros(dim)
        f_x = count_func(x)
        
        # Initial step size
        # Ridge functions often have a very flat direction and a steep normal direction.
        # We need a step size that allows movement along the ridge but not too large
        # to jump over the valley.
        # Let's start with a moderate step.
        step_sizes = np.ones(dim) * 1.0
        
        # Rotation matrix (identity initially)
        R = np.eye(dim)
        
        # Adaptive line search parameters
        # We will perform a 1D search along each rotated coordinate.
        
        while calls < budget and time.time() < deadline:
            # --- PCA Update ---
            # If we have enough history, update the rotation matrix
            if len(history) >= max(10, dim + 1):
                try:
                    # Extract points
                    pts = np.array([h[0] for h in history])
                    # Center the points
                    mean_pt = np.mean(pts, axis=0)
                    centered = pts - mean_pt
                    
                    # Compute covariance
                    cov = np.cov(centered, rowvar=False)
                    # Ensure symmetry
                    cov = (cov + cov.T) / 2
                    
                    # PCA
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    
                    # Sort by eigenvalue descending
                    idx = np.argsort(eigvals)[::-1]
                    eigvals = eigvals[idx]
                    eigvecs = eigvecs[:, idx]
                    
                    # The first eigenvector is the direction of maximum variance.
                    # In a ridge function, the "cloud" of recent evaluations might
                    # spread along the ridge if the optimizer is moving along it,
                    # or perpendicular if it's trying to cross it.
                    # Actually, for a sharp ridge, the function value changes rapidly
                    # in the normal direction and slowly in the ridge direction.
                    # If we are converging, the points will cluster.
                    # A better heuristic: Align the FIRST axis with the direction of
                    # steepest ascent/descent? No, we don't have gradients.
                    
                    # Let's assume the "ridge" is the direction where the function
                    # is relatively flat compared to others.
                    # However, PCA on positions finds the direction of spread.
                    # If the optimizer is oscillating across the ridge, the spread
                    # will be in the normal direction.
                    # If it is sliding along the ridge, the spread will be in the
                    # ridge direction.
                    
                    # Strategy: Use the eigenvector with the LARGEST eigenvalue as
                    # the primary search direction (likely the ridge or the main
                    # axis of the current search region).
                    # We rotate the coordinate system so that the new x-axis
                    # aligns with this principal direction.
                    
                    new_R = eigvecs
                    # Ensure right-handedness if needed, but for coordinate descent
                    # order doesn't matter as much as alignment.
                    R = new_R
                    
                except Exception:
                    pass # Keep old R if PCA fails

            # --- Coordinate Descent in Rotated Space ---
            # We iterate over each dimension in the rotated frame.
            # For each dimension i, we search along the i-th column of R.
            
            improved = False
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Direction vector in original space
                d = R[:, i]
                
                # Current step size for this dimension
                step = step_sizes[i]
                
                # Try positive and negative steps
                # Simple adaptive line search:
                # Try x + step * d
                x_trial = x + step * d
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = count_func(x_trial)
                
                if f_trial < f_x:
                    x = x_trial
                    f_x = f_trial
                    improved = True
                    # Increase step size slightly if we improved
                    step_sizes[i] *= 1.2
                    # But cap it
                    step_sizes[i] = min(step_sizes[i], 5.0)
                else:
                    # Try negative step
                    x_trial_neg = x - step * d
                    x_trial_neg = np.clip(x_trial_neg, -5, 5)
                    f_trial_neg = count_func(x_trial_neg)
                    
                    if f_trial_neg < f_x:
                        x = x_trial_neg
                        f_x = f_trial_neg
                        improved = True
                        step_sizes[i] *= 1.2
                        step_sizes[i] = min(step_sizes[i], 5.0)
                    else:
                        # No improvement, shrink step size
                        step_sizes[i] *= 0.7
                        step_sizes[i] = max(step_sizes[i], 1e-4)
            
            # If no improvement in a full cycle, we might be stuck or converged.
            # To avoid infinite loops on flat spots, we can add a small random perturbation
            # or reduce step sizes further.
            if not improved:
                # Global shrink
                step_sizes *= 0.5
                if np.all(step_sizes < 1e-3):
                    # Considered converged, break to save time?
                    # Or just break.
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
