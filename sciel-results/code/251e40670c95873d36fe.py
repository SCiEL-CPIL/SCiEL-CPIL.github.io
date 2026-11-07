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
        history = []  # list of (x, f) tuples for PCA
        
        def count_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return 1e10
            calls += 1
            f = func(x)
            history.append((x.copy(), f))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Coarse search with random restarts and local coordinate descent
        # We'll do a few rounds of "sample, then refine along coordinates"
        
        # Initial random samples to get a rough idea
        n_init = 20
        for _ in range(n_init):
            if calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = count_func(x)
        
        # Phase 2: Iterative coordinate descent with PCA rotation
        # We maintain a rotation matrix R. 
        # At each major step:
        # 1. Project recent history onto principal components.
        # 2. Update R to align with these components.
        # 3. Perform coordinate descent in the rotated frame.
        
        R = np.eye(dim)
        
        # We will do a fixed number of iterations, or until budget is exhausted
        # Each iteration consists of:
        # - PCA update (if enough points)
        # - Coordinate descent steps
        
        max_iter = 50
        cd_steps_per_iter = 50  # Max function calls for CD in one iteration
        
        for it in range(max_iter):
            if time.time() > deadline or calls >= budget:
                break
            
            # Update PCA rotation
            if len(history) >= dim * 2:
                # Take the last min(100, len(history)) points
                n_pca = min(100, len(history))
                recent = history[-n_pca:]
                X = np.array([h[0] for h in recent])
                F = np.array([h[1] for h in recent])
                
                # Center the data
                X_mean = X.mean(axis=0)
                X_centered = X - X_mean
                
                # Compute covariance
                try:
                    cov = np.cov(X_centered, rowvar=False)
                    # PCA
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    # Sort by eigenvalues descending
                    idx = np.argsort(eigenvalues)[::-1]
                    eigenvectors = eigenvectors[:, idx]
                    eigenvalues = eigenvalues[idx]
                    
                    # Update R to be the eigenvectors
                    # We want the first axis to be the direction of maximum variance (ridge direction)
                    # Actually, for a sharp ridge, we want to align the "tight" direction with an axis.
                    # The ridge is flat in one direction and sharp in another.
                    # The direction of steepest descent is the "sharp" direction.
                    # In PCA, the direction of minimum variance is often the ridge direction (flat).
                    # So we want to align the last axis (or first?) with the flat direction?
                    # Let's align the last axis with the direction of minimum variance.
                    # Or, more simply, just use the eigenvectors as the new basis.
                    R = eigenvectors
                except:
                    pass
            
            # Coordinate descent in the rotated frame
            # Current best point in original space: best_x
            # Convert to rotated space: y = R.T @ x
            y = R.T @ best_x
            
            # Perform coordinate descent on y
            # We'll do a simple line search or fixed step size
            step_size = 0.5
            for step in range(cd_steps_per_iter):
                if calls >= budget or time.time() > deadline:
                    break
                
                improved = False
                for d in range(dim):
                    # Try positive and negative steps along dimension d
                    # Use a small line search
                    y_trial = y.copy()
                    # Try a few step sizes
                    for s in [step_size, -step_size, step_size*0.5, -step_size*0.5, step_size*2, -step_size*2]:
                        y_trial[d] = y[d] + s
                        # Convert back to original space
                        x_trial = R @ y_trial
                        # Clip to box
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = count_func(x_trial)
                        
                        if f_trial < best_f:
                            # Accept
                            y = y_trial.copy()
                            best_x = x_trial.copy()
                            improved = True
                            break
                    # If improved, we can break the dimension loop early or continue
                    # Let's continue to explore other dimensions in the same pass
                    # But if we found a good improvement, we might want to re-center?
                    # For simplicity, just continue.
                
                if not improved:
                    # No improvement in this pass, break
                    break
            
            # If no improvement in the entire iteration, we might be stuck
            # But we don't have a direct signal for "stuck" other than no improvement in CD
            # We can add a random perturbation if stuck
            # For now, just continue to next PCA update

        return best_x, best_f
# EVOLVE-BLOCK-END
