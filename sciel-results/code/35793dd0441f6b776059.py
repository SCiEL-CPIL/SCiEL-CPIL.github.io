"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Two-phase hybrid: deterministic radial sweep then aggressive (1+1) ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Track best so far
        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return np.inf
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                return np.inf
            calls += 1
            return f

        # Phase 1: dense deterministic radial sweep from origin
        # Discus is extremely anisotropic; we sweep along many directions
        # to quickly find the "needle" axis and a good starting point.
        
        # Create a set of deterministic directions:
        # - coordinate axes (positive and negative)
        # - diagonal directions in each coordinate plane
        # - a few random directions for robustness
        
        directions = []
        
        # Coordinate axes
        for i in range(dim):
            v = np.zeros(dim)
            v[i] = 1.0
            directions.append(v)
            v[i] = -1.0
            directions.append(v)
        
        # Plane diagonals
        for i in range(dim):
            for j in range(i+1, dim):
                v = np.zeros(dim)
                v[i] = 1.0
                v[j] = 1.0
                v /= np.linalg.norm(v)
                directions.append(v)
                v[i] = -1.0
                v[j] = 1.0
                v /= np.linalg.norm(v)
                directions.append(v)
                v[i] = 1.0
                v[j] = -1.0
                v /= np.linalg.norm(v)
                directions.append(v)
                v[i] = -1.0
                v[j] = -1.0
                v /= np.linalg.norm(v)
                directions.append(v)
        
        # Add some random directions
        for _ in range(20):
            v = rng.normal(size=dim)
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                v /= norm
                directions.append(v)
        
        # Radial sweep: for each direction, evaluate at several radii
        # Start with coarse sweep, then refine around best direction
        radii_coarse = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        
        best_dir = None
        best_radius = 0.0
        
        for d in directions:
            for r in radii_coarse:
                if calls >= budget - 50:  # Reserve calls for Phase 2
                    break
                x = r * d
                f = safe_call(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    best_dir = d.copy()
                    best_radius = r
        
        # Refine around best direction with finer radii
        if best_dir is not None and calls < budget - 30:
            # Fine sweep around best_radius
            fine_radii = np.linspace(max(0.1, best_radius - 1.0), min(4.9, best_radius + 1.0), 15)
            for r in fine_radii:
                if calls >= budget - 30:
                    break
                x = r * best_dir
                f = safe_call(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    best_radius = r
        
        # Also try a few points near origin in case the minimum is close
        if calls < budget - 20:
            for scale in [0.1, 0.3, 0.7]:
                for i in range(dim):
                    x = np.zeros(dim)
                    x[i] = scale
                    f = safe_call(x)
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    x[i] = -scale
                    f = safe_call(x)
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
        
        # Phase 2: Aggressive (1+1) ES with adaptive step size
        # High mutation rate to quickly explore and exploit the needle axis
        
        # Initialize step size based on best_radius
        if best_radius > 0.1:
            step = max(0.1, best_radius * 0.3)
        else:
            step = 1.0
        
        # (1+1) ES loop
        success_count = 0
        failure_count = 0
        target_rate = 0.2  # Aim for 20% success rate for fast convergence
        
        while calls < budget:
            # Generate mutation
            mutation = rng.normal(scale=step, size=dim)
            x_new = best_x + mutation
            x_new = np.clip(x_new, -5.0, 5.0)
            
            f_new = safe_call(x_new)
            if calls >= budget:
                break
            
            if f_new < best_f:
                # Success
                best_f = f_new
                best_x = x_new.copy()
                success_count += 1
                # Increase step slightly on success
                step *= 1.1
            else:
                # Failure
                failure_count += 1
                # Decrease step on failure
                step *= 0.9
            
            # Adaptive step size adjustment based on success rate
            total = success_count + failure_count
            if total > 20:
                current_rate = success_count / total
                if current_rate > target_rate:
                    step *= 1.05
                elif current_rate < target_rate:
                    step *= 0.95
            
            # Keep step within bounds
            step = np.clip(step, 0.01, 5.0)
            
            # Reset counters periodically to adapt to changing landscape
            if total > 100:
                success_count = 0
                failure_count = 0
        
        return best_x, best_f
# EVOLVE-BLOCK-END
