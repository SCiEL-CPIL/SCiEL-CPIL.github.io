"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Two-phase hybrid: radial sweep then aggressive (1+1)-ES for Discus."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Dense deterministic radial sweep from origin to identify needle axis
        # The Discus function has extreme anisotropy - we need to find the narrow axis
        # Sweep along each coordinate axis and some diagonal directions
        
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Phase 1: Radial sweep - evaluate points at various radii along key directions
        # Key directions: coordinate axes and some diagonals
        directions = []
        # Coordinate axes (positive and negative)
        for i in range(dim):
            for sign in [1, -1]:
                d = np.zeros(dim)
                d[i] = sign
                directions.append(d)
        
        # Diagonal directions
        diag1 = np.ones(dim) / np.sqrt(dim)
        diag2 = np.ones(dim) / np.sqrt(dim)
        diag2[0] = -1
        directions.append(diag1)
        directions.append(diag2)
        
        # More diagonal combinations for 5D
        if dim >= 3:
            for i in range(dim):
                for j in range(i+1, dim):
                    d = np.zeros(dim)
                    d[i] = 1/np.sqrt(2)
                    d[j] = 1/np.sqrt(2)
                    directions.append(d)
        
        # Radii to test: logarithmically spaced from small to large
        radii = np.concatenate([
            np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
            np.linspace(4.2, 5.0, 5)
        ])
        
        # Limit Phase 1 to ~100 calls to leave budget for Phase 2
        max_phase1_calls = min(100, budget - 1)
        points_to_test = []
        for d in directions:
            for r in radii:
                x = r * d
                if np.all(np.abs(x) <= 5):
                    points_to_test.append(x)
        
        # Shuffle and take only as many as we can afford
        rng.shuffle(points_to_test)
        points_to_test = points_to_test[:max_phase1_calls]
        
        for x in points_to_test:
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            if calls >= budget:
                break
        
        if calls >= budget:
            return best_x, best_f
        
        # Phase 2: Aggressive (1+1)-ES with adaptive step size
        # Start from the best point found in Phase 1
        x = best_x.copy()
        f_best = best_f
        
        # Initialize step size - start moderately large
        sigma = 0.5
        success_count = 0
        total_trials = 0
        
        # Aggressive mutation rate and adaptation
        c_sigma = 0.5  # Fast adaptation
        c_mu = 1.0     # Step size adaptation rate
        
        while calls < budget:
            # Generate mutation
            z = rng.standard_normal(dim)
            x_trial = x + sigma * z
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = func(x_trial)
            calls += 1
            total_trials += 1
            
            if f_trial < f_best:
                # Success - move to new point
                x = x_trial
                f_best = f_trial
                success_count += 1
                
                # Increase step size slightly on success (aggressive)
                sigma = min(sigma * 1.1, 5.0)
            else:
                # Failure - decrease step size
                sigma = max(sigma * 0.8, 0.01)
            
            # Adaptive step size based on success rate
            if total_trials > 10:
                success_rate = success_count / total_trials
                # Target success rate ~0.2 for (1+1)-ES
                target_rate = 0.2
                if success_rate > target_rate:
                    sigma = min(sigma * 1.05, 5.0)
                elif success_rate < target_rate:
                    sigma = max(sigma * 0.95, 0.01)
            
            # Reset counters periodically
            if total_trials % 50 == 0:
                success_count = 0
                total_trials = 0
        
        return x, f_best
# EVOLVE-BLOCK-END
