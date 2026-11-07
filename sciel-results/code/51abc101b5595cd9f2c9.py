# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-Start Nelder-Mead optimizer for BBOB f7 (Step ellipsoidal) on [-5,5]^d.
    
    Strategy:
    1. Generate a diverse set of initial points (grid + random).
    2. Run Nelder-Mead local optimization from each point, allocating budget dynamically.
    3. Nelder-Mead is particularly effective for smooth, unimodal (or locally unimodal) functions
       like the ellipsoidal part of f7 once the 'step' is crossed.
    4. This avoids the overhead and slow convergence of population-based methods (like CMA-ES)
       for this specific problem structure, allowing for faster descent and better AOCC.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s/330s kill
        dim = self.dim
        budget = self.budget

        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None

        def wrapped(x):
            self._call_count += 1
            # Ensure x is a valid array
            x = np.asarray(x, dtype=float)
            # Clip to box [-5, 5] as per problem statement (though func does it, 
            # doing it here ensures consistency in our tracking if needed, 
            # but func is the source of truth. The problem says func clips.
            # We just pass x to func.
            fx = float(func(x))
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = np.array(x, dtype=float)
            return fx

        # Generate initial points
        # Mix of grid points (to cover space) and random points
        # For dim=5, a full grid is too large. Use a sparse grid + random.
        
        # 1. Center
        points = [np.zeros(dim)]
        
        # 2. Random points
        rng = np.random.default_rng(42)
        n_random = 10
        for _ in range(n_random):
            points.append(rng.uniform(-4.5, 4.5, dim))
            
        # 3. Sparse Grid Points (Halving the space)
        # Create points at the center of each octant/hyper-octant
        # For dim=5, there are 2^5 = 32 hyper-octants.
        # Center of each is at +/- 2.5 in each dimension.
        # This ensures we sample all "corners" of the box.
        signs = np.array(list(np.product([-1, 1], repeat=dim)))
        for s in signs:
            p = s * 2.5
            # Jitter slightly to avoid exact symmetry issues if any, though NM handles it
            points.append(p)
            
        # Shuffle points to avoid bias in order
        rng.shuffle(points)
        
        # Allocate budget
        # We have ~500 calls.
        # Nelder-Mead in dim d requires d+1 = 6 points initially.
        # Each iteration is roughly 1-2 evaluations.
        # Let's allocate a fixed budget per start, but allow early termination if converged.
        # If we have 15 starts, ~33 calls each.
        
        n_starts = len(points)
        # Prioritize starts: Center first? Or shuffle?
        # Shuffled is fine.
        
        # Dynamic budget allocation:
        # Give each start a base budget, but allow "rich" starts to take more if they improve fast?
        # Simpler: Fixed budget per start, e.g., 30 calls.
        # Total: 15 * 30 = 450. Leaves 50 for final polish.
        
        budget_per_start = 30
        remaining_budget = budget
        
        for i, x0 in enumerate(points):
            if self._call_count >= budget - 10: # Keep 10 for final
                break
            if time.time() > deadline:
                break
            
            # Check if we already have a very good solution
            # If best_f is very low, we might want to spend less time on new starts?
            # For now, keep fixed budget to ensure exploration.
            
            # Initial simplex
            # Standard NM simplex: x0, x0 + delta*e_i
            delta = 0.5
            simplex = [x0]
            for j in range(dim):
                x_j = x0.copy()
                x_j[j] += delta
                simplex.append(x_j)
                
            # Run Nelder-Mead
            # We implement a lightweight version to control call count precisely,
            # or use scipy.optimize.minimize with a callback to count.
            # Scipy's NM can be unpredictable in call count.
            # Custom lightweight NM is safer for strict budget.
            
            self._run_nelder_mead(x0, simplex, budget_per_start, wrapped)
            
        # Final Polish:
        # If we have budget left, run NM again from the best point with a tighter delta
        if self._best_x is not None and self._call_count < budget - 5:
            # Reset delta to be small
            delta_final = 0.1
            x0_final = self._best_x.copy()
            simplex_final = [x0_final]
            for j in range(dim):
                x_j = x0_final.copy()
                x_j[j] += delta_final
                simplex_final.append(x_j)
            
            self._run_nelder_mead(x0_final, simplex_final, budget - self._call_count - 5, wrapped)

        # Ensure we have a result
        if self._best_x is None:
            x0 = np.zeros(dim)
            self._best_f = wrapped(x0)
            self._best_x = x0

        return self._best_x, self._best_f

    def _run_nelder_mead(self, x0, initial_simplex, max_calls, func):
        """
        Lightweight Nelder-Mead implementation with strict call budget.
        """
        dim = len(x0)
        simplex = [np.array(p, dtype=float) for p in initial_simplex]
        fvals = [func(p) for p in simplex]
        
        # Parameters
        alpha = 0.5  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        tol = 1e-8
        
        # Initial evaluation count
        calls_made = dim + 1
        
        # Main Loop
        while calls_made < max_calls:
            # Sort simplex by function value
            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            # Check convergence
            if abs(fvals[0] - fvals[-1]) < tol:
                break
            if np.max(np.abs(simplex[0] - simplex[-1])) < tol:
                break
            
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = func(x_r)
            calls_made += 1
            
            if fvals[0] <= f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            elif f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = func(x_e)
                calls_made += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - simplex[-1])
                    
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = func(x_c)
                calls_made += 1
                
                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls_made += 1
                        if calls_made >= max_calls:
                            break
                    if calls_made >= max_calls:
                        break

        # Return best
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]

# EVOLVE-BLOCK-END
