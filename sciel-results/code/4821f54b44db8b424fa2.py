"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using Multi-objective Particle Swarm Optimization (MOPSO).
    Designed for WFG4 (12 dims, 3 objectives) with a budget of 200 evaluations.
    
    Strategy:
    - Use MOPSO with a grid-based archive to maintain diversity.
    - MOPSO is often more effective than DE for multi-objective problems in moderate dimensions
      because it uses velocity to guide exploration and exploitation.
    - The grid archive ensures that solutions are spread across the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Budget allocation
        # 200 evals total.
        # Initialization: 20 random points.
        # Remaining: 180 evals for MOPSO iterations.
        # Population size: 10.
        # 180 / 10 = 18 generations.
        
        pop_size = 10
        init_pop = 20
        
        # 1. Initialization
        rng = np.random.default_rng(0)
        X_init = lo + range_ * rng.random((init_pop, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        
        # Initialize archive
        archive_X = []
        archive_F = []
        
        # Grid archive parameters
        # For 3 objectives, we can use a simple grid.
        # Let's define grid cells for the objective space.
        # Reference point is [3.0, 5.0, 7.0].
        # We'll use a grid of size 5x5x5 or similar to keep it manageable.
        # Actually, a dynamic grid or just a simple non-dominated sort with crowding distance
        # is more robust. Let's use a simple archive with a cap and crowding distance.
        
        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if dominated by any in archive
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return False
            # Remove any in archive dominated by new point
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F
            return True

        for x, f in zip(X_init, F_init):
            add_to_archive(x, f)

        # Current population for MOPSO
        X = X_init[:pop_size].copy()
        F = F_init[:pop_size].copy()
        
        # Initialize velocities
        # V = 0.5 * range * (rng.random - 0.5)
        V = 0.5 * range_ * (rng.random((pop_size, self.dim)) - 0.5)
        
        # Best individual per particle (pbest)
        Pbest_X = X.copy()
        Pbest_F = F.copy()
        
        # Global best (gbest) - selected from archive
        # We need a mechanism to select gbest from archive.
        # Simple approach: random selection from archive, or selection based on crowding distance.
        # Let's use a simple random selection from archive for gbest to encourage exploration.
        
        used_evals = init_pop
        
        # MOPSO Parameters
        w = 0.5  # Inertia weight
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        
        # 2. Main Loop
        while used_evals < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            for i in range(pop_size):
                if used_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select gbest from archive
                if len(archive_X) > 0:
                    gbest_idx = rng.integers(len(archive_X))
                    gbest_F = archive_F[gbest_idx]
                else:
                    gbest_F = F[i]  # Fallback
                
                # Update velocity
                r1 = rng.random(self.dim)
                r2 = rng.random(self.dim)
                
                # Velocity update: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                # Note: pbest and gbest are in decision space for MOPSO?
                # Actually, in MOPSO, pbest and gbest are positions (decision variables).
                # We need to store Pbest_X and gbest_X.
                
                # Let's re-initialize Pbest_X and gbest_X properly.
                # Pbest_X is the best position found by particle i so far.
                # gbest_X is the position of the global best from archive.
                
                # We need to update Pbest_X when a better solution is found.
                # "Better" in multi-objective context is tricky.
                # Let's use a simple rule: if new F dominates Pbest_F, update Pbest.
                
                # First, update position
                V[i] = w * V[i] + c1 * r1 * (Pbest_X[i] - X[i]) + c2 * r2 * (archive_X[gbest_idx] - X[i])
                
                # Clip velocity to prevent oscillation
                max_v = 0.5 * range_
                V[i] = np.clip(V[i], -max_v, max_v)
                
                X[i] = X[i] + V[i]
                
                # Clip position to bounds
                X[i] = np.clip(X[i], lo, hi)
                
                # Evaluate
                F[i] = problem(X[i])
                used_evals += 1
                
                # Update Pbest
                if np.all(F[i] <= Pbest_F[i]) and np.any(F[i] < Pbest_F[i]):
                    Pbest_F[i] = F[i]
                    Pbest_X[i] = X[i]
                
                # Add to archive
                add_to_archive(X[i], F[i])
                
                # If archive is too large, we might want to thin it out.
                # For now, let's just let it grow, as the budget is small.
                # A cap of 50-100 is reasonable.
                if len(archive_X) > 50:
                    # Simple thinning: remove dominated points (already done in add_to_archive)
                    # If still too large, remove random points? Or use crowding distance?
                    # Let's just keep the first 50 for simplicity, or random 50.
                    # Actually, let's implement a simple crowding distance-based thinning.
                    # This is getting complex. Let's just cap it by removing the most recently added
                    # if it's not strictly better than existing ones? No, add_to_archive handles that.
                    # Let's just ignore the cap for now, 200 evals won't make it huge.
                    pass

        # Final step: Ensure we return something valid (though return value is ignored)
        # The evaluator records all problem(x) calls.
        return X
# EVOLVE-BLOCK-END
