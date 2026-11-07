"""Rewritten program using Multi-Objective Particle Swarm Optimization (MOPSO)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Objective Particle Swarm Optimization for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        # Population size
        N = 40
        if N > budget:
            N = budget
            
        # Initialize Particles
        X = lo + (hi - lo) * rng.random((N, n))
        # Velocity initialization
        v_max = 0.5 * (hi - lo)
        V = (rng.random((N, n)) - 0.5) * v_max
        
        F = np.array([problem(x) for x in X], dtype=float)
        used = N
        
        # pbest tracking
        pbest_X = X.copy()
        pbest_F = F.copy()
        
        # Archive of (X, F) tuples
        archive_X = []
        archive_F = []
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def update_archive(x_new, f_new):
            nonlocal archive_X, archive_F
            # Check if f_new is dominated by any in archive
            for i in range(len(archive_F)):
                if dominates(archive_F[i], f_new):
                    return
            
            # Remove dominated points from archive
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_F)):
                if dominates(f_new, archive_F[i]):
                    continue
                new_archive_X.append(archive_X[i])
                new_archive_F.append(archive_F[i])
            
            new_archive_X.append(x_new)
            new_archive_F.append(f_new)
            archive_X = new_archive_X
            archive_F = new_archive_F

        # Initialize Archive
        for i in range(N):
            update_archive(X[i].copy(), F[i].copy())

        # PSO Parameters
        w = 0.7
        c1 = 1.5
        c2 = 1.5

        # Main Loop
        while used < budget:
            for i in range(N):
                if used >= budget:
                    break
                
                # Select Leader
                # Find archive point closest to current F[i] in objective space
                if len(archive_F) == 0:
                    leader_X = X[i]
                else:
                    # Compute squared distances
                    dists = [np.sum((f - F[i])**2) for f in archive_F]
                    leader_idx = np.argmin(dists)
                    leader_X = archive_X[leader_idx]

                # Update Velocity
                r1 = rng.random(n)
                r2 = rng.random(n)
                V[i] = w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (leader_X - X[i])
                
                # Clamp Velocity
                V[i] = np.clip(V[i], -v_max, v_max)
                
                # Update Position
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], lo, hi)
                
                # Evaluate
                F_new = problem(X[i])
                used += 1
                
                # Update pbest
                if dominates(F_new, pbest_F[i]):
                    pbest_F[i] = F_new
                    pbest_X[i] = X[i].copy()
                
                # Update Archive
                update_archive(X[i].copy(), F_new)

        return X
# EVOLVE-BLOCK-END
