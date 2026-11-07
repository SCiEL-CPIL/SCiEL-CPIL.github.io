"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """MOPSO (Multi-Objective Particle Swarm Optimization) for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        N = 20
        if budget < N:
            N = budget

        # Initialize positions
        X = lo + (hi - lo) * rng.random((N, dim))
        # Initialize velocities to 0
        V = np.zeros((N, dim))
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X], dtype=float)
        used = N

        # Personal Bests
        Pbest_X = X.copy()
        Pbest_F = F.copy()
        
        # External Archive (stores non-dominated solutions)
        # We will store indices or just the F and X values
        archive_F = []
        archive_X = []
        
        def is_dominated(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def update_archive(f_new, x_new, arch_F, arch_X):
            # Check if new point is dominated by any in archive
            for f_a in arch_F:
                if is_dominated(f_new, f_a):
                    return arch_F, arch_X
            
            # Remove points in archive dominated by new point
            new_arch_F = []
            new_arch_X = []
            for f_a, x_a in zip(arch_F, arch_X):
                if not is_dominated(f_new, f_a):
                    new_arch_F.append(f_a)
                    new_arch_X.append(x_a)
            
            new_arch_F.append(f_new)
            new_arch_X.append(x_new)
            
            # Limit archive size if it grows too large (optional, but good for diversity selection)
            # For now, keep all. If > 50, prune using crowding distance
            if len(new_arch_F) > 50:
                # Simple pruning: keep 50 most diverse
                # Calculate crowding distance
                cd = np.zeros(len(new_arch_F))
                for k in range(n_obj):
                    f_vals = np.array([f[k] for f in new_arch_F])
                    sorted_idx = np.argsort(f_vals)
                    if len(new_arch_F) > 2:
                        cd[sorted_idx[0]] = float('inf')
                        cd[sorted_idx[-1]] = float('inf')
                        f_range = f_vals.max() - f_vals.min()
                        if f_range > 0:
                            for m in range(1, len(new_arch_F) - 1):
                                idx = sorted_idx[m]
                                prev_idx = sorted_idx[m-1]
                                next_idx = sorted_idx[m+1]
                                cd[idx] += (f_vals[next_idx] - f_vals[prev_idx]) / f_range
                # Keep top 50 by crowding distance
                keep_indices = np.argsort(cd)[-50:]
                new_arch_F = [new_arch_F[i] for i in keep_indices]
                new_arch_X = [new_arch_X[i] for i in keep_indices]
                
            return new_arch_F, new_arch_X

        # Initialize archive with initial population
        for i in range(N):
            archive_F, archive_X = update_archive(F[i], X[i], archive_F, archive_X)

        # PSO Parameters
        w = 0.5  # Inertia weight
        c1 = 1.5 # Cognitive coefficient
        c2 = 1.5 # Social coefficient
        
        # Main Loop
        while used + N <= budget:
            # Select Global Best from Archive using Crowding Distance
            if len(archive_F) == 0:
                # Fallback if archive is empty (shouldn't happen)
                Gbest_idx = 0
            else:
                # Calculate crowding distance for archive
                cd = np.zeros(len(archive_F))
                for k in range(n_obj):
                    f_vals = np.array([f[k] for f in archive_F])
                    sorted_idx = np.argsort(f_vals)
                    if len(archive_F) > 2:
                        cd[sorted_idx[0]] = float('inf')
                        cd[sorted_idx[-1]] = float('inf')
                        f_range = f_vals.max() - f_vals.min()
                        if f_range > 0:
                            for m in range(1, len(archive_F) - 1):
                                idx = sorted_idx[m]
                                prev_idx = sorted_idx[m-1]
                                next_idx = sorted_idx[m+1]
                                cd[idx] += (f_vals[next_idx] - f_vals[prev_idx]) / f_range
                # Pick the one with highest crowding distance (most isolated)
                Gbest_idx = np.argmax(cd)
            
            Gbest_F = archive_F[Gbest_idx]
            Gbest_X = archive_X[Gbest_idx]

            # Update Particles
            for i in range(N):
                r1 = rng.random()
                r2 = rng.random()
                
                # Velocity Update
                V[i] = w * V[i] + c1 * r1 * (Pbest_X[i] - X[i]) + c2 * r2 * (Gbest_X - X[i])
                
                # Position Update
                X[i] = X[i] + V[i]
                
                # Clip to bounds
                X[i] = np.clip(X[i], lo, hi)
                
                # Evaluate
                F[i] = problem(X[i])
                
                # Update Personal Best
                if is_dominated(Pbest_F[i], F[i]):
                    Pbest_F[i] = F[i]
                    Pbest_X[i] = X[i].copy()
                
                # Update Archive
                archive_F, archive_X = update_archive(F[i], X[i], archive_F, archive_X)
            
            used += N

        # Final non-dominated set from population and archive
        # Combine population and archive, find non-dominated
        all_F = np.vstack([F, np.array(archive_F)]) if len(archive_F) > 0 else F
        all_X = np.vstack([X, np.array(archive_X)]) if len(archive_X) > 0 else X
        
        n_total = len(all_F)
        non_dom_idx = []
        for i in range(n_total):
            dominated = False
            for j in range(n_total):
                if i == j: continue
                if is_dominated(all_F[i], all_F[j]):
                    dominated = True
                    break
            if not dominated:
                non_dom_idx.append(i)
        
        if not non_dom_idx:
            non_dom_idx = list(range(n_total))
            
        return all_X[non_dom_idx]

# EVOLVE-BLOCK-END
