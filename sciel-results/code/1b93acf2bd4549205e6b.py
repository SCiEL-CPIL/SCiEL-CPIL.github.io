"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Objective Particle Swarm Optimization (MOPSO) with Pareto Archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Parameters
        pop_size = 20
        max_iter = self.budget // pop_size
        omega = 0.9  # Inertia weight
        c1 = 1.49445  # Cognitive coefficient
        c2 = 1.49445  # Social coefficient
        
        # Initialize positions
        X = lo + span * rng.random((pop_size, self.dim))
        # Initialize velocities to zero
        V = np.zeros((pop_size, self.dim))
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Initialize personal bests
        p_best_X = X.copy()
        p_best_F = F.copy()
        
        # Initialize archive
        archive_X = [X[0]]
        archive_F = [F[0]]
        
        def is_dominated(a, b):
            """Check if a dominates b (minimization)."""
            return np.all(a <= b) and np.any(a < b)
        
        def update_archive(x, f, arch_X, arch_F):
            """Update archive with new solution x, f. Returns new archive lists."""
            new_arch_X = []
            new_arch_F = []
            
            # Remove dominated solutions from archive
            for ax, af in zip(arch_X, arch_F):
                if is_dominated(f, af):
                    continue # New solution dominates archive member, remove it
                if is_dominated(af, f):
                    continue # Archive member dominates new solution, skip adding
                # Non-dominated with respect to each other
                new_arch_X.append(ax)
                new_arch_F.append(af)
            
            # Check if new solution is dominated by any remaining archive member
            dominated = False
            for af in new_arch_F:
                if is_dominated(af, f):
                    dominated = True
                    break
            
            if not dominated:
                new_arch_X.append(x)
                new_arch_F.append(f)
                
            return new_arch_X, new_arch_F

        def select_global_best(arch_F):
            """Select a global best from the archive using crowding distance."""
            if len(arch_F) == 0:
                return None
            if len(arch_F) == 1:
                return arch_F[0]
            
            # Calculate crowding distance
            n = len(arch_F)
            cd = np.zeros(n)
            m = self.n_obj
            
            for k in range(m):
                obj_k = np.array([f[k] for f in arch_F])
                sorted_idx = np.argsort(obj_k)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_min = obj_k[sorted_idx[0]]
                f_max = obj_k[sorted_idx[-1]]
                f_range = f_max - f_min
                if f_range > 1e-10:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (obj_k[sorted_idx[i+1]] - obj_k[sorted_idx[i-1]]) / f_range
            
            # Select with highest crowding distance
            idx = np.argmax(cd)
            return arch_F[idx]

        # Main MOPSO loop
        for gen in range(max_iter):
            if used >= self.budget:
                break
            
            # Select global best from archive
            g_best_F = select_global_best(archive_F)
            if g_best_F is None:
                # Fallback to best in population if archive empty (should not happen)
                idx_best = np.argmin(np.sum(F, axis=1))
                g_best_F = F[idx_best]
            else:
                # Find corresponding X in archive? 
                # For simplicity, we can just store the X of the selected g_best
                # We need to track which X corresponds to which F in archive
                # Let's modify update_archive to return indices or we store pairs.
                # Actually, let's keep a list of (X, F) in archive.
                pass

            # Re-implementing archive handling to keep X and F paired
            # Let's restart the loop logic with proper archive structure
            
            # Since we are in the middle of the function, let's just use a simple list of dicts or tuples
            # But we already defined the function. Let's adjust the main loop to use the archive properly.
            
            # We need the X of the g_best. 
            # Let's find the index of g_best_F in archive_F
            # This is inefficient but okay for small archives.
            g_best_X = None
            for ax, af in zip(archive_X, archive_F):
                if np.allclose(af, g_best_F):
                    g_best_X = ax
                    break
            if g_best_X is None:
                g_best_X = p_best_X[0] # Fallback

            new_X = X.copy()
            new_F = F.copy()
            
            for i in range(pop_size):
                if used >= self.budget:
                    break
                
                r1 = rng.random()
                r2 = rng.random()
                
                # Velocity update
                V[i] = omega * V[i] + c1 * r1 * (p_best_X[i] - X[i]) + c2 * r2 * (g_best_X - X[i])
                
                # Position update
                new_X[i] = X[i] + V[i]
                
                # Boundary handling
                new_X[i] = np.clip(new_X[i], lo, hi)
                
                # Reset velocity if out of bounds
                for j in range(self.dim):
                    if X[i][j] + V[i][j] < lo[j] or X[i][j] + V[i][j] > hi[j]:
                        V[i][j] = 0.0
                        
                # Evaluate
                f_trial = problem(new_X[i])
                used += 1
                new_F[i] = f_trial
                
                # Update personal best
                if is_dominated(f_trial, p_best_F[i]):
                    p_best_F[i] = f_trial
                    p_best_X[i] = new_X[i].copy()
                    
                # Update archive
                archive_X, archive_F = update_archive(new_X[i], f_trial, archive_X, archive_F)
                
                # Update global best if new solution is added to archive and has high crowding?
                # We select g_best from archive at the start of the next iteration.
                # For current iteration, we use the g_best selected at the start.
                
            X = new_X
            F = new_F
            
        return X
# EVOLVE-BLOCK-END
