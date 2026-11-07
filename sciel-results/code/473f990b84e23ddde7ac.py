"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        range_vec = hi - lo
        
        # 1. Initialization using Lattice Hyper-Sampling (LHS) for better spread
        # Budget is 200. Let's use 30 for initialization.
        init_size = 30
        
        # Generate LHS points
        # Split range into init_size bins for each dimension
        # For simplicity and speed, we can use a quasi-random sequence or simple LHS logic
        # Here we use a simple LHS implementation
        X_init = np.zeros((init_size, self.dim))
        for d in range(self.dim):
            # Random permutation of bins
            perm = np.random.permutation(init_size)
            # Map to range [lo, hi]
            # bin_size = range / init_size
            # point = lo + (perm + 0.5) * bin_size
            bin_size = range_vec[d] / init_size
            X_init[:, d] = lo[d] + (perm + 0.5) * bin_size
        
        # Evaluate initial population
        F_init = np.array([problem(x) for x in X_init])
        used = len(X_init)
        
        # Initialize Archive
        archive_X = []
        archive_F = []
        
        for i in range(len(F_init)):
            if self._is_non_dominated(F_init[i], archive_F):
                # Remove dominated points from archive
                new_archive_X = []
                new_archive_F = []
                for j in range(len(archive_F)):
                    if not self._dominates(F_init[i], archive_F[j]):
                        new_archive_X.append(archive_X[j])
                        new_archive_F.append(archive_F[j])
                archive_X.append(X_init[i].copy())
                archive_F.append(F_init[i].copy())
        
        # Initialize Swarm
        # Population size for PSO
        pop_size = 20
        # Initialize particles randomly within bounds
        X_swarm = np.random.rand(pop_size, self.dim) * range_vec + lo
        V_swarm = np.random.randn(pop_size, self.dim) * (range_vec * 0.1)
        
        # Initialize personal bests
        F_swarm = np.array([problem(x) for x in X_swarm])
        used += pop_size
        
        P_best_X = X_swarm.copy()
        P_best_F = F_swarm.copy()
        
        # Global best selection: 
        # In multi-objective, "global best" is tricky. 
        # We select a random point from the archive as the "leader" for each particle? 
        # Or a single leader? 
        # Standard MO-PSO often uses a random archive point as the global best for each particle,
        # or a weighted sum. Let's use a strategy where each particle follows a random archive member
        # to encourage exploration of different parts of the front.
        
        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Update velocities and positions
            # w: inertia weight (decreases over time)
            # c1, c2: cognitive and social acceleration coefficients
            w = 0.9 - 0.5 * (used / self.budget)
            c1 = 2.0
            c2 = 2.0
            
            new_X = np.zeros_like(X_swarm)
            
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                
                # Select a global best leader from the archive
                if len(archive_X) > 0:
                    leader_idx = np.random.randint(len(archive_X))
                    G_best = archive_X[leader_idx]
                else:
                    G_best = mid
                
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                # Update velocity
                V_swarm[i] = w * V_swarm[i] + c1 * r1 * (P_best_X[i] - X_swarm[i]) + c2 * r2 * (G_best - X_swarm[i])
                
                # Clamp velocity
                v_max = range_vec * 0.2
                V_swarm[i] = np.clip(V_swarm[i], -v_max, v_max)
                
                # Update position
                new_X[i] = X_swarm[i] + V_swarm[i]
                new_X[i] = np.clip(new_X[i], lo, hi)
            
            # Evaluate new positions
            F_new = np.array([problem(x) for x in new_X])
            used += pop_size
            
            # Update Personal Bests and Archive
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                
                # Update Personal Best
                if self._is_non_dominated(F_new[i], [P_best_F[i]]):
                    # If new point is non-dominated by current personal best
                    # (In multi-objective, we keep the non-dominated set per particle? 
                    #  Standard practice: keep the point if it's non-dominated by the current P_best.
                    #  If P_best is dominated by new, replace. If non-dominated, keep both? 
                    #  For simplicity in PSO, we often just replace if new is better in some sense, 
                    #  or keep a small archive per particle. Let's use a simple rule:
                    #  If new dominates P_best, replace. If P_best dominates new, keep P_best.
                    #  If non-dominated, keep the one with better "quality" (e.g. distance to ideal) 
                    #  or just keep new to encourage movement? 
                    #  Let's keep the one that is non-dominated. If both are non-dominated, 
                    #  we might want to keep the one closer to the current front.
                    #  Simple heuristic: If new dominates P_best, replace. Else if P_best dominates new, keep P_best.
                    #  Else (non-dominated), keep the one with smaller sum of objectives (heuristic).
                    
                    if self._dominates(F_new[i], P_best_F[i]):
                        P_best_X[i] = new_X[i].copy()
                        P_best_F[i] = F_new[i].copy()
                    elif not self._dominates(P_best_F[i], F_new[i]):
                        # Non-dominated. Keep the one closer to the current "ideal" (min of archive)
                        # Or just keep new to explore? Let's keep the one with better HV contribution proxy.
                        # Simpler: keep new to allow exploration.
                        P_best_X[i] = new_X[i].copy()
                        P_best_F[i] = F_new[i].copy()
                    # Else: P_best dominates new, so keep P_best
                
                # Update Global Archive
                if self._is_non_dominated(F_new[i], archive_F):
                    # Remove dominated points from archive
                    new_archive_X = []
                    new_archive_F = []
                    for j in range(len(archive_F)):
                        if not self._dominates(F_new[i], archive_F[j]):
                            new_archive_X.append(archive_X[j])
                            new_archive_F.append(archive_F[j])
                    archive_X.append(new_X[i].copy())
                    archive_F.append(F_new[i].copy())
                    
                    # Truncate archive if too large? 
                    # With 200 budget, archive size is manageable. 
                    # But if it grows too large, selection becomes slow. 
                    # Let's cap it at 50 for efficiency, keeping best HV contributors.
                    if len(archive_F) > 50:
                        # Simple truncation: keep random 50? Or best HV?
                        # For speed, let's just keep the last 50 or random.
                        # Better: keep points that are far apart.
                        # For now, simple random truncation to save time.
                        idxs = np.random.choice(len(archive_F), 50, replace=False)
                        archive_X = [archive_X[i] for i in idxs]
                        archive_F = [archive_F[i] for i in idxs]

            X_swarm = new_X
            
        return archive_X if archive_X else [X_init[0]]

    def _dominates(self, f1, f2):
        """
        Returns True if f1 dominates f2 (minimization).
        f1 <= f2 for all objectives, and f1 < f2 for at least one.
        """
        if np.all(f1 <= f2) and np.any(f1 < f2):
            return True
        return False

    def _is_non_dominated(self, f_new, archive_F):
        """
        Returns True if f_new is non-dominated by any point in archive_F.
        """
        for f_arch in archive_F:
            if self._dominates(f_arch, f_new):
                return False
        return True
# EVOLVE-BLOCK-END
