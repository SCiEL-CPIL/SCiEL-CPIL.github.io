"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective Optimizer using MOPSO (Multi-Objective Particle Swarm Optimization)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # MOPSO Parameters
        # With budget 200 and dim 4, a smaller swarm is appropriate to ensure
        # enough iterations for the swarm to converge.
        # 10 particles * 20 iterations = 200 evaluations.
        swarm_size = 10
        max_iter = self.budget // swarm_size
        w = 0.9  # Inertia weight
        c1 = 1.5 # Cognitive coefficient
        c2 = 1.5 # Social coefficient
        
        # Initialize swarm randomly within bounds
        X = lo + (hi - lo) * np.random.rand(swarm_size, dim)
        V = (hi - lo) * np.random.rand(swarm_size, dim) * 0.1 # Small initial velocity
        
        # Evaluate initial swarm
        F = np.zeros((swarm_size, n_obj))
        eval_count = 0
        
        for i in range(swarm_size):
            if eval_count >= self.budget or time.time() > deadline:
                return None
            F[i] = problem(X[i])
            eval_count += 1
            
        # Initialize Archive with initial non-dominated points
        # We need a helper to check non-dominance and update archive
        # Archive stores [x, f]
        archive = []
        
        def dominates(f1, f2):
            # f1 dominates f2 if f1 <= f2 in all objectives and f1 < f2 in at least one
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        def update_archive(f_new):
            # Returns True if added, False if dominated
            # Also removes points dominated by f_new
            for i, point in enumerate(archive):
                if dominates(f_new, point[1]):
                    archive.pop(i)
                    i -= 1
                    continue
                if dominates(point[1], f_new):
                    return False
            archive.append(f_new)
            return True
        
        # Initial archive update
        for i in range(swarm_size):
            update_archive(F[i])
            
        # Crowding distance calculation for archive selection
        def crowding_distance(points):
            # points is list of objective vectors
            n = len(points)
            if n <= 2:
                return [float('inf')] * n
            distances = [0.0] * n
            points = np.array(points)
            for m in range(n_obj):
                # Sort indices by objective m
                order = np.argsort(points[:, m])
                # End points get infinite distance
                distances[order[0]] = float('inf')
                distances[order[-1]] = float('inf')
                f_min = points[order[0], m]
                f_max = points[order[-1], m]
                if f_max == f_min:
                    continue
                for k in range(1, n - 1):
                    distances[order[k]] += (points[order[k+1], m] - points[order[k-1], m]) / (f_max - f_min)
            return distances
        
        # Main loop
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            if eval_count >= self.budget:
                return None
                
            # Update velocities and positions
            # For each particle
            for i in range(swarm_size):
                # Select global best (gbest) from archive
                # Strategy: Pick a point from archive with highest crowding distance
                # This promotes diversity
                if not archive:
                    # Should not happen if we have evaluations, but safeguard
                    gbest = np.random.uniform(lo, hi)
                else:
                    # Calculate crowding distance
                    cd = crowding_distance([p[1] for p in archive])
                    # Find index with max crowding distance
                    # Randomly select among top few if ties or to avoid bias?
                    # Simple max is standard for MOPSO to fill gaps
                    max_cd_idx = np.argmax(cd)
                    # The archive stores only f vectors in my simplified update_archive above?
                    # Wait, my update_archive above appends f_new (vector).
                    # I need x as well for gbest? 
                    # Actually, for weighted sum or simple MOPSO, gbest is a point in objective space 
                    # used to guide velocity. But velocity is in decision space.
                    # Standard MOPSO: gbest is a point from the archive.
                    # My archive currently only stores f vectors. I need to store x too.
                    # Let's fix the archive structure.
                    pass
                    
            # RE-IMPLEMENTATION OF ARCHIVE TO STORE X AND F
            # Let's restart the logic inside the loop properly or fix the structure.
            # The previous block was pseudo-code. Let's write the real loop.
            
            pass # The previous code block was just drafting. Here is the real implementation.

        # ACTUAL IMPLEMENTATION BELOW REPLACES THE DRAFT ABOVE
        # We need to re-run the initialization correctly to ensure X and F are linked in Archive.
        
        # Reset state for correct implementation
        X = lo + (hi - lo) * np.random.rand(swarm_size, dim)
        V = (hi - lo) * np.random.rand(swarm_size, dim) * 0.1
        F = np.zeros((swarm_size, n_obj))
        eval_count = 0
        archive = [] # List of [x, f]
        
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        def update_archive(x_new, f_new):
            # Check if f_new is dominated by any existing archive point
            for point in archive:
                if dominates(point[1], f_new):
                    return False
            # Remove points dominated by f_new
            new_archive = []
            for point in archive:
                if not dominates(f_new, point[1]):
                    new_archive.append(point)
            archive.clear()
            archive.extend(new_archive)
            archive.append([x_new, f_new])
            return True
        
        # Initial Evaluation
        for i in range(swarm_size):
            if eval_count >= self.budget or time.time() > deadline:
                return None
            F[i] = problem(X[i])
            eval_count += 1
            update_archive(X[i].copy(), F[i].copy())
            
        # Main Loop
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            if eval_count >= self.budget:
                return None
                
            for i in range(swarm_size):
                if eval_count >= self.budget or time.time() > deadline:
                    return None
                    
                # Select gbest from archive using Crowding Distance
                if len(archive) > 0:
                    # Calculate crowding distance for archive
                    f_archive = np.array([p[1] for p in archive])
                    n_arch = len(archive)
                    if n_arch <= 2:
                        cd = [float('inf')] * n_arch
                    else:
                        cd = [0.0] * n_arch
                        for m in range(n_obj):
                            order = np.argsort(f_archive[:, m])
                            cd[order[0]] = float('inf')
                            cd[order[-1]] = float('inf')
                            f_min = f_archive[order[0], m]
                            f_max = f_archive[order[-1], m]
                            if f_max > f_min:
                                for k in range(1, n_arch - 1):
                                    cd[order[k]] += (f_archive[order[k+1], m] - f_archive[order[k-1], m]) / (f_max - f_min)
                            # If f_max == f_min, distance remains 0 for that objective
                            
                    # Select index with max crowding distance
                    # To avoid always picking the same one if distances are equal or similar,
                    # we can pick randomly from the top 20% or just max.
                    # Max is standard.
                    max_cd_idx = np.argmax(cd)
                    gbest = archive[max_cd_idx][0]
                else:
                    # Fallback if archive is empty (should not happen)
                    gbest = np.random.uniform(lo, hi)
                    
                # Personal best is tracked implicitly by checking if new point dominates current personal best?
                # In simple MOPSO, we often just use the global best. 
                # However, standard PSO uses personal best.
                # Let's maintain personal bests P_i
                # For simplicity and robustness in this constrained budget, 
                # using only gbest (Global Best MOPSO) is a valid variant and often faster to converge.
                # But let's add Personal Best for better exploration.
                
                # We didn't initialize P. Let's assume P_i = X_i initially.
                # We need to store P.
                # Since I can't easily restructure the whole class without risking bugs,
                # I will stick to a "Global Best only" strategy which is simpler and effective for small budgets.
                # Velocity update: V_i = w*V_i + c1*r1*(gbest - X_i) + c2*r2*(gbest - X_i)
                # Since we don't have distinct P and G in this simplified loop, we treat G as the attractor.
                
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                
                V[i] = w * V[i] + c1 * r1 * (gbest - X[i]) + c2 * r2 * (gbest - X[i])
                
                # Clip velocity
                V_max = 0.5 * (hi - lo)
                V[i] = np.clip(V[i], -V_max, V_max)
                
                X[i] = X[i] + V[i]
                
                # Clip position
                X[i] = np.clip(X[i], lo, hi)
                
                # Evaluate
                F[i] = problem(X[i])
                eval_count += 1
                
                # Update Archive
                update_archive(X[i].copy(), F[i].copy())
                
        return None
# EVOLVE-BLOCK-END
