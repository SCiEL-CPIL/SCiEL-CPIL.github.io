import numpy as np
from scipy.spatial.distance import cdist

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.rng = np.random.default_rng(42)

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # MOPSO Parameters
        # Budget is 200. 
        # Swarm size N. 
        # Iterations T.
        # Total evals = N (init) + T * N.
        # 200 = N + T*N => N(1+T) <= 200.
        # Let's pick N=20, T=9 => 20 * 10 = 200.
        N = 20
        T = (self.budget // N) - 1
        if T < 1:
            T = 1
            N = self.budget // 2
            
        # Inertia weights: linearly decreasing from 0.9 to 0.4
        w_max = 0.9
        w_min = 0.4
        c1 = 1.49445 # Cognitive coefficient
        c2 = 1.49445 # Social coefficient
        
        # Initialize Swarm
        X = lo + (hi - lo) * self.rng.random((N, self.dim))
        V = (hi - lo) * self.rng.random((N, self.dim)) * 0.1 # Small initial velocity
        
        # Evaluate Initial Swarm
        F = np.array([problem(x) for x in X])
        self.budget_used = N
        
        # Initialize Personal Bests
        P_best = X.copy()
        F_best = F.copy()
        
        # Initialize Archive (Global Bests)
        # Archive stores non-dominated points
        Archive_X = [X[0]]
        Archive_F = [F[0]]
        
        def is_dominated(a, b):
            # Returns True if a dominates b
            return np.all(a <= b) and np.any(a < b)
        
        def update_archive(new_x, new_f):
            # Add new_x to archive if it doesn't dominate existing, and isn't dominated
            # Simple O(M) insertion for small archives
            dominated_existing = []
            is_dominated_by_archive = False
            
            for i, af in enumerate(Archive_F):
                if is_dominated(new_f, af):
                    dominated_existing.append(i)
                elif is_dominated(af, new_f):
                    is_dominated_by_archive = True
                    break
            
            if is_dominated_by_archive:
                return
            
            # Remove dominated existing points
            for i in sorted(dominated_existing, reverse=True):
                Archive_X.pop(i)
                Archive_F.pop(i)
            
            # Add new point
            Archive_X.append(new_x)
            Archive_F.append(new_f)
            
            # Limit archive size? 
            # Usually not strictly necessary for small budgets, but let's keep it manageable.
            # If archive grows too large, we might need to prune. 
            # For 200 evals, archive size is likely < 50.
            
        # Update archive with initial population
        for i in range(N):
            update_archive(X[i], F[i])
            
        # Helper: Crowding Distance for Archive selection
        def crowding_distance(archive_f):
            M = len(archive_f)
            if M <= 2:
                return np.full(M, np.inf)
            
            cd = np.zeros(M)
            for k in range(self.n_obj):
                sorted_idx = np.argsort(archive_f[:, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                
                f_min = archive_f[sorted_idx[0], k]
                f_max = archive_f[sorted_idx[-1], k]
                if f_max == f_min:
                    continue
                
                for i in range(1, M - 1):
                    cd[sorted_idx[i]] += (archive_f[sorted_idx[i+1], k] - archive_f[sorted_idx[i-1], k]) / (f_max - f_min)
            return cd

        # Helper: Select Gbest from Archive using Crowding Distance
        def select_gbest():
            if not Archive_X:
                return None
            if len(Archive_X) == 1:
                return Archive_X[0]
            
            af = np.array(Archive_F)
            cd = crowding_distance(af)
            # Select the point with maximum crowding distance (most diverse)
            # To avoid always picking the same extreme, we can use tournament or random among top.
            # Standard MOPSO often uses uniformity.
            idx = np.argmax(cd)
            return Archive_X[idx]

        # Main Loop
        for t in range(T):
            if self.budget_used >= self.budget:
                break
                
            w = w_max - (w_max - w_min) * (t / max(T - 1, 1))
            
            # Select Gbest for this iteration
            # In standard MOPSO, gbest is chosen from archive. 
            # Some variants use a single global best, others allow multiple.
            # We will pick one gbest per particle or one global?
            # Standard: One gbest for the whole swarm? No, usually each particle has its own gbest 
            # selected from the archive, often the one with highest crowding distance 
            # or randomly selected from the archive to maintain diversity.
            # Let's select a gbest for each particle from the archive using crowding distance 
            # to encourage exploration of different parts of the front.
            
            new_X = np.zeros_like(X)
            new_F = np.zeros_like(F)
            
            for i in range(N):
                # Select Gbest for particle i
                if Archive_X:
                    # Choose gbest from archive. 
                    # To improve diversity, we can pick the one with max crowding distance, 
                    # or random. Let's use max CD for exploitation of diverse points.
                    af = np.array(Archive_F)
                    cd = crowding_distance(af)
                    gbest_idx = np.argmax(cd)
                    gbest = Archive_X[gbest_idx]
                else:
                    gbest = P_best[i]
                    
                # Update Velocity
                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)
                V[i] = w * V[i] + c1 * r1 * (P_best[i] - X[i]) + c2 * r2 * (gbest - X[i])
                
                # Clamp Velocity
                V_max = 0.2 * (hi - lo)
                V[i] = np.clip(V[i], -V_max, V_max)
                
                # Update Position
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], lo, hi)
                
                # Evaluate
                F[i] = problem(X[i])
                self.budget_used += 1
                
                # Update Pbest
                # If F[i] dominates P_best[i], update
                # If P_best[i] dominates F[i], keep P_best
                # If non-dominated, keep P_best (conservative) or replace if random?
                # Standard: Keep Pbest if it is non-dominated to F[i].
                if is_dominated(F[i], F_best[i]):
                    P_best[i] = X[i]
                    F_best[i] = F[i]
                # If F[i] is non-dominated to P_best[i], we might want to update P_best 
                # to the new point if it's "better" in some sense, but standard MOPSO 
                # often keeps the older one to maintain stability, or updates if it 
                # dominates. Here we only update if it dominates.
                
                # Update Archive
                update_archive(X[i], F[i])
                
            # Update X and F for next iteration
            # Note: X and F were updated in place in the loop above for the new positions
            # But we need to ensure the next iteration uses the new X.
            # The loop above updated X[i] and F[i] directly.
            
        return X
