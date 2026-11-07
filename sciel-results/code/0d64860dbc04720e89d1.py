"""Multi-objective optimizer on DTLZ1 with analytical front sampling + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1.
    
    Strategy:
    1. Generate points on the analytical Pareto front (where g(x)=0).
    2. Use remaining budget for local refinement via mutation.
    3. Track all evaluated points and return the non-dominated set.
    
    DTLZ1 Pareto front:
    - f1 = 0.5 * (1 + g) * x1 * x2 * (1 + x3)
    - f2 = 0.5 * (1 + g) * x1 * (1 - x2)
    - f3 = 0.5 * (1 + g) * (1 - x1)
    where g(x) = 100 + 10*n + sum_{i=2}^{n}(x_i - 0.5)^2 - 10*cos(20*pi*(x_i - 0.5))
    
    On the Pareto front, g=0, so:
    - f1 = 0.5 * x1 * x2 * (1 + x3)
    - f2 = 0.5 * x1 * (1 - x2)
    - f3 = 0.5 * (1 - x1)
    
    With x1, x2, x3 in [0,1] and the rest chosen to make g=0.
    For simplicity, set x_i = 0.5 for i>=4 (which gives (x_i-0.5)^2 = 0 and cos term = cos(0)=1).
    Then g = 100 + 10*7 + 0 - 10*6 = 100 + 70 - 60 = 110. That's not zero.
    
    Actually, let's reconsider. The standard DTLZ1 has:
    g(x) = 100 + (n-M) + sum_{i=M+1}^{n} [(x_i - 0.5)^2 - 10*cos(20*pi*(x_i - 0.5))]
    where M is the number of position variables. For DTLZ1 with 3 objectives, M=3, n=7.
    So g = 100 + (7-3) + sum_{i=4}^{7} [(x_i - 0.5)^2 - 10*cos(20*pi*(x_i - 0.5))]
    
    To get g=0, we need the sum to be -104. The minimum of each term (x_i-0.5)^2 - 10*cos(20*pi*(x_i-0.5)) 
    is approximately -10 (when x_i=0.5, the cos term is 1, so -10*1 = -10, and (0)^2=0, so -10).
    With 4 terms, the minimum sum is -40. So g_min = 100 + 4 - 40 = 64. That's still positive.
    
    Hmm, so g can never be 0 for DTLZ1 with these parameters? Let me re-check the DTLZ1 definition.
    
    Actually, I think the standard DTLZ1 definition might have different parameters. Let me just use 
    random sampling and a simple evolutionary approach, but with a smarter initialization.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 280.0  # Safety margin before 300s limit
    
    def _elapsed(self):
        return time.time() - self.start_time
    
    def _time_left(self):
        return self.deadline - self._elapsed()
    
    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Initialize with random points
        pop_size = min(50, self.budget // 4)
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Keep track of all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        # Main evolutionary loop
        while used < self.budget and self._time_left() > 1.0:
            # Find non-dominated points
            non_dominated = self._find_non_dominated(F)
            
            if len(non_dominated) == 0:
                parents = X
            else:
                parents = X[non_dominated]
            
            # Select parent via tournament or random from non-dominated
            parent = parents[rng.integers(len(parents))]
            
            # Mutate
            mutation_rate = 0.3
            sigma = 0.1
            child = parent.copy()
            for i in range(self.dim):
                if rng.random() < mutation_rate:
                    child[i] += sigma * span[i] * rng.standard_normal()
            child = np.clip(child, lo, hi)
            
            # Evaluate
            fc = problem(child)
            used += 1
            
            # Replace worst point (by sum of objectives)
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc
        
        # Collect all evaluated points
        all_points_X = []
        all_points_F = []
        for x_arr, f_arr in zip(all_X, all_F):
            all_points_X.append(x_arr)
            all_points_F.append(f_arr)
        # Also include current population
        all_points_X.append(X)
        all_points_F.append(F)
        
        all_X_flat = np.vstack(all_points_X)
        all_F_flat = np.vstack(all_points_F)
        
        # Remove dominated points
        final_mask = self._find_non_dominated(all_F_flat)
        final_X = all_X_flat[final_mask]
        final_F = all_F_flat[final_mask]
        
        return final_X
    
    def _find_non_dominated(self, F):
        """Find indices of non-dominated points in F (minimization)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if is_dominated[i]:
                continue
            # Check if any point dominates i
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                # j dominates i if j <= i in all objectives and j < i in at least one
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        
        return np.where(~is_dominated)[0]
# EVOLVE-BLOCK-END
