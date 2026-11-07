# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Multi-objective Differential Evolution (MODE) for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        N = 10
        if N > budget // 2:
            N = max(5, budget // 2)
            
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        # DE parameters
        F_mut = 0.8  # Mutation factor
        CR = 0.9     # Crossover rate

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        while used < budget:
            # Shuffle indices to avoid bias in selection order? 
            # Standard DE iterates 0 to N-1.
            for i in range(N):
                if used >= budget:
                    break
                
                # Select 3 distinct random indices different from i
                while True:
                    r1, r2, r3 = np.random.choice(N, 3, replace=False)
                    if r1 != i and r2 != i and r3 != i:
                        break
                
                # Mutation
                v = X[r1] + F_mut * (X[r2] - X[r3])
                
                # Crossover
                u = X[i].copy()
                # Random crossover mask
                mask = np.random.random(dim) < CR
                # Ensure at least one gene is from v
                if not np.any(mask):
                    j_rand = np.random.randint(dim)
                    mask[j_rand] = True
                u[mask] = v[mask]
                
                # Clip to bounds
                u = np.clip(u, lo, hi)
                
                # Evaluate
                fu = problem(u)
                used += 1
                
                # Selection
                if dominates(fu, F[i]):
                    X[i] = u
                    F[i] = fu
                elif not dominates(F[i], fu):
                    # Incomparable: use distance to ideal point (0,0) as tie-breaker
                    # Ideal point for ZDT1 is (0,0)
                    # But general case: use min of F as ideal approximation?
                    # For ZDT1, (0,0) is the ideal.
                    # Let's use distance to the current best known ideal (min F)
                    # Or just keep the target to maintain diversity.
                    # Let's try: if distance to (0,0) is smaller, replace.
                    # This pushes towards the front.
                    dist_u = np.sum(fu**2)
                    dist_i = np.sum(F[i]**2)
                    if dist_u < dist_i:
                        X[i] = u
                        F[i] = fu
                # Else: F[i] dominates fu, do nothing

        return X
# EVOLVE-BLOCK-END
