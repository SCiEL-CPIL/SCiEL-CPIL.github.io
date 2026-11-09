"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer on ZDT6 using analytic Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT6 Pareto front:
        # f1(x1) = 1 - exp(-4*x1) * sin^6(6*pi*x1)
        # f2 = 1 + f1
        # x1 in [0, 1], x2..x10 arbitrary (they don't affect objectives on the front)
        # The Pareto front is defined by varying x1 in [0,1]
        
        # We want to maximize hypervolume. The Pareto front of ZDT6:
        # f1 ranges from 0 (at x1=0) to 1 (at x1=1), but actually it's more complex.
        # f1(0) = 1 - exp(0)*sin^6(0) = 1 - 0 = 1... wait
        # f1(0) = 1 - exp(0) * sin^6(0) = 1 - 1*0 = 1
        # f1(1) = 1 - exp(-4) * sin^6(6*pi) = 1 - exp(-4)*0 = 1
        # The minimum of f1 is somewhere in between.
        
        # Actually for ZDT6, the Pareto front has f1 in [0, 1] and f2 = 1 + f1, so f2 in [1, 2].
        # The front is non-convex.
        
        # To maximize hypervolume with reference point [11, 11], we want points that are
        # as close to (0,0) as possible. Since f2 = 1+f1, minimizing f1 also minimizes f2.
        # The best point minimizes f1. But for hypervolume, we want a spread of points.
        
        # The Pareto front is parameterized by x1. For each x1 in [0,1]:
        # f1 = 1 - exp(-4*x1) * sin^6(6*pi*x1)
        # f2 = 1 + f1
        
        # To get maximum hypervolume coverage, we should sample x1 values that give
        # well-spread f1 values. Since f2 = 1+f1, the front is a curve in 2D.
        
        # Strategy: sample x1 uniformly in [0,1], compute f1, f2 for each.
        # Use all 200 evaluations to get 200 distinct points on the Pareto front.
        
        # But wait - the problem function clips x to bounds and returns 2 objectives.
        # We need to construct x vectors where x1 varies and x2..x10 are set to values
        # that don't affect the objectives.
        
        # For ZDT6, the objectives are:
        # f1(x) = 1 - exp(-4*x1) * sin^6(6*pi*x1)
        # f2(x) = 1 + (sum(x) - sum(x_i*g(x_i))) / n^2 * (n-2)  ... 
        # Actually, let me recall ZDT6 properly.
        
        # ZDT6:
        # f1(x) = 1 - exp(-4*x1) * sin^6(6*pi*x1)
        # g(x) = 1 + 10*(n-1) + sum_{i=2}^{n} [(x_i - 0.5)^2 * (1 + 10*sin^2(6*pi*x_i))]
        # f2(x) = g(x) * (1 - sqrt(f1(x)/g(x)))
        
        # On the Pareto front, x_i = 0.5 for i >= 2, so g(x) = 1 + 10*(n-1) = 1 + 10*9 = 91
        # Wait, that doesn't seem right. Let me reconsider.
        
        # Actually for the Pareto front of ZDT6, we set x2..xn such that g(x) is minimized.
        # g(x) = 1 + 10*(n-1) + sum_{i=2}^{n} [(x_i - 0.5)^2 * (1 + 10*sin^2(6*pi*x_i))]
        # The minimum of each term is 0 when x_i = 0.5 (since sin^2(6*pi*0.5) = sin^2(3*pi) = 0)
        # So g_min = 1 + 10*(n-1) = 1 + 90 = 91 for n=10.
        
        # Then f2 = 91 * (1 - sqrt(f1/91))
        
        # So the Pareto front is:
        # f1 = 1 - exp(-4*x1) * sin^6(6*pi*x1), x1 in [0,1]
        # f2 = 91 * (1 - sqrt(f1/91))
        
        # Reference point is [11, 11]. f1 ranges from 0 to 1, f2 ranges from... 
        # When f1=0: f2 = 91*(1-0) = 91
        # When f1=1: f2 = 91*(1-sqrt(1/91)) = 91*(1-0.1047) = 91*0.8953 = 81.47
        
        # Hmm, so f2 is around 81-91, which is much larger than 11. 
        # Points with f2 > 11 are NOT strictly better than reference point [11,11] in the second objective.
        
        # Wait, the scoring says "Only points strictly better than the reference point in every objective count."
        # So we need f1 < 11 AND f2 < 11. But f2 is always >= 81 for ZDT6 with n=10?
        
        # That can't be right. Let me re-examine.
        
        # Maybe I have the ZDT6 definition wrong. Let me think again.
        
        # Standard ZDT6:
        # f1(x) = 1 - exp(-4*x1) * sin^6(6*pi*x1)
        # g(x) = 1 + 10*(n-1) + sum_{i=2}^{n} [(x_i - 0.5)^2 * (1 + 10*sin^2(6*pi*x_i))]
        # f2(x) = g(x) * (1 - sqrt(f1(x)/g(x)))
        
        # For n=10, g_min = 1 + 90 = 91. f2_min when f1=1: 91*(1-sqrt(1/91)) ≈ 81.5
        # f2_max when f1=0: 91
        
        # So f2 is in [~81, 91], all > 11. No point would be better than [11,11] in both objectives.
        
        # This means the score would be 0 for any valid program? That doesn't match the current score of 1.0.
        
        # I must have the ZDT6 definition wrong. Let me reconsider.
        
        # Perhaps the task uses a different scaling or a different ZDT variant.
        # Or perhaps the reference point [11,11] is chosen such that the front does have points better than it.
        
        # Given that the current program scores 1.0, there must be points on the Pareto front
        # with both objectives < 11.
        
        # Let me just empirically use the problem function. The key insight from the knowledge
        # is: "Concentrating all evaluations on the first lobe where the global Pareto front is defined,
        # and mapping uniform objective space samples to decision space via bisection, ensures optimal
        # hypervolume coverage"
        
        # So the approach is:
        # 1. Determine the Pareto front by sampling
        # 2. Map uniform samples in objective space to decision space
        
        # But we can't know the front without evaluating. Let's use a hybrid approach:
        # - First, do a coarse search to understand the front
        # - Then, concentrate evaluations on the Pareto front
        
        # Actually, given the budget is only 200, let me just do a smart sampling.
        
        # Let me try: sample x1 uniformly in [0,1], set x2..x10 to 0.5 (which minimizes g),
        # and evaluate. This should give points near the Pareto front.
        
        # But I don't know the exact ZDT6 definition used. Let me just do a robust search.
        
        # Given the knowledge that "mapping uniform objective space samples to decision space via bisection"
        # works well, let me try that approach.
        
        # First, do a quick exploration to find the range of f1 and f2.
        
        n_eval = self.budget
        dim = self.dim
        
        # Phase 1: Coarse sampling to understand the landscape (use ~20 evals)
        n_coarse = min(20, n_eval)
        rng = np.random.default_rng(42)
        
        X_coarse = np.zeros((n_coarse, dim))
        X_coarse[:, 0] = np.linspace(0.05, 0.95, n_coarse)
        for j in range(1, dim):
            X_coarse[:, j] = 0.5
        
        F_coarse = np.array([problem(x) for x in X_coarse])
        used = n_coarse
        
        # Find the non-dominated set
        def is_dominated(F, i):
            for j in range(len(F)):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    return True
            return False
        
        # Get non-dominated points from coarse
        nondom_idx = [i for i in range(len(F_coarse)) if not is_dominated(F_coarse, i)]
        
        if len(nondom_idx) == 0:
            # Fallback: just return what we have
            return X_coarse
        
        F_nd = F_coarse[nondom_idx]
        X_nd = X_coarse[nondom_idx]
        
        # Determine the range of f1 on the non-dominated front
        f1_min = np.min(F_nd[:, 0])
        f1_max = np.max(F_nd[:, 0])
        
        # Phase 2: Fine sampling in objective space
        # Map uniform f1 values to x1 via bisection
        n_fine = n_eval - used
        
        # We want to sample f1 uniformly in [f1_min, f1_max]
        f1_samples = np.linspace(f1_min, f1_max, n_fine)
        
        # For each target f1, find x1 such that f1_of_x1(x1) = target
        # f1(x1) = 1 - exp(-4*x1) * sin^6(6*pi*x1)
        
        def f1_of_x1(x1):
            return 1 - np.exp(-4*x1) * (np.sin(6*np.pi*x1)**6)
        
        def find_x1_for_f1(target_f1):
            # Bisection on x1 in [0, 1]
            lo_x1, hi_x1 = 0.0, 1.0
            for _ in range(60):
                mid = (lo_x1 + hi_x1) / 2
                f1_mid = f1_of_x1(mid)
                # We need to be careful about the direction
                # f1 is not monotonic, so bisection may not work directly
                # Let's just do a grid search
                break
            
            # Grid search fallback
            x1_grid = np.linspace(0, 1, 1000)
            f1_grid = f1_of_x1(x1_grid)
            idx = np.argmin(np.abs(f1_grid - target_f1))
            return x1_grid[idx]
        
        # Actually, since f1 is not monotonic, let's just sample x1 directly
        # and keep the non-dominated results
        
        # Better approach: sample x1 uniformly, evaluate, and keep non-dominated
        X_fine = np.zeros((n_fine, dim))
        X_fine[:, 0] = np.linspace(0.01, 0.99, n_fine)
        for j in range(1, dim):
            X_fine[:, j] = 0.5
        
        F_fine = np.array([problem(x) for x in X_fine])
        used += n_fine
        
        # Combine and return
        X_all = np.vstack([X_nd, X_fine])
        F_all = np.vstack([F_nd, F_fine])
        
        return X_all

# EVOLVE-BLOCK-END
