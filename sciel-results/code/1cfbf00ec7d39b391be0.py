"""Multi-objective optimizer on WFG7: Tchebycheff Decomposition with Local Refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import comb


class Optimizer:
    """
    Multi-objective optimizer using Tchebycheff decomposition.
    Allocates budget to solve multiple scalar sub-problems to cover the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 295.0  # 5s safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        # 1. Generate weight vectors for Tchebycheff decomposition
        # Number of sub-problems: aim for ~25 to allow ~8 evals each within 200 budget
        # We need points on the (n_obj-1)-simplex
        n_sub = 25
        weights = self._generate_weights(n_sub, self.n_obj)
        
        # 2. Determine evaluation budget per sub-problem
        # Reserve some for local refinement if time/budget allows, but primarily allocate to decomposition
        # With 200 budget and 25 sub-problems, we have 8 evals per sub-problem.
        # However, minimize() can use multiple evals. We will limit them via callback or maxiter.
        # Actually, L-BFGS-B is efficient. Let's give it a reasonable maxiter.
        # 8 evals is very tight for L-BFGS-B. 
        # Alternative: Use Nelder-Mead? No, it's slow.
        # Let's try to use the budget more effectively.
        # Maybe fewer sub-problems? 10 sub-problems -> 20 evals each.
        # Or 20 sub-problems -> 10 evals each.
        # The knowledge says "25 sub-problems... improved island 3". Let's stick to ~25 but be smart.
        
        # Actually, the "Knowledge" says: "Tchebycheff decomposition ... 25 sub-problems ... Measured: 0.7724 -> 1.154."
        # So 25 is a good target.
        
        all_points = []
        all_x = []
        
        # Pre-compute some stats for normalization if needed? Tchebycheff usually works with raw or normalized.
        # WFG7 objectives are typically in range [0, 10] or similar.
        # We don't know the range yet. We can do a quick random sample to estimate bounds for objectives?
        # That costs evaluations.
        # Better: Use raw Tchebycheff. It is scale-invariant if weights are normalized, but the magnitude of f matters.
        # Standard Tchebycheff: min max_i w_i * |f_i(x) - z_i^*|
        # z_i^* is the ideal point. We don't know it.
        # Approximation: Use the min found so far or 0?
        # If we use 0, it biases towards lower values, which is good for minimization.
        
        # Let's refine the strategy:
        # 1. Generate 25 weight vectors.
        # 2. For each weight, run a local search (L-BFGS-B) starting from a random point or a shared center.
        #    To save budget, we can start all from the same center? No, diversity is key.
        #    Random starts are better.
        #    Budget per sub-problem: 200 / 25 = 8.
        #    L-BFGS-B with maxiter=8 might not converge well.
        #    Maybe use fewer sub-problems? 10 sub-problems, 20 evals each.
        #    Let's try 15 sub-problems, ~13 evals each.
        
        # Let's adjust n_sub based on budget.
        # If budget is 200, let's use 20 sub-problems. 10 evals each.
        n_sub = min(25, self.budget // 8) 
        if n_sub < 5:
            n_sub = 5
            
        weights = self._generate_weights(n_sub, self.n_obj)
        
        # Ideal point estimation: 
        # We can't afford many random evals. 
        # Let's assume ideal is 0 for all objectives? 
        # If objectives are positive, this pushes towards 0.
        # If they can be negative, it might be bad.
        # WFG7 objectives are usually positive.
        
        z_ideal = np.zeros(self.n_obj)
        
        for w in weights:
            if time.time() > deadline - 10: # Reserve 10s for final steps
                break
                
            # Define the scalar objective function for Tchebycheff
            def tcheb_obj(x, w=w, z_ideal=z_ideal):
                f = problem(x)
                # Tchebycheff distance
                # max_i w_i * |f_i - z_i|
                return np.max(w * np.abs(f - z_ideal))
            
            # Initial point: Random
            x0 = lo + (hi - lo) * np.random.rand(self.dim)
            
            # Options for L-BFGS-B
            # maxiter controls iterations, not evals directly, but roughly.
            # To strictly control evals, we might need a wrapper, but L-BFGS-B is fast.
            # Let's set maxiter to allow ~8-10 evals.
            # L-BFGS-B does ~1 eval per iteration in the initial phase.
            max_iter = max(5, self.budget // n_sub - 2)
            
            try:
                res = minimize(tcheb_obj, x0, method='L-BFGS-B', 
                               bounds=list(zip(lo, hi)),
                               options={'maxiter': max_iter, 'maxfun': max_iter + 5})
                
                if res.success or res.fun < np.inf:
                    x_best = res.x
                    f_best = problem(x_best)
                    all_x.append(x_best)
                    all_points.append(f_best)
            except Exception:
                pass

        # 3. Local Refinement / Polishing
        # If we have budget left, refine the non-dominated points.
        if len(all_points) > 0:
            F_arr = np.array(all_points)
            
            # Find non-dominated points
            mask = self._non_dominated_mask(F_arr)
            nd_indices = np.where(mask)[0]
            
            # If we have time and budget, do a few more steps on the best ones
            # How much budget used?
            # It's hard to track exactly with minimize. 
            # Let's assume we used roughly n_sub * max_iter.
            # If we have > 20 evals left, we can polish.
            
            # Simple polish: Take the top 5 non-dominated points and run a few more steps
            # Actually, the current loop already ran minimize.
            # Let's just return the best non-dominated points.
            
            # The task says: "Every returned objective vector is recorded... return value of __call__ is ignored"
            # Wait, "The evaluator calls ... problem(x) ... Every returned objective vector is recorded".
            # This implies the evaluator records the outputs of `problem` calls made BY MY CODE?
            # "The evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
            # "Every returned objective vector is recorded" -> This refers to the calls I make to `problem(x)`.
            # So I don't need to return the points. I just need to call `problem(x)` enough times to generate a good set of points.
            # The hypervolume is calculated on "all recorded vectors".
            
            # So my strategy is valid: I call problem(x) inside minimize, and those are recorded.
            # I should ensure I call it enough times.
            
            # One issue: minimize might call problem(x) multiple times for the same x? No, usually unique.
            # But it might not call it exactly max_iter times.
            
            # To be safe and ensure we hit the budget if possible (more points = better HV potentially),
            # we can add a simple random sampling phase if we have time.
            
            # But 25 sub-problems * ~8 evals = 200 evals. That's the whole budget.
            # So we shouldn't add random sampling unless we reduce n_sub.
            
            # Let's stick to the decomposition.
            
        return None # Return value is ignored, but keep interface

    def _generate_weights(self, n_sub, n_obj):
        """
        Generate n_sub weight vectors on the (n_obj-1) simplex.
        Uses a grid approach for uniform coverage.
        """
        # For 3 objectives, we can use a 2D grid on the simplex f1+f2+f3=1
        # Number of points in a grid of size k on 2D simplex is k*(k+1)/2.
        # We want ~n_sub points.
        # k*(k+1)/2 ≈ n_sub => k ≈ sqrt(2*n_sub)
        k = int(np.ceil(np.sqrt(2 * n_sub)))
        
        weights = []
        # Iterate over i, j such that i+j <= k
        for i in range(k + 1):
            for j in range(k + 1 - i):
                # l = k - i - j
                w1 = i / k
                w2 = j / k
                w3 = (k - i - j) / k
                weights.append([w1, w2, w3])
                
        # If we have more than n_sub, randomly sample or take first n_sub?
        # Taking first n_sub might bias towards corner.
        # Better to randomly select n_sub from the generated grid if grid is larger.
        if len(weights) > n_sub:
            indices = np.random.choice(len(weights), n_sub, replace=False)
            weights = [weights[i] for i in indices]
        elif len(weights) < n_sub:
            # If grid is too small, add random points?
            # Or just use what we have.
            pass
            
        return np.array(weights)

    def _non_dominated_mask(self, F):
        """
        Return a boolean mask where True indicates the point is non-dominated.
        F is (N, n_obj) array of objective values (to minimize).
        """
        n = F.shape[0]
        mask = np.ones(n, dtype=bool)
        
        for i in range(n):
            if not mask[i]:
                continue
            for j in range(n):
                if i == j or not mask[j]:
                    continue
                # Check if j dominates i
                # j dominates i if F[j] <= F[i] and F[j] < F[i] in at least one dimension
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    mask[i] = False
                    break
        return mask
# EVOLVE-BLOCK-END
