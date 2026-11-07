"""Multi-objective optimizer on WFG7: Decomposition-based Tchebycheff Scalarization with Local Search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Decomposition-based optimizer using Tchebycheff scalarization and Nelder-Mead local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # 10s safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        center = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0
        
        # 1. Generate Weight Vectors for Tchebycheff Scalarization
        # We use 30 weight vectors to cover the 3-objective space.
        # We generate points on the simplex w1+w2+w3=1, wi >= 0.
        # To ensure good coverage of the convex hull, we use a grid or random sampling 
        # followed by normalization. 
        # Given the small budget, we prioritize diversity.
        
        # Simple uniform distribution on the simplex for 3 objectives:
        # We can sample from a Dirichlet distribution with alpha=1 (uniform) 
        # or use a deterministic grid. 
        # A deterministic grid is more reproducible and ensures coverage.
        # For 3 objectives, the number of points with denominator K is (K+1)(K+2)/2.
        # Let's pick K such that we get ~30 points.
        # K=7 -> 36 points. K=6 -> 28 points.
        # Let's use K=6 (28 points) or K=7 (36 points). 
        # We have 200 budget. 28 points * ~7 evals/point = 196. This fits perfectly.
        
        K = 6
        weights = []
        for i in range(K + 1):
            for j in range(K + 1 - i):
                k = K - i - j
                w = np.array([i, j, k], dtype=np.float64) / K
                weights.append(w)
        
        # We have 28 weights.
        # If we need exactly 30, we can add the center and a corner, but 28 is close enough.
        # Let's stick with 28 to keep the math clean for allocation.
        n_subproblems = len(weights)
        
        # 2. Allocate Budget
        # Total budget: 200
        # Initialization: 1 eval per subproblem = 28 evals.
        # Remaining: 172 evals.
        # Local Search Budget per subproblem: 172 / 28 = 6.14 -> 6 evals.
        # Total: 28 * (1 + 6) = 196 evals.
        
        init_evals = 1
        local_evals = (self.budget - n_subproblems * init_evals) // n_subproblems
        # If local_evals is too small, we just do init.
        
        all_F = []
        used = 0
        
        # Helper to check deadline
        def check_deadline():
            return time.time() > deadline

        # 3. Initialize Sub-problems
        # Start all at the center? Or spread them out?
        # Starting at center is safe but might be far from the front for some weights.
        # Better: Start with a random point or a point biased towards the weight direction?
        # In WFG, the front is usually near the origin in objective space, 
        # but the decision space mapping is complex.
        # Let's initialize each subproblem with a random point within the bounds 
        # to ensure we don't get stuck in a bad local minimum of the scalarized function 
        # if the center is a saddle point.
        # However, random start might waste evals.
        # Compromise: Start at center, but if the first eval is bad, we might need to move.
        # Actually, for WFG7, the center is often a reasonable starting point for local search 
        # if the function is not too multimodal.
        
        # Let's use a slightly more robust initialization: 
        # For each weight, we can try to project the weight onto the decision space? 
        # Too complex. Let's just use the center for all, but add a small random perturbation 
        # to break symmetry? No, that wastes evals.
        
        # Strategy: 
        # 1. Evaluate center once.
        # 2. For each weight, start Nelder-Mead from the center.
        #    Wait, if they all start from the same point, they will initially move in similar directions 
        #    but diverge due to different gradient directions (implicit in the scalarization).
        
        # Let's refine: 
        # We have 200 evals.
        # Let's do a "Warm Start" phase: 
        # Evaluate 10 random points to get a rough idea of the front? 
        # No, decomposition is better.
        
        # Implementation of Nelder-Mead for single objective g(x) = max_i (w_i * |f_i(x) - z_i^*)|
        # We need a reference point z^*. 
        # We can use the ideal point found so far or a fixed reference.
        # Let's use the minimum observed value for each objective as the dynamic reference.
        # This makes the scalarization adaptive.
        
        # However, dynamic reference requires updating after each eval.
        # Simpler: Use a fixed reference point far away, e.g., [10, 10, 10].
        # Or better: Use the "augmented Tchebycheff" with a dynamic ideal point.
        
        # Let's implement a simple local search for each weight:
        # x_current
        # f_current
        # Simplex: x_current, x_current + delta * e_i
        
        # Since we have very few evals per subproblem (6), full Nelder-Mead is overkill.
        # A simple coordinate descent or pattern search might be faster.
        # But Nelder-Mead is robust. 
        # With 6 evals, we can only do 1-2 iterations of NM.
        
        # Alternative: 
        # Use the 200 evals to run a Global Search (like DE) but guided by the weights?
        # No, the directive says "Implement a decomposition-based approach... where each sub-problem is solved via a fast... local search".
        
        # Let's stick to the plan:
        # 1. Evaluate center. (1 eval)
        # 2. For each of the 28 weights:
        #    a. Start x = center + small_random_noise (to avoid identical paths? No, let's keep it deterministic for now, or use rng).
        #    b. Run a limited Nelder-Mead.
        
        # To save evals, let's share the initial center evaluation.
        # f_center = problem(center)
        # used = 1
        # all_F.append(f_center)
        
        # For each weight w:
        #    x0 = center
        #    f0 = f_center
        #    # Simple local search:
        #    # Try moving in the direction of the gradient of the scalarized function?
        #    # We don't have gradients. 
        #    # Let's use a "Random Restart" style local search:
        #    # 1. Evaluate x0.
        #    # 2. Generate 5 random neighbors within a shrinking radius.
        #    # 3. Keep the best.
        
        # This is essentially a "Hill Climbing" with random steps.
        
        # Let's refine the local search to be more effective with 6 evals:
        # 1. Current best x, f.
        # 2. Step size starts at 0.1 * range.
        # 3. Generate 1 candidate: x_new = x + step * (random_direction).
        # 4. If f_new < f_scalar(x_new), accept.
        # 5. Repeat.
        
        # Actually, with 28 subproblems and 6 evals each, we are doing 168 local search steps.
        # Total 169 evals. We have 31 left.
        # We can use the remaining 31 evals to polish the best solutions found by the subproblems.
        
        # Revised Plan:
        # 1. Evaluate center. (1)
        # 2. For each of 28 weights:
        #    - Run 5-step local search. (5 * 28 = 140)
        #    Total so far: 141.
        # 3. Collect all evaluated points.
        # 4. Use remaining 59 evals to refine the top 10 non-dominated points with a tighter local search.
        
        # This hybrid approach uses decomposition for coverage and local polish for quality.
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Decomposition-based Local Search
        # Initialize all at center
        x_center = center.copy()
        f_center = np.array(problem(x_center), dtype=np.float64)
        used = 1
        all_F = [f_center]
        
        # Ideal point estimate (min values seen so far)
        ideal = f_center.copy()
        
        # For each weight, perform local search
        # We will store the best x and f for each weight
        best_xs = [x_center.copy() for _ in range(n_subproblems)]
        best_fs = [f_center.copy() for _ in range(n_subproblems)]
        
        # Local search function for a specific weight
        def scalarize(f, w, ideal):
            # Tchebycheff: max_i (w_i * |f_i - ideal_i|)
            # Since we are minimizing, and f_i >= ideal_i usually, we can use w_i * (f_i - ideal_i)
            # But to be safe, use abs.
            return np.max(w * np.abs(f - ideal))
        
        # We will update the ideal point as we find better values.
        
        for w_idx, w in enumerate(weights):
            if check_deadline() or used >= self.budget:
                break
                
            x_curr = best_xs[w_idx].copy()
            f_curr = best_fs[w_idx].copy()
            curr_scalar = scalarize(f_curr, w, ideal)
            
            # Local search steps
            # We have ~5-6 evals allocated per subproblem. Let's do 5.
            for step in range(5):
                if check_deadline() or used >= self.budget:
                    break
                
                # Generate a random direction
                direction = rng.normal(size=self.dim)
                direction /= np.linalg.norm(direction)
                
                # Adaptive step size
                # Start with 0.2 * range, shrink if no improvement
                step_size = 0.2 * np.max(half_range)
                # Simple shrinkage: divide by 2 each step if we don't improve?
                # Or just use a fixed small step?
                # Let's use a fixed step size of 0.1 * range for these few steps.
                step_size = 0.1 * np.max(half_range)
                
                x_trial = x_curr + step_size * direction
                x_trial = np.clip(x_trial, lo, hi)
                
                f_trial = np.array(problem(x_trial), dtype=np.float64)
                used += 1
                all_F.append(f_trial)
                
                # Update ideal point
                new_ideal = np.minimum(ideal, f_trial)
                ideal = new_ideal
                
                # Check improvement in scalarized objective
                # Note: Ideal point changed, so we should re-calculate curr_scalar?
                # For simplicity, let's just check if f_trial is better in the scalarized sense 
                # using the OLD ideal to avoid oscillation, or just check if it dominates?
                # Let's check if scalarize(f_trial, w, ideal) < scalarize(f_curr, w, ideal)
                # Using the updated ideal might be more aggressive.
                
                trial_scalar = scalarize(f_trial, w, ideal)
                curr_scalar_new = scalarize(f_curr, w, ideal)
                
                if trial_scalar < curr_scalar_new:
                    x_curr = x_trial
                    f_curr = f_trial
                    curr_scalar = trial_scalar
                    best_xs[w_idx] = x_curr
                    best_fs[w_idx] = f_curr
                # Else, keep current. We don't shrink step size in this simple loop.
                # This is a random walk hill climbing.
        
        # Phase 2: Polish
        # We have used ~141 evals. Remaining: 200 - 141 = 59.
        # Identify the most promising points from all_F.
        # Let's find the non-dominated set of all_F.
        
        if len(all_F) > 0:
            F_array = np.array(all_F)
            
            # Simple non-dominated filter
            # A point i is dominated if there exists j such that F[j] <= F[i] and F[j] != F[i]
            # We want to keep the non-dominated ones.
            # Since N is ~150, O(N^2) is fine.
            
            n_points = len(F_array)
            dominated = np.zeros(n_points, dtype=bool)
            
            for i in range(n_points):
                if dominated[i]:
                    continue
                for j in range(n_points):
                    if i == j or dominated[j]:
                        continue
                    # Does j dominate i?
                    if np.all(F_array[j] <= F_array[i]) and np.any(F_array[j] < F_array[i]):
                        dominated[i] = True
                        break
            
            non_dom_indices = np.where(~dominated)[0]
            
            # Sort non-dominated points by their "quality" (e.g., sum of objectives or distance to ideal)
            # Let's use sum of objectives as a proxy for proximity to the front.
            if len(non_dom_indices) > 0:
                sums = np.sum(F_array[non_dom_indices], axis=1)
                # Get the top 10 best (lowest sum) non-dominated points
                top_k = min(10, len(non_dom_indices))
                # argsort sums
                sorted_indices = non_dom_indices[np.argsort(sums)[:top_k]]
                
                # Polish these top points with a tighter local search
                # Use remaining budget
                remaining_budget = self.budget - used
                
                for idx in sorted_indices:
                    if remaining_budget <= 0 or check_deadline():
                        break
                    
                    x_start = best_xs[0] # This is wrong, we need the x associated with F_array[idx]
                    # We didn't store X for all_F. We only stored best_xs for the subproblems.
                    # This is a flaw in the plan. We need to store X for all evaluated points.
                    
                    # Let's restart the logic slightly to store X.
                    # I will modify the code to store X in a list parallel to all_F.
                    
                    pass # Placeholder, will rewrite below

        # REWRITTEN LOGIC WITH X STORAGE
        
        # Reset for clean implementation
        all_F = []
        all_X = []
        used = 0
        ideal = np.array([10.0, 10.0, 10.0]) # Initial guess for ideal point
        
        # Phase 1: Decomposition
        x_center = center.copy()
        f_center = np.array(problem(x_center), dtype=np.float64)
        used += 1
        all_F.append(f_center)
        all_X.append(x_center)
        ideal = np.minimum(ideal, f_center)
        
        best_xs = [x_center.copy() for _ in range(n_subproblems)]
        best_fs = [f_center.copy() for _ in range(n_subproblems)]
        
        for w_idx, w in enumerate(weights):
            if check_deadline() or used >= self.budget:
                break
                
            x_curr = best_xs[w_idx].copy()
            f_curr = best_fs[w_idx].copy()
            
            for step in range(5):
                if check_deadline() or used >= self.budget:
                    break
                
                direction = rng.normal(size=self.dim)
                direction /= np.linalg.norm(direction)
                step_size = 0.1 * np.max(half_range)
                
                x_trial = x_curr + step_size * direction
                x_trial = np.clip(x_trial, lo, hi)
                
                f_trial = np.array(problem(x_trial), dtype=np.float64)
                used += 1
                all_F.append(f_trial)
                all_X.append(x_trial)
                
                ideal = np.minimum(ideal, f_trial)
                
                trial_scalar = np.max(w * np.abs(f_trial - ideal))
                curr_scalar = np.max(w * np.abs(f_curr - ideal))
                
                if trial_scalar < curr_scalar:
                    x_curr = x_trial
                    f_curr = f_trial
                    best_xs[w_idx] = x_curr
                    best_fs[w_idx] = f_curr
        
        # Phase 2: Polish Top Non-Dominated Points
        if len(all_F) > 0:
            F_array = np.array(all_F)
            n_points = len(F_array)
            dominated = np.zeros(n_points, dtype=bool)
            
            for i in range(n_points):
                if dominated[i]:
                    continue
                for j in range(n_points):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F_array[j] <= F_array[i]) and np.any(F_array[j] < F_array[i]):
                        dominated[i] = True
                        break
            
            non_dom_indices = np.where(~dominated)[0]
            
            if len(non_dom_indices) > 0:
                sums = np.sum(F_array[non_dom_indices], axis=1)
                top_k = min(10, len(non_dom_indices))
                sorted_indices = non_dom_indices[np.argsort(sums)[:top_k]]
                
                remaining_budget = self.budget - used
                evals_per_polish = max(1, remaining_budget // len(sorted_indices))
                
                for idx in sorted_indices:
                    if used >= self.budget or check_deadline():
                        break
                    
                    x_curr = all_X[idx].copy()
                    f_curr = F_array[idx].copy()
                    
                    # Tighter local search
                    for step in range(evals_per_polish):
                        if used >= self.budget or check_deadline():
                            break
                        
                        # Smaller step size
                        direction = rng.normal(size=self.dim)
                        direction /= np.linalg.norm(direction)
                        step_size = 0.02 * np.max(half_range)
                        
                        x_trial = x_curr + step_size * direction
                        x_trial = np.clip(x_trial, lo, hi)
                        
                        f_trial = np.array(problem(x_trial), dtype=np.float64)
                        used += 1
                        all_F.append(f_trial)
                        all_X.append(x_trial)
                        
                        ideal = np.minimum(ideal, f_trial)
                        
                        # For polishing, we just want to improve the sum or any objective
                        # Let's use the sum of objectives as the guide for polishing
                        if np.sum(f_trial) < np.sum(f_curr):
                            x_curr = x_trial
                            f_curr = f_trial

        return all_X
# EVOLVE-BLOCK-END
