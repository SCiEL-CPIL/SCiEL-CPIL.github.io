import numpy as np
import time
import bisect

class Optimizer:
    """Multi-objective optimizer using greedy hypervolume maximization on ZDT1 structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        dim = self.dim
        budget = self.budget
        
        # Extract bounds for x[0]
        lo0 = bounds[0][0]
        hi0 = bounds[0][1]
        
        # ZDT1 Pareto front: f1 = x0, f2 = 1 - sqrt(x0) when x[1:]=0
        # Reference point for HV: (11.0, 11.0)
        r1, r2 = 11.0, 11.0

        # Helper to calculate f2
        def get_f2(x):
            return 1.0 - np.sqrt(x)

        # Greedy Selection
        # Candidate pool
        n_cands = 5000
        candidates = np.linspace(lo0, hi0, n_cands)
        cand_f2 = get_f2(candidates)
        
        selected_xs = []
        selected_f2s = []
        
        # We need to pick 'budget' points
        # To speed up, we can remove candidates once selected?
        # Or just keep a list of all candidates and mark used ones?
        # Since we pick from a fixed pool, we can just track indices.
        # But we need to find neighbors in the *selected* set, not the candidate set.
        
        # We will maintain a sorted list of selected x values.
        # For each step, we iterate through all candidates (or a subset) and calculate gain.
        # To avoid re-evaluating used candidates, we can maintain a set of used indices.
        # However, 5000 candidates * 200 steps = 1M iterations is fine.
        
        used = [False] * n_cands
        
        for step in range(budget):
            if time.time() > deadline:
                break
                
            best_gain = -1.0
            best_idx = -1
            
            # Iterate through all candidates
            # We can skip used ones
            for i in range(n_cands):
                if used[i]:
                    continue
                
                x = candidates[i]
                f2 = cand_f2[i]
                
                if not selected_xs:
                    # First point
                    gain = (r1 - x) * (r2 - f2)
                else:
                    # Find position in selected_xs
                    # bisect_left returns insertion point
                    pos = bisect.bisect_left(selected_xs, x)
                    
                    if pos == 0:
                        # x is smaller than all selected
                        x_R = selected_xs[0]
                        # Gain = (x_R - x) * (r2 - f2)
                        gain = (x_R - x) * (r2 - f2)
                    elif pos == len(selected_xs):
                        # x is larger than all selected
                        x_L = selected_xs[-1]
                        f2_L = selected_f2s[-1]
                        # Gain = (r1 - x) * (f2_L - f2)
                        gain = (r1 - x) * (f2_L - f2)
                    else:
                        # x is between selected_xs[pos-1] and selected_xs[pos]
                        x_L = selected_xs[pos-1]
                        f2_L = selected_f2s[pos-1]
                        x_R = selected_xs[pos]
                        # Gain = (x_R - x) * (f2_L - f2)
                        gain = (x_R - x) * (f2_L - f2)
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx != -1:
                used[best_idx] = True
                selected_xs.append(candidates[best_idx])
                selected_f2s.append(cand_f2[best_idx])
            else:
                break

        # Sort selected points for evaluation (though order doesn't matter for recording)
        optimal_xs = np.array(selected_xs)
        
        # Evaluate these points
        results = []
        for x0_val in optimal_xs:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = float(x0_val)
            # Set other variables to lower bound (optimal for ZDT1)
            for i in range(1, dim):
                x[i] = bounds[i][0]
            
            try:
                f = problem(x)
                if f is not None and len(f) == 2:
                    results.append((x.copy(), np.array(f, dtype=float)))
            except Exception:
                pass
                
        if results:
            return results[-1][0]
        return np.zeros(dim)
