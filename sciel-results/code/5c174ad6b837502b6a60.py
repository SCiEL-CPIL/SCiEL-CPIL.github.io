import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Split budget
        n_random = 50
        n_local = 70
        n_sample = self.budget - n_random - n_local # 80
        
        evals_used = 0
        
        def eval_g(x_tail):
            nonlocal evals_used
            x = np.concatenate(([0.0], x_tail))
            x = np.clip(x, lo, hi)
            f = problem(x)
            evals_used += 1
            return f[1]
            
        candidates = []
        
        # Phase 1: Random Search
        for _ in range(n_random):
            if evals_used >= self.budget:
                break
            x_tail = lo[1:] + (hi[1:] - lo[1:]) * np.random.random(self.dim - 1)
            g = eval_g(x_tail)
            candidates.append((g, x_tail))
            
        # Sort and keep best 5
        candidates.sort(key=lambda x: x[0])
        best_candidates = candidates[:5]
        
        # Phase 2: Local Search on best candidates
        # Distribute n_local evaluations among them
        evals_per_candidate = n_local // len(best_candidates)
        
        for g, x_tail in best_candidates:
            current_tail = x_tail.copy()
            current_g = g
            step = 0.1
            local_evals = 0
            
            while local_evals < evals_per_candidate and step > 0.001:
                improved = False
                # Coordinate descent
                for i in range(self.dim - 1):
                    if local_evals >= evals_per_candidate:
                        break
                    # Positive
                    new_tail = current_tail.copy()
                    new_tail[i] += step
                    new_tail = np.clip(new_tail, lo[1:], hi[1:])
                    new_g = eval_g(new_tail)
                    local_evals += 1
                    if new_g < current_g:
                        current_g = new_g
                        current_tail = new_tail
                        improved = True
                    # Negative
                    if local_evals >= evals_per_candidate:
                        break
                    new_tail = current_tail.copy()
                    new_tail[i] -= step
                    new_tail = np.clip(new_tail, lo[1:], hi[1:])
                    new_g = eval_g(new_tail)
                    local_evals += 1
                    if new_g < current_g:
                        current_g = new_g
                        current_tail = new_tail
                        improved = True
                if improved:
                    step *= 0.8
                else:
                    step *= 0.5
                    
            # Update global best if improved
            if current_g < best_candidates[0][0]:
                best_candidates[0] = (current_g, current_tail)
                
        best_tail = best_candidates[0][1]
        
        # Phase 3: Sample Front
        remaining_budget = self.budget - evals_used
        if remaining_budget > 0:
            x1_values = np.linspace(0.0, 1.0, remaining_budget)
            for x1 in x1_values:
                x = np.concatenate(([x1], best_tail))
                x = np.clip(x, lo, hi)
                problem(x)
                
        return None
