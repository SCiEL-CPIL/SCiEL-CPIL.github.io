import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 285.0

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        evaluated = 0
        best_x = None
        best_f = np.inf

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global Sampling
        global_budget = int(budget * 0.2)
        candidates = []
        
        for _ in range(global_budget):
            if not elapsed_ok():
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f = eval_point(x)
            candidates.append((f, x))

        # Sort and pick top 3 distinct
        candidates.sort(key=lambda item: item[0])
        top_starts = []
        for f, x in candidates:
            if len(top_starts) >= 3:
                break
            is_new = True
            for _, existing_x in top_starts:
                if np.linalg.norm(x - existing_x) < 1.0:
                    is_new = False
                    break
            if is_new:
                top_starts.append((f, x))
        
        while len(top_starts) < 3 and candidates:
            f, x = candidates.pop(0)
            top_starts.append((f, x))

        if not top_starts:
            return best_x, best_f

        # Phase 2: Local Refinement
        remaining_budget = budget - evaluated
        allocations = [0.5, 0.3, 0.2]
        if len(top_starts) < 3:
            total_weight = sum(allocations[:len(top_starts)])
            allocations = [a / total_weight for a in allocations[:len(top_starts)]]
        
        for i, (f_start, x_start) in enumerate(top_starts):
            if not elapsed_ok():
                break
            
            budget_alloc = int(remaining_budget * allocations[i])
            if budget_alloc <= 0:
                continue
                
            start_evaluated = evaluated
            self._nelder_mead(func, x_start, budget_alloc, dim, rng, eval_point, elapsed_ok, start_evaluated)
            
            remaining_budget = budget - evaluated
            if evaluated >= budget:
                break

        return best_x, best_f

    def _nelder_mead(self, func, x0, budget, dim, rng, eval_point, elapsed_ok, start_evaluated):
        # Initial Simplex with randomized orientation
        A = rng.normal(size=(dim, dim))
        Q, _ = np.linalg.qr(A)
        alpha = 2.0
        
        simplex = [x0.copy()]
        for i in range(dim):
            v = x0 + alpha * Q[:, i]
            simplex.append(v)
        
        simplex_f = []
        for x in simplex:
            if not elapsed_ok() or (evaluated - start_evaluated >= budget):
                return
            f = eval_point(x)
            simplex_f.append(f)
        
        r = 1.0
        e = 2.0
        c = 0.5
        s = 0.5
        
        while (evaluated - start_evaluated) < budget:
            if not elapsed_ok():
                break
            
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                break

            centroid = np.mean(simplex[:-1], axis=0)
            worst_x = simplex[-1]
            worst_f = simplex_f[-1]
            
            x_r = centroid + r * (centroid - worst_x)
            f_r = eval_point(x_r)
            if (evaluated - start_evaluated) >= budget:
                break

            if f_r < simplex_f[0]:
                x_e = centroid + e * (centroid - worst_x)
                f_e = eval_point(x_e)
                if (evaluated - start_evaluated) >= budget:
                    break
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                if f_r < worst_f:
                    x_c = centroid + c * (x_r - centroid)
                else:
                    x_c = centroid - c * (centroid - worst_x)
                f_c = eval_point(x_c)
                if (evaluated - start_evaluated) >= budget:
                    break
                if f_c < min(f_r, worst_f):
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + s * (simplex[i] - simplex[0])
                        simplex_f[i] = eval_point(simplex[i])
                        if (evaluated - start_evaluated) >= budget:
                            return

