"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class BudgetExceeded(Exception):
    pass


class Optimizer:
    """Multi-objective optimizer using Augmented Epsilon-Constraint Method with Nelder-Mead."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # 20s safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (lo + hi) / 2.0

        used = 0

        def check_time():
            return time.time() < deadline

        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                raise BudgetExceeded()
            if not check_time():
                raise BudgetExceeded()
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1
            return f

        # Phase 1: Initial sampling using Latin Hypercube for better coverage
        n_init = 15
        # Latin hypercube sampling
        X_init = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(n_init)
            X_init[:, d] = (perm + rng.random(n_init)) / n_init
        X_init = lo + span * X_init
        
        F_init = []
        for x in X_init:
            try:
                F_init.append(evaluate(x))
            except BudgetExceeded:
                break
        if not F_init:
            return mid
        F_init = np.array(F_init)

        # Get objective bounds from initial samples
        f1_min, f1_max = F_init[:, 0].min(), F_init[:, 0].max()
        f2_min, f2_max = F_init[:, 1].min(), F_init[:, 1].max()

        archive_X = []
        archive_F = []

        # Add initial points to archive
        for i in range(len(F_init)):
            archive_X.append(X_init[i])
            archive_F.append(F_init[i])

        # Find individual optima using Nelder-Mead (derivative-free)
        x_opt1 = None
        x_opt2 = None
        try:
            def obj_f1(x):
                return evaluate(x)[0]
            
            res1 = minimize(obj_f1, mid, method='Nelder-Mead',
                            options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
            if res1.success:
                x_opt1 = np.clip(res1.x, lo, hi)
                f_opt1 = evaluate(x_opt1)
                archive_X.append(x_opt1)
                archive_F.append(f_opt1)
                f1_min = min(f1_min, f_opt1[0])
        except BudgetExceeded:
            pass
        except Exception:
            pass

        try:
            def obj_f2(x):
                return evaluate(x)[1]
            
            res2 = minimize(obj_f2, mid, method='Nelder-Mead',
                            options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
            if res2.success:
                x_opt2 = np.clip(res2.x, lo, hi)
                f_opt2 = evaluate(x_opt2)
                archive_X.append(x_opt2)
                archive_F.append(f_opt2)
                f2_min = min(f2_min, f_opt2[1])
        except BudgetExceeded:
            pass
        except Exception:
            pass

        # Phase 2: Epsilon-Constraint Method to trace Pareto front
        # Use more epsilon levels for denser front tracing
        n_eps = 8
        remaining_budget = self.budget - used
        if remaining_budget < 40:
            n_eps = max(2, remaining_budget // 10)
        
        # Create epsilon values from f2_max to f2_min
        if f2_max > f2_min:
            eps_values = np.linspace(f2_max * 1.02, f2_min * 0.98, n_eps)
        else:
            eps_values = np.array([f2_min])
        eps_values = np.sort(eps_values)[::-1]

        # Weights for starting points
        weights = np.linspace(1.0, 0.0, len(eps_values))

        for i, (eps, w) in enumerate(zip(eps_values, weights)):
            if not check_time() or used >= self.budget - 15:
                break

            # Starting point: blend between the two individual optima if available
            if x_opt1 is not None and x_opt2 is not None:
                x0 = w * x_opt1 + (1 - w) * x_opt2
            else:
                x0 = mid
            x0 = np.clip(x0, lo, hi)

            # Augmented epsilon-constraint: minimize f1 + mu*(f2 - eps)^+
            mu = 5.0

            def objective(x):
                x_clipped = np.clip(x, lo, hi)
                f = evaluate(x_clipped)
                f1, f2 = f[0], f[1]
                penalty = mu * max(0, f2 - eps) ** 2
                return f1 + penalty

            # Use Nelder-Mead (derivative-free) for black-box optimization
            try:
                result = minimize(objective, x0, method='Nelder-Mead',
                                  options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-8, 'adaptive': True})
                if result.success:
                    x_best = np.clip(result.x, lo, hi)
                    f_best = evaluate(x_best)
                    archive_X.append(x_best)
                    archive_F.append(f_best)
            except BudgetExceeded:
                break
            except Exception:
                pass

        # Phase 3: Local refinement - high-precision polish on non-dominated points
        if check_time() and used < self.budget - 5:
            # Get current non-dominated set
            if archive_F:
                F_arr = np.array(archive_F)
                nondom_mask = np.ones(len(F_arr), dtype=bool)
                for i in range(len(F_arr)):
                    if not nondom_mask[i]:
                        continue
                    for j in range(len(F_arr)):
                        if i == j or not nondom_mask[j]:
                            continue
                        if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                            nondom_mask[i] = False
                            break
                
                # Polish each non-dominated point with a tight local search
                for idx in np.where(nondom_mask)[0]:
                    if not check_time() or used >= self.budget - 3:
                        break
                    ax = archive_X[idx]
                    
                    # Small Nelder-Mead refinement around this point
                    def local_obj(x):
                        x_clipped = np.clip(x, lo, hi)
                        f = evaluate(x_clipped)
                        # Use weighted sum with weights based on position on front
                        w1 = 1.0 / (1.0 + max(0, f[1] - f2_min))
                        w2 = 1.0 / (1.0 + max(0, f[0] - f1_min))
                        return w1 * f[0] + w2 * f[1]
                    
                    try:
                        result = minimize(local_obj, ax, method='Nelder-Mead',
                                          options={'maxiter': 20, 'xatol': 1e-6, 'fatol': 1e-9, 'adaptive': True})
                        if result.success:
                            x_polished = np.clip(result.x, lo, hi)
                            f_polished = evaluate(x_polished)
                            archive_X.append(x_polished)
                            archive_F.append(f_polished)
                    except BudgetExceeded:
                        break
                    except Exception:
                        pass

        # Phase 4: Final cleanup - return best point
        if archive_F:
            F_arr = np.array(archive_F)
            # Return the point with best hypervolume contribution (simple heuristic: best f1+f2 sum)
            scores = F_arr[:, 0] + F_arr[:, 1]
            best_idx = np.argmin(scores)
            return archive_X[best_idx]
        return mid
# EVOLVE-BLOCK-END
