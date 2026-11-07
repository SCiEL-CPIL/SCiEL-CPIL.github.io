import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0
        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return None
            calls += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                f_val = func(x_clipped)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global Exploration
        # Use a stratified or random sample to find the basin.
        # f13 minimum is in [-4, 4]^5.
        n_global = 20
        for _ in range(n_global):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            safe_call(x)

        if best_x is None:
            x0 = np.zeros(dim)
            safe_call(x0)
            return best_x, best_f

        # Phase 2: Local Refinement with Nelder-Mead
        # We will run NM multiple times if budget allows.
        # Strategy: Run NM from best_x. If it converges, restart from a perturbed best_x or new random point.
        
        def nelder_mead(x_start, f_start, max_iter=100):
            nonlocal calls, best_f, best_x
            
            # Initialize simplex
            simplex = [x_start.copy()]
            f_vals = [f_start]
            
            # Initial simplex size
            delta = 0.5
            for i in range(dim):
                x_new = x_start.copy()
                x_new[i] += delta
                f_new = safe_call(x_new)
                if f_new is None:
                    return
                simplex.append(x_new)
                f_vals.append(f_new)

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for it in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break

                # Sort
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]

                # Convergence check
                spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                if np.max(spread) < 1e-6 and abs(f_vals[-1] - f_vals[0]) < 1e-10:
                    break

                # Centroid
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = safe_call(x_r)
                if f_r is None:
                    break

                if f_r < f_vals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = safe_call(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_vals[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    f_c = safe_call(x_c)
                    if f_c is None:
                        break
                    if f_c < f_r:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_new = safe_call(simplex[i])
                            if f_new is None:
                                return
                            f_vals[i] = f_new

                # Update global best
                idx_min = np.argmin(f_vals)
                if f_vals[idx_min] < best_f:
                    best_f = f_vals[idx_min]
                    best_x = simplex[idx_min].copy()

        # Run NM
        n_restarts = 5
        for r in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break
            
            if r == 0:
                nelder_mead(best_x.copy(), best_f)
            else:
                # Restart strategy:
                # 1. Try a random point to see if we missed a better basin (unlikely for f13 but safe)
                # 2. Or perturb the current best to escape local minima.
                # For f13, the basin is large. Random restart is good if NM got stuck.
                # But NM rarely gets stuck in f13 if it finds the ridge.
                # Let's try a random point with some probability, otherwise perturb.
                if rng.random() < 0.5:
                    x_new = rng.uniform(-5.0, 5.0, size=dim)
                else:
                    x_new = best_x + rng.normal(0, 0.5, size=dim)
                
                f_new = safe_call(x_new)
                if f_new is None:
                    break
                
                # If the new point is better, restart NM from there
                # If not, we still run NM from there? No, better to restart from best_x if new is worse?
                # Actually, running NM from a worse point is wasteful.
                # Only restart NM from the new point if it's better, or if we suspect we are stuck.
                # If we are stuck, best_f hasn't improved.
                # Let's just run NM from the new point if it's better than the current best.
                if f_new < best_f:
                    nelder_mead(x_new, f_new)
                else:
                    # If not better, we might be in a local min.
                    # Try NM from a perturbed best_x
                    x_pert = best_x + rng.normal(0, 0.1, size=dim)
                    f_pert = safe_call(x_pert)
                    if f_pert is not None:
                        nelder_mead(x_pert, f_pert)

        return best_x, best_f
