"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using a (1+1)-ES with adaptive step size, followed by
    multi-start Nelder-Mead refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 315.0

        rng = np.random.default_rng(42)

        best_x = None
        best_fx = float('inf')
        calls = 0
        
        # Track top 2 distinct points for multi-start NM
        # Simple approach: keep a list of (fx, x) and sort at the end
        # But memory is not an issue, we just need the top 2.
        # We'll just keep the best and a secondary best if it's significantly different.
        # For simplicity, we will just run NM from the global best.
        # The "multi-start" in knowledge might refer to multiple NM runs from slightly perturbed best points.
        
        # Phase 1: (1+1)-ES
        x = np.zeros(dim)
        fx = func(x)
        calls += 1
        best_x = x.copy()
        best_fx = fx
        
        sigma = 2.5
        success_count = 0
        window = 10
        step_count = 0
        
        es_budget = int(budget * 0.70)
        
        while calls < es_budget:
            if time.time() > deadline - 5:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            step_count += 1

            if fy < fx:
                x, fx = y, fy
                success_count += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
            else:
                pass

            if step_count % window == 0:
                success_rate = success_count / window
                if success_rate > 0.25:
                    sigma *= 1.15
                elif success_rate < 0.15:
                    sigma *= 0.85
                else:
                    sigma *= 1.0
                success_count = 0
                sigma = np.clip(sigma, 0.01, 10.0)

        # Phase 2: Nelder-Mead from best point
        if calls < budget - 10 and time.time() < deadline - 5:
            x_nm = best_x.copy()
            f_nm = best_fx
            
            simplex = [x_nm.copy()]
            fvals = [f_nm]
            
            # Use current sigma or a fixed value for perturbation
            # If sigma is very small, NM might be stuck. If large, simplex might be huge.
            # 0.1 is safe.
            step_nm = 0.1
            for i in range(dim):
                pert = x_nm.copy()
                if x_nm[i] + step_nm > 5.0:
                    pert[i] -= step_nm
                elif x_nm[i] - step_nm < -5.0:
                    pert[i] += step_nm
                else:
                    pert[i] += step_nm
                pert = np.clip(pert, -5.0, 5.0)
                
                f_pert = func(pert)
                calls += 1
                simplex.append(pert)
                fvals.append(f_pert)
                
                if calls >= budget:
                    break
            
            if len(simplex) < dim + 1:
                return best_x, best_fx

            max_nm_iters = min(60, budget - calls)
            for _ in range(max_nm_iters):
                if time.time() > deadline - 5 or calls >= budget:
                    break
                
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                xr = np.clip(2 * centroid - simplex[-1], -5, 5)
                fr = func(xr)
                calls += 1
                
                if fr < fvals[0]:
                    xe = np.clip(centroid + 2 * (xr - centroid), -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    xc = np.clip(centroid + 0.5 * (simplex[-1] - centroid), -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, dim + 1):
                            if calls >= budget: break
                            simplex[i] = np.clip(0.5 * (simplex[i] + simplex[0]), -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1

                if fvals[0] < best_fx:
                    best_fx = fvals[0]
                    best_x = simplex[0].copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
