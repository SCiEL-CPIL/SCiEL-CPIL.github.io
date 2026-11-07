"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE for global exploration, (1+1)-ES with covariance for local refinement, NM polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        n = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)

        # Phase 1: Differential Evolution (global exploration)
        pop_size = 8
        F = 0.8
        CR = 0.9
        de_budget = int(budget * 0.55)
        
        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, n))
        fpop = np.array([func(x) for x in pop])
        calls = pop_size
        
        best_idx = np.argmin(fpop)
        best_x = pop[best_idx].copy()
        best_fx = fpop[best_idx]

        while calls < de_budget:
            if time.time() > deadline:
                break
            
            # DE: best/1/bin
            i = rng.integers(pop_size)
            a, b, c = rng.choice(pop_size, 3, replace=False)
            while a == b or a == c or b == c:
                a, b, c = rng.choice(pop_size, 3, replace=False)
            
            # Mutation: v = pop[best] + F * (pop[a] - pop[b])
            v = pop[best_idx] + F * (pop[a] - pop[b])
            
            # Crossover: binomial with CR
            j_rand = rng.integers(n)
            for j in range(n):
                if rng.random() < CR or j == j_rand:
                    v[j] = v[j]
                else:
                    v[j] = pop[i][j]
            
            v = np.clip(v, -5, 5)
            fv = func(v)
            calls += 1
            
            if fv <= fpop[i]:
                pop[i] = v
                fpop[i] = fv
                if fv < best_fx:
                    best_fx = fv
                    best_x = v.copy()
                    best_idx = i
            elif fv < best_fx:
                best_fx = fv
                best_x = v.copy()
                best_idx = i

        # Phase 2: (1+1)-ES with covariance adaptation (local refinement)
        es_budget = int(budget * 0.25)
        es_end = de_budget + es_budget
        
        x = best_x.copy()
        fx = best_fx
        sigma = 0.5
        C = np.eye(n)
        
        window_size = 12
        successes = []
        
        while calls < es_end:
            if time.time() > deadline:
                break
            
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
            
            z = rng.standard_normal(n)
            step = sigma * np.dot(L, z)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls += 1
            
            success = fy < fx
            successes.append(1 if success else 0)
            if len(successes) > window_size:
                successes.pop(0)
            
            if success:
                x = y
                fx = fy
                if fy < best_fx:
                    best_fx = fy
                    best_x = x.copy()
                
                mu_c = 0.2
                C = (1 - mu_c) * C + mu_c * np.outer(z, z)
                trace_C = np.trace(C)
                if trace_C > 1e-10:
                    C = C * (n / trace_C)
            else:
                pass
            
            # Sigma adaptation
            if len(successes) >= 5:
                success_rate = np.mean(successes)
                if success_rate > 0.4:
                    sigma *= 1.06
                elif success_rate < 0.2:
                    sigma *= 0.94
                else:
                    sigma *= 1.01
            sigma = np.clip(sigma, 1e-6, 5.0)
            
            # Periodic re-centering
            if calls % 25 == 0:
                C = 0.85 * C + 0.15 * np.eye(n)
                trace_C = np.trace(C)
                if trace_C > 1e-10:
                    C = C * (n / trace_C)
            
            # Eigenvalue clipping
            if calls % 80 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 0.01, 30.0)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                trace_C = np.trace(C)
                if trace_C > 1e-10:
                    C = C * (n / trace_C)

        # Phase 3: Coordinate descent polish
        if calls < budget and time.time() < deadline:
            coord_budget = min(15, budget - calls)
            x_cd = best_x.copy()
            fx_cd = best_fx
            step_sizes = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            
            for _ in range(coord_budget):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(n):
                    for ss in step_sizes:
                        y_pos = x_cd.copy()
                        y_pos[i] = np.clip(x_cd[i] + ss, -5, 5)
                        fy_pos = func(y_pos)
                        calls += 1
                        if fy_pos < fx_cd:
                            x_cd = y_pos
                            fx_cd = fy_pos
                            improved = True
                            break
                        
                        y_neg = x_cd.copy()
                        y_neg[i] = np.clip(x_cd[i] - ss, -5, 5)
                        fy_neg = func(y_neg)
                        calls += 1
                        if fy_neg < fx_cd:
                            x_cd = y_neg
                            fx_cd = fy_neg
                            improved = True
                            break
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break
                if not improved:
                    break
            
            if fx_cd < best_fx:
                best_fx = fx_cd
                best_x = x_cd.copy()

        # Phase 4: Nelder-Mead polish
        if calls < budget and time.time() < deadline:
            try:
                from scipy.optimize import minimize
                remaining = budget - calls
                if remaining > 10:
                    result = minimize(func, best_x, method='Nelder-Mead',
                                      options={'maxiter': remaining, 'xatol': 1e-12, 'fatol': 1e-14,
                                               'adaptive': True})
                    if result.fun < best_fx:
                        best_fx = result.fun
                        best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
