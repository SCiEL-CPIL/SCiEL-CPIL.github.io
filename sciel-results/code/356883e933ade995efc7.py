"""Optimizer for BBOB f17 (Schaffer's F7) using island-based CMA-ES with Nelder-Mead refinement."""
import time
import numpy as np


class Optimizer:
    """Island-based CMA-ES with Nelder-Mead refinement for multimodal Schaffer's F7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin below 300s
        n = self.dim
        total_budget = self.budget
        lo, hi = -5.0, 5.0
        rng = np.random.default_rng(42)

        def check_time():
            return time.time() < deadline

        # Nelder-Mead implementation
        def nelder_mead(func, x0, calls_used, max_iter=100):
            """Nelder-Mead simplex optimization."""
            n = len(x0)
            # Initial simplex
            simplex = [x0.copy()]
            for i in range(n):
                pt = x0.copy()
                if pt[i] == 0:
                    pt[i] = 0.25
                else:
                    pt[i] *= 1.05
                pt[i] = np.clip(pt[i], lo, hi)
                simplex.append(pt)
            
            fvals = []
            for i, pt in enumerate(simplex):
                f = func(pt)
                calls_used[0] += 1
                fvals.append(f)
            
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            
            for _ in range(max_iter):
                if not check_time() or calls_used[0] >= total_budget:
                    break
                
                # Sort by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = fvals[-1]
                
                # Reflection
                xr = centroid + alpha * (centroid - worst)
                xr = np.clip(xr, lo, hi)
                fr = func(xr)
                calls_used[0] += 1
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - worst)
                    xe = np.clip(xe, lo, hi)
                    fe = func(xe)
                    calls_used[0] += 1
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
                    # Contraction
                    if fr < worst_f:
                        # Outside contraction
                        xc = centroid + sigma * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - sigma * (xr - centroid)
                    xc = np.clip(xc, lo, hi)
                    fc = func(xc)
                    calls_used[0] += 1
                    if fc < worst_f:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            fvals[i] = func(simplex[i])
                            calls_used[0] += 1
                            if calls_used[0] >= total_budget or not check_time():
                                break
                        
                        if calls_used[0] >= total_budget or not check_time():
                            break
                
                # Check convergence
                if len(fvals) > 1:
                    spread = np.std(fvals)
                    if spread < 1e-10:
                        break
            
            # Return best point
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx], calls_used

        # CMA-ES implementation
        def cma_es(func, x0, sigma0, pop_size, calls_used, max_iter=200):
            """CMA-ES optimization."""
            n = len(x0)
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            C_inv = np.eye(n)
            
            # Strategy parameters
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)
            cc = (4.0 + mueff / n) / ((n + 4.0) ** 2 + mueff)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))
            
            ps = np.zeros(n)
            hsig = 0.0
            damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
            best_x = x.copy()
            best_f = np.inf
            
            for iteration in range(max_iter):
                if not check_time() or calls_used[0] >= total_budget:
                    break
                
                # Sample population
                offspring = []
                for _ in range(pop_size):
                    # Cholesky decomposition of C
                    try:
                        L = np.linalg.cholesky(C)
                    except:
                        L = np.eye(n)
                    
                    z = rng.standard_normal(n)
                    y = L @ z
                    child = x + sigma * y
                    child = np.clip(child, lo, hi)
                    offspring.append(child)
                
                # Evaluate offspring
                fvals = []
                for child in offspring:
                    f = func(child)
                    calls_used[0] += 1
                    fvals.append(f)
                    if f < best_f:
                        best_f = f
                        best_x = child.copy()
                    if not check_time() or calls_used[0] >= total_budget:
                        break
                
                if len(fvals) < pop_size:
                    break
                
                # Sort
                order = np.argsort(fvals)
                offspring = [offspring[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Update mean
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * (offspring[i] - x) / sigma
                
                # Update step-size control
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * C_inv @ y_w
                norm_ps = np.linalg.norm(ps)
                
                # Update covariance matrix
                delta_hsig = (norm_ps / (1.0 - (1.0 - cs) ** (iteration + 1)) / chi_n - 1.0) > 1.4 + 2.0 / (n + 1)
                hsig = 1.0 if delta_hsig else 0.0
                rank_one = np.outer(ps, ps)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    zi = (offspring[i] - x) / sigma
                    rank_mu += weights[i] * np.outer(zi, zi)
                
                sigma *= np.exp(cs / damps * (norm_ps / chi_n - 1.0))
                sigma = max(sigma, 1e-10)
                
                C = (1.0 - cs * cc - c1 - cmu) * C + cs * cc * (rank_one + (hsig - 1.0) * C) + c1 * (rank_one - C) + cmu * rank_mu
                
                # Symmetrize
                C = (C + C.T) / 2.0
                
                # Update C_inv periodically
                if iteration % 10 == 0:
                    try:
                        C_inv = np.linalg.inv(C)
                    except:
                        C_inv = np.eye(n)
                
                # Update mean
                x = x + sigma * y_w
                x = np.clip(x, lo, hi)
            
            return best_x, best_f, calls_used

        # Initial diverse starting points
        starts = []
        starts.append(np.zeros(n))
        
        # Halton sequence for low-discrepancy sampling
        def halton(idx, base):
            f, r = 1.0, 0.0
            while idx > 0:
                f /= base
                r += f * (idx % base)
                idx //= base
            return r
        
        for k in range(1, 15):
            pt = np.array([halton(k, b) for b in range(2, n + 2)])
            starts.append(lo + pt * (hi - lo))
        
        # Random points
        while len(starts) < 20:
            starts.append(rng.uniform(lo, hi, n))
        
        # Island-based approach: run CMA-ES from multiple starts, then NM on best
        calls_used = [0]
        best_x = None
        best_f = np.inf
        
        # Phase 1: Island-based CMA-ES (3 islands)
        n_islands = 3
        island_budget = total_budget // (n_islands + 1)  # Reserve budget for NM
        
        for island in range(n_islands):
            if not check_time() or calls_used[0] >= total_budget:
                break
            
            # Select start point for this island
            start_idx = island * 3  # Use different starts for each island
            if start_idx >= len(starts):
                start_idx = 0
            
            x0 = starts[start_idx].copy()
            
            # Evaluate start
            f0 = func(x0)
            calls_used[0] += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Run CMA-ES from this start
            pop_size = max(8, 4 * n)
            max_iter = island_budget // pop_size
            x_opt, f_opt, calls_used = cma_es(func, x0, 1.5, pop_size, calls_used, max_iter)
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
        
        # Phase 2: Nelder-Mead refinement on best point
        if best_x is not None and check_time() and calls_used[0] < total_budget:
            remaining = total_budget - calls_used[0]
            nm_max_iter = min(100, remaining // (n + 1))
            if nm_max_iter > 5:
                x_nm, f_nm, calls_used = nelder_mead(func, best_x, calls_used, nm_max_iter)
                if f_nm < best_f:
                    best_f = f_nm
                    best_x = x_nm.copy()
        
        # Phase 3: If still have budget, do another round of NM with perturbation
        if best_x is not None and check_time() and calls_used[0] < total_budget:
            remaining = total_budget - calls_used[0]
            if remaining > 20:
                # Perturb best point slightly
                x_pert = best_x + 0.1 * rng.standard_normal(n)
                x_pert = np.clip(x_pert, lo, hi)
                f_pert = func(x_pert)
                calls_used[0] += 1
                
                if f_pert < best_f:
                    best_f = f_pert
                    best_x = x_pert.copy()
                
                # NM from perturbed point
                if check_time() and calls_used[0] < total_budget:
                    remaining = total_budget - calls_used[0]
                    nm_max_iter = min(50, remaining // (n + 1))
                    if nm_max_iter > 5:
                        x_nm2, f_nm2, calls_used = nelder_mead(func, best_x, calls_used, nm_max_iter)
                        if f_nm2 < best_f:
                            best_f = f_nm2
                            best_x = x_nm2.copy()
        
        # Final: use remaining budget for more NM iterations if possible
        if best_x is not None and check_time() and calls_used[0] < total_budget:
            remaining = total_budget - calls_used[0]
            if remaining > 10:
                nm_max_iter = min(30, remaining // (n + 1))
                if nm_max_iter > 3:
                    x_final, f_final, calls_used = nelder_mead(func, best_x, calls_used, nm_max_iter)
                    if f_final < best_f:
                        best_f = f_final
                        best_x = x_final.copy()
        
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_used[0] += 1
        
        return best_x, best_f
