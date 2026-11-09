"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with LHS initialization and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _lhs(self, n, dim, rng):
        """Latin Hypercube Sampling in [-5, 5]^dim."""
        points = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            points[:, d] = (perm + rng.random(n)) / n * 10.0 - 5.0
        return points

    def _nelder_mead(self, func, x0, budget, rng):
        """Nelder-Mead simplex method with adaptive restart."""
        dim = self.dim
        n = dim + 1
        # Initialize simplex
        simplex = np.zeros((n, dim))
        simplex[0] = x0
        fvals = np.zeros(n)
        fvals[0] = func(x0)
        
        # Spread other points
        for i in range(1, n):
            simplex[i] = x0 + 0.5 * rng.standard_normal(dim)
            simplex[i] = np.clip(simplex[i], -5, 5)
            fvals[i] = func(simplex[i])
        
        used = n
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        min_step = 1e-10
        
        while used < budget:
            # Sort simplex
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Check convergence
            if np.ptp(fvals) < 1e-12:
                break
            
            # Centroid excluding worst
            centroid = simplex[:-1].mean(axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            used += 1
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                used += 1
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
                if fr < fvals[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = func(xc)
                    used += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            used += 1
                else:
                    # Shrink
                    for i in range(1, n):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        used += 1
                
                if used >= budget:
                    break
        
        order = np.argsort(fvals)
        return simplex[order[0]], fvals[order[0]], used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        # Phase 1: LHS global exploration
        n_lhs = 50
        lhs_points = self._lhs(n_lhs, dim, rng)
        lhs_fvals = np.array([func(p) for p in lhs_points])
        budget -= n_lhs
        
        # Find best from LHS
        best_idx = np.argmin(lhs_fvals)
        best_x = lhs_points[best_idx].copy()
        best_f = lhs_fvals[best_idx]
        
        # Phase 2: Multi-start Nelder-Mead from top diverse points
        # Select top 5 diverse points
        selected = []
        selected_f = []
        for idx in np.argsort(lhs_fvals):
            if len(selected) >= 5:
                break
            # Check diversity
            is_diverse = True
            for sx in selected:
                if np.linalg.norm(lhs_points[idx] - sx) < 1.0:
                    is_diverse = False
                    break
            if is_diverse:
                selected.append(lhs_points[idx].copy())
                selected_f.append(lhs_fvals[idx])
        
        # If not enough diverse points, add remaining best
        for idx in np.argsort(lhs_fvals):
            if len(selected) >= 5:
                break
            if lhs_points[idx] not in selected:
                selected.append(lhs_points[idx].copy())
                selected_f.append(lhs_fvals[idx])
        
        # Run Nelder-Mead from each selected point
        nm_budget = max(10, budget // len(selected))
        
        for i, x0 in enumerate(selected):
            if time.time() > deadline:
                break
            if budget <= 0:
                break
            
            x_opt, f_opt, used = self._nelder_mead(func, x0, nm_budget, rng)
            budget -= used
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
        
        # Phase 3: Final polish with remaining budget
        if budget > 10 and time.time() < deadline:
            x_opt, f_opt, used = self._nelder_mead(func, best_x, budget, rng)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
