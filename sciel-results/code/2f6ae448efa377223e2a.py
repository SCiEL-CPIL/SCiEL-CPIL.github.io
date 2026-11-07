"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer combining hypervolume-aware selection with L-BFGS-B refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, pts, ref_2):
        """
        Compute hypervolume of 2D points relative to ref_2.
        pts: (N, 2) array
        ref_2: (2,) array
        """
        if len(pts) == 0:
            return 0.0
        msk = np.all(pts < ref_2, axis=1)
        p = pts[msk]
        if len(p) == 0:
            return 0.0
        p = p[np.argsort(p[:, 0])]
        n_p = len(p)
        hv = 0.0
        for i in range(n_p):
            next_x = p[i+1, 0] if i + 1 < n_p else ref_2[0]
            width = next_x - p[i, 0]
            height = ref_2[1] - p[i, 1]
            hv += width * height
        return hv

    def _hv_3d(self, pts, ref):
        """
        Compute hypervolume of 3D points relative to ref.
        pts: (N, 3) array
        ref: (3,) array
        """
        if len(pts) == 0:
            return 0.0
        msk = np.all(pts < ref, axis=1)
        p = pts[msk]
        if len(p) == 0:
            return 0.0
        p = p[np.argsort(p[:, 0])]
        n_p = len(p)
        hv = 0.0
        for i in range(n_p):
            next_f0 = p[i+1, 0] if i + 1 < n_p else ref[0]
            width = next_f0 - p[i, 0]
            # 2D HV of points[i:] projected to f1, f2
            sub_pts = p[i:, 1:]
            area = self._hv_2d(sub_pts, ref[1:])
            hv += width * area
        return hv

    def _hv_contributions(self, pts, ref):
        """
        Compute HV contribution of each point.
        pts: (N, 3) array
        ref: (3,) array
        Returns: (N,) array of contributions
        """
        n = len(pts)
        if n == 0:
            return np.array([])
        
        mask = np.all(pts < ref, axis=1)
        if not np.any(mask):
            return np.zeros(n)
        
        contrib = np.zeros(n)
        full_hv = self._hv_3d(pts, ref)
        
        # For small N, removal method is fine
        for i in range(n):
            if not mask[i]:
                contrib[i] = 0.0
                continue
            mask_remove = np.ones(n, dtype=bool)
            mask_remove[i] = False
            hv_without = self._hv_3d(pts[mask_remove], ref)
            contrib[i] = full_hv - hv_without
        
        return contrib

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])
        
        used = 0
        solutions = []  # list of (x, f)
        
        def eval_point(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            used += 1
            return x, f
        
        def time_check():
            return time.time() > deadline
        
        # Phase 1: Latin Hypercube Sampling for initial spread
        n_lhs = 40
        X_lhs = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(n_lhs)
            X_lhs[:, d] = (perm + rng.random(n_lhs)) / n_lhs
        X_lhs = lo + X_lhs * (hi - lo)
        
        for x in X_lhs:
            if used >= self.budget or time_check():
                break
            x_c, f = eval_point(x)
            solutions.append((x_c, f))
        
        # Phase 2: Hypervolume-guided evolutionary search with L-BFGS-B refinement
        # We maintain a population and use HV contributions for selection
        
        pop_size = 30
        # Initialize population from existing solutions + random
        if len(solutions) < pop_size:
            while len(solutions) < pop_size and used < self.budget and not time_check():
                x0 = lo + rng.random(self.dim) * (hi - lo)
                x0, f0 = eval_point(x0)
                solutions.append((x0, f0))
        
        # Keep a fixed-size population, replacing worst HV contribution points
        # We'll do a series of iterations:
        # 1. Select parent based on HV contribution (or random if all non-dominated)
        # 2. Generate child via perturbation or L-BFGS-B
        # 3. Evaluate child
        # 4. If child is non-dominated, add to population
        # 5. If population exceeds max size, remove point with lowest HV contribution
        
        max_pop = 50
        n_iterations = 0
        
        while used < self.budget and not time_check():
            n_iterations += 1
            
            # Get current population
            X_pop = np.array([x for x, _ in solutions])
            F_pop = np.array([f for _, f in solutions])
            n_pop = len(X_pop)
            
            if n_pop == 0:
                break
            
            # Identify non-dominated points
            dominated = np.zeros(n_pop, dtype=bool)
            for i in range(n_pop):
                for j in range(n_pop):
                    if i != j:
                        if np.all(F_pop[j] <= F_pop[i]) and np.any(F_pop[j] < F_pop[i]):
                            dominated[i] = True
                            break
            
            nd_idx = np.where(~dominated)[0]
            
            # Select parent
            if len(nd_idx) > 0:
                # Use HV contribution to select parent (higher contribution = better)
                contribs = self._hv_contributions(F_pop, ref)
                # Only consider non-dominated
                nd_contribs = contribs[nd_idx]
                # Weighted random selection based on contribution
                if np.sum(nd_contribs) > 0:
                    probs = nd_contribs / np.sum(nd_contribs)
                    parent_nd_idx = nd_idx[rng.choice(len(nd_idx), p=probs)]
                else:
                    parent_nd_idx = nd_idx[rng.integers(len(nd_idx))]
            else:
                parent_nd_idx = rng.integers(n_pop)
            
            parent_x = X_pop[parent_nd_idx].copy()
            
            # Generate child
            # Strategy: 50% perturbation, 50% L-BFGS-B with random weight
            if rng.random() < 0.5 and used < self.budget - 10:
                # Perturbation
                scale = 0.1 * (hi - lo)
                child_x = parent_x + scale * rng.standard_normal(self.dim)
                child_x = np.clip(child_x, lo, hi)
                child_x, child_f = eval_point(child_x)
            else:
                # L-BFGS-B with random weight
                w = rng.dirichlet(np.ones(self.n_obj))
                
                def objective(x):
                    nonlocal used
                    x = np.clip(x, lo, hi)
                    f = np.array(problem(x))
                    used += 1
                    return np.dot(w, f)
                
                bounds_scipy = list(zip(lo, hi))
                try:
                    res = minimize(
                        objective,
                        parent_x,
                        method='L-BFGS-B',
                        bounds=bounds_scipy,
                        options={'maxiter': 30, 'maxfun': 20}
                    )
                    if time_check() or used >= self.budget:
                        break
                    child_x = np.clip(res.x, lo, hi)
                    child_x, child_f = eval_point(child_x)
                except Exception:
                    continue
            
            # Add child to population
            solutions.append((child_x, child_f))
            
            # If population exceeds max size, remove point with lowest HV contribution
            if len(solutions) > max_pop:
                X_pop = np.array([x for x, _ in solutions])
                F_pop = np.array([f for _, f in solutions])
                contribs = self._hv_contributions(F_pop, ref)
                # Remove point with lowest contribution (index 0 of sorted)
                min_idx = np.argmin(contribs)
                solutions.pop(min_idx)
        
        # Phase 3: Final refinement of Pareto front
        if used < self.budget and not time_check():
            X_pop = np.array([x for x, _ in solutions])
            F_pop = np.array([f for _, f in solutions])
            n_pop = len(X_pop)
            
            if n_pop > 0:
                # Identify non-dominated points
                dominated = np.zeros(n_pop, dtype=bool)
                for i in range(n_pop):
                    for j in range(n_pop):
                        if i != j:
                            if np.all(F_pop[j] <= F_pop[i]) and np.any(F_pop[j] < F_pop[i]):
                                dominated[i] = True
                                break
                
                nd_idx = np.where(~dominated)[0]
                
                # For each non-dominated point, do a few local L-BFGS-B runs
                for idx in nd_idx:
                    if used >= self.budget or time_check():
                        break
                    
                    x_center = X_pop[idx]
                    
                    for _ in range(2):
                        if used >= self.budget or time_check():
                            break
                        
                        w = rng.dirichlet(np.ones(self.n_obj))
                        
                        def objective(x):
                            nonlocal used
                            x = np.clip(x, lo, hi)
                            f = np.array(problem(x))
                            used += 1
                            return np.dot(w, f)
                        
                        bounds_scipy = list(zip(lo, hi))
                        try:
                            res = minimize(
                                objective,
                                x_center,
                                method='L-BFGS-B',
                                bounds=bounds_scipy,
                                options={'maxiter': 30, 'maxfun': 20}
                            )
                            if time_check() or used >= self.budget:
                                break
                            x_res = np.clip(res.x, lo, hi)
                            x_res, f_res = eval_point(x_res)
                            solutions.append((x_res, f_res))
                        except Exception:
                            pass
        
        # Return the last solution (return value is ignored by evaluator)
        if len(solutions) > 0:
            return solutions[-1][0]
        return lo.copy()
# EVOLVE-BLOCK-END
